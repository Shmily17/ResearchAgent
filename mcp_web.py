import asyncio
import platform
import sys
import logging
import os
import json
import re
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
import urllib.parse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.client import MCPClient, STUDENT_EMAILS
from src.utils import STUDENT_PAPERS_DIR, ensure_dirs_exist, get_student_id_for_path

if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

templates = Jinja2Templates(directory=str(SRC_DIR / "templates"))
app_mcp_client: Optional[MCPClient] = None
log_queue = asyncio.Queue(maxsize=200)

class QueueHandler(logging.Handler):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def emit(self, record):
        log_entry = self.format(record)
        try:
            self.queue.put_nowait(log_entry)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(log_entry)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

queue_handler = QueueHandler(log_queue)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s (%(name)s)')
queue_handler.setFormatter(formatter)
queue_handler.setLevel(logging.INFO)

root_logger = logging.getLogger()
root_logger.addHandler(queue_handler)
root_logger.setLevel(logging.INFO)

logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)
logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global app_mcp_client
    
    await log_queue.put("FastAPI App: Starting up...")
    temp_client = None
    try:
        ensure_dirs_exist()
        await log_queue.put("FastAPI App: Initializing MCPClient...")
        temp_client = MCPClient()
        
        server_script_path = str(SRC_DIR / "server.py")

        if not Path(server_script_path).exists():
            msg = f"FastAPI App: MCP Server script not found at {server_script_path}."
            await log_queue.put(msg)
            raise RuntimeError(msg)
        else:
            await log_queue.put(f"FastAPI App: Connecting to server script: {server_script_path}")
            await temp_client.connect_to_server(server_script_path)
            await log_queue.put("FastAPI App: Successfully connected to server script.")
        
        app_mcp_client = temp_client
        await log_queue.put("FastAPI App: MCPClient assigned globally and ready.")
        
    except Exception as e:
        error_msg = f"FastAPI App: CRITICAL Error during MCPClient startup: {type(e).__name__} - {str(e)}"
        logging.critical("FastAPI App: MCPClient startup failed catastrophically.", exc_info=True)
        await log_queue.put(error_msg)
        if temp_client and hasattr(temp_client, 'cleanup'):
            try:
                await temp_client.cleanup()
            except Exception as cleanup_err:
                cleanup_failure_msg = f"FastAPI App: Error during cleanup after failed startup: {cleanup_err}"
                logging.error(cleanup_failure_msg, exc_info=True)
                await log_queue.put(cleanup_failure_msg)
        raise RuntimeError(f"MCPClient failed to initialize during application startup: {e}") from e

    yield

    await log_queue.put("FastAPI App: Shutting down...")
    if app_mcp_client and hasattr(app_mcp_client, 'cleanup'):
        await log_queue.put("FastAPI App: Cleaning up MCPClient...")
        try:
            await app_mcp_client.cleanup()
            await log_queue.put("FastAPI App: MCPClient cleaned up successfully.")
        except Exception as e_cleanup:
            error_msg_cleanup = f"FastAPI App: Error during MCPClient cleanup: {type(e_cleanup).__name__} - {str(e_cleanup)}"
            logging.error("FastAPI App: MCPClient cleanup failed.", exc_info=True)
            await log_queue.put(error_msg_cleanup)
    else:
        await log_queue.put("FastAPI App: No MCPClient instance to clean up.")
    await log_queue.put("FastAPI App: Shutdown complete.")

app = FastAPI(title="科研智能体API", lifespan=lifespan)

class GeneralQueryRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "student_emails_json": json.dumps(STUDENT_EMAILS, ensure_ascii=False)
    })

@app.post("/api/chat")
async def process_general_chat_query(request_data: GeneralQueryRequest):
    if not app_mcp_client:
        raise HTTPException(status_code=503, detail="服务尚未准备就绪，请稍后重试。")
    
    query = request_data.query
    if not query:
        raise HTTPException(status_code=400, detail="查询内容不能为空。")

    await log_queue.put(f"API Request: /api/chat - Query: {query[:100]}...")
    try:
        ai_response = await app_mcp_client.process_query(query)
        await log_queue.put(f"API Response: /api/chat - AI Response: {ai_response[:100]}...")
        return JSONResponse(content={"response": ai_response})
    except Exception as e:
        error_detail = f"处理查询时发生错误: {str(e)}"
        await log_queue.put(f"API Error: /api/chat - {error_detail}")
        logging.error(f"Error processing general query '{query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="处理您的请求时发生内部错误。")

@app.post("/trigger-research/{student_name}")
async def trigger_research(student_name: str):
    if not app_mcp_client:
        raise HTTPException(status_code=503, detail="服务尚未准备就绪，请稍后重试。")
    if student_name not in STUDENT_EMAILS:
        raise HTTPException(status_code=404, detail=f"学生 '{student_name}' 未在配置中找到。")

    await log_queue.put(f"API Request: /trigger-research - Student: {student_name}")
    try:
        status_message = await app_mcp_client.process_research_request(student_name)
        await log_queue.put(f"API Info: /trigger-research - Student '{student_name}' processing complete. Status: {status_message}")

        student_id_for_path = get_student_id_for_path(student_name)
        student_folder = STUDENT_PAPERS_DIR / student_id_for_path
        report_filename = None

        if student_folder.is_dir():
            txt_files = sorted(student_folder.glob("recommended_papers_*.txt"), key=os.path.getmtime, reverse=True)
            if txt_files:
                report_filename = txt_files[0].name
                await log_queue.put(f"API Info: Found report for '{student_name}': {report_filename}")
        
        return {
            "message": status_message,
            "report_filename": report_filename,
            "student_id_for_path": student_id_for_path
        }
    except Exception as e:
        error_detail = f"处理文献推荐时出错: {str(e)}"
        await log_queue.put(f"API Error: /trigger-research - {error_detail}")
        logging.error(f"Error processing research request for '{student_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="处理文献推荐请求时发生内部错误。")

@app.get("/stream-logs")
async def stream_logs(request: Request):
    async def log_generator():
        yield f"data: Successfully connected to log stream. Client: {request.client.host}\n\n"
        
        while True:
            try:
                if await request.is_disconnected():
                    logging.info(f"Log stream client {request.client.host} disconnected.")
                    break
                
                log_entry = await asyncio.wait_for(log_queue.get(), timeout=25)
                yield f"data: {log_entry}\n\n"
                log_queue.task_done()
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
            except asyncio.CancelledError:
                logging.info(f"Log stream task for {request.client.host} cancelled.")
                break
            except Exception as e:
                logging.warning(f"Error in log_generator for {request.client.host}: {e}", exc_info=False)
                try:
                    yield f"data: [LOG_STREAM_ERROR] {str(e)}\n\n"
                except Exception:
                    pass
                await asyncio.sleep(1)
    
    return StreamingResponse(log_generator(), media_type="text/event-stream",
                             headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'X-Accel-Buffering': 'no'})

@app.get("/get-report/{student_id_for_path}/{filename}")
async def get_report_file(student_id_for_path: str, filename: str):
    decoded_filename = urllib.parse.unquote(filename)

    if ".." in decoded_filename or decoded_filename.startswith(("/", "\\")):
        raise HTTPException(status_code=400, detail="无效的文件名。")
    
    file_path = (STUDENT_PAPERS_DIR / student_id_for_path / decoded_filename).resolve()

    if not file_path.is_relative_to(STUDENT_PAPERS_DIR.resolve()):
        raise HTTPException(status_code=403, detail="禁止访问。")

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"报告文件 '{decoded_filename}' 未找到。")
    
    return FileResponse(
        path=str(file_path),
        media_type='text/plain', 
        filename=decoded_filename
    )

if __name__ == "__main__":
    print(f"Web UI should be accessible at: http://127.0.0.1:8001")
    uvicorn.run("mcp_web:app", host="127.0.0.1", port=8001, reload=False, log_level="info")
