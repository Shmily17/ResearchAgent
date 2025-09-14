# mcp_web.py

import asyncio
import platform
import sys

if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        print("INFO: Successfully set asyncio event loop policy to WindowsProactorEventLoopPolicy.", flush=True)
    except Exception as e_policy:
        print(f"WARNING: Could not set WindowsProactorEventLoopPolicy (may already be set or conflict): {e_policy}", flush=True)

import logging
import os
import json
import re
from pathlib import Path
from typing import Optional, Any
from contextlib import asynccontextmanager
import urllib.parse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles # 如果有静态文件目录
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

# 导入你的MCPClient和学生邮箱配置
from client import MCPClient, STUDENT_EMAILS # 确保 client.py 在PYTHONPATH中

# pypinyin 用于确定学生文件夹路径
from pypinyin import Style, pinyin


# --- FastAPI 应用设置 ---
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# --- MCP Client 实例 ---
app_mcp_client: Optional[MCPClient] = None

# --- 日志相关 ---
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
            # 在队列满时，尝试移除一个旧日志再添加新的，或者直接丢弃
            try:
                self.queue.get_nowait() # 移除一个
                self.queue.put_nowait(log_entry) # 添加新的
            except asyncio.QueueEmpty: # 如果在尝试移除时队列又空了
                pass
            except asyncio.QueueFull: # 如果移除后尝试添加时又满了（并发情况）
                print(f"Log queue still full after trying to make space. Dropping: {log_entry}", file=sys.stderr)


queue_handler = QueueHandler(log_queue)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s (%(name)s)')
queue_handler.setFormatter(formatter)
queue_handler.setLevel(logging.INFO)

# 配置根日志记录器
root_logger = logging.getLogger()
root_logger.addHandler(queue_handler)
root_logger.setLevel(logging.INFO) # 全局日志级别

# 配置特定库的日志级别（如果需要）
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)
logging.getLogger("multipart").setLevel(logging.WARNING) # httpx/multipart 日志可能很冗余
logging.getLogger("httpcore").setLevel(logging.WARNING)


# --- FastAPI Lifespan 事件处理 ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global app_mcp_client
    
    await log_queue.put("FastAPI App (lifespan): Starting up...")
    temp_client = None
    try:
        await log_queue.put("FastAPI App (lifespan): Initializing MCPClient...")
        temp_client = MCPClient()
        
        # server.py 的路径
        current_file_dir = Path(__file__).resolve().parent
        server_script_path = str(current_file_dir / "server.py") # 假设 server.py 与 mcp_web.py 同级
        # 如果 server.py 在特定项目目录:
        # server_script_path = r"E:\实习\MCP\MCP_Demo\mcp-project\server.py"

        if not Path(server_script_path).exists():
            msg = f"FastAPI App (lifespan): MCP Server script not found at {server_script_path}. Cannot connect."
            print(msg, file=sys.stderr)
            await log_queue.put(msg)
            # 根据需要，如果 server.py 是核心依赖，这里应该 raise RuntimeError
            # raise RuntimeError(msg) 
        else:
            await log_queue.put(f"FastAPI App (lifespan): Connecting to server script: {server_script_path}")
            await temp_client.connect_to_server(server_script_path)
            await log_queue.put("FastAPI App (lifespan): Successfully connected to server script.")

        # 可选: 连接 mcp.json 中定义的服务 (如果 client.py 中 connect_to_mcp_services 被启用)
        # await log_queue.put("FastAPI App (lifespan): Connecting to MCP JSON services...")
        # await temp_client.connect_to_mcp_services()
        # await log_queue.put("FastAPI App (lifespan): Successfully connected to MCP JSON services.")
        
        app_mcp_client = temp_client
        await log_queue.put("FastAPI App (lifespan): MCPClient assigned globally and ready.")
        
    except Exception as e:
        error_msg = f"FastAPI App (lifespan): CRITICAL Error during MCPClient startup: {type(e).__name__} - {str(e)}"
        print(error_msg, file=sys.stderr)
        logging.critical("FastAPI App (lifespan): MCPClient startup failed catastrophically.", exc_info=True)
        await log_queue.put(error_msg) # 也将错误放入日志队列
        if temp_client and hasattr(temp_client, 'cleanup'):
            try:
                await temp_client.cleanup()
            except Exception as cleanup_err:
                cleanup_failure_msg = f"FastAPI App (lifespan): Error during cleanup after failed startup: {cleanup_err}"
                logging.error(cleanup_failure_msg, exc_info=True)
                await log_queue.put(cleanup_failure_msg)
        raise RuntimeError(f"MCPClient failed to initialize during application startup: {e}") from e

    yield  # 服务运行阶段

    # --- Shutdown ---
    await log_queue.put("FastAPI App (lifespan): Shutting down...")
    if app_mcp_client and hasattr(app_mcp_client, 'cleanup'):
        await log_queue.put("FastAPI App (lifespan): Cleaning up MCPClient...")
        try:
            await app_mcp_client.cleanup()
            await log_queue.put("FastAPI App (lifespan): MCPClient cleaned up successfully.")
        except Exception as e_cleanup:
            error_msg_cleanup = f"FastAPI App (lifespan): Error during MCPClient cleanup: {type(e_cleanup).__name__} - {str(e_cleanup)}"
            print(error_msg_cleanup, file=sys.stderr)
            logging.error("FastAPI App (lifespan): MCPClient cleanup failed.", exc_info=True)
            await log_queue.put(error_msg_cleanup)
    else:
        await log_queue.put("FastAPI App (lifespan): No MCPClient instance to clean up or already cleaned up.")
    await log_queue.put("FastAPI App (lifespan): Shutdown complete.")


app = FastAPI(title="科研智能体API", lifespan=lifespan)

# 如果你的 CSS/JS 文件在 static 文件夹，可以挂载
# app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# --- 辅助函数 ---
def get_student_id_for_path(student_name: str) -> str:
    pinyin_list = pinyin(student_name, style=Style.NORMAL)
    sanitized_name_parts = [item[0] for item in pinyin_list]
    sanitized_id = "_".join(sanitized_name_parts)
    sanitized_id = re.sub(r'[^a-zA-Z0-9_]', '', sanitized_id)
    sanitized_id = re.sub(r'_+', '_', sanitized_id).strip('_')
    if not sanitized_id:
        sanitized_id = "default_student"
    return sanitized_id

# --- API 端点 ---
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "student_emails_json": json.dumps(STUDENT_EMAILS, ensure_ascii=False) # ensure_ascii=False for Chinese names
    })

class GeneralQueryRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def process_general_chat_query(request_data: GeneralQueryRequest):
    if not app_mcp_client:
        await log_queue.put("API Error: /api/chat called but MCPClient not initialized.")
        raise HTTPException(status_code=503, detail="服务尚未准备就绪，请稍后重试。")
    
    query = request_data.query
    if not query:
        raise HTTPException(status_code=400, detail="查询内容不能为空。")

    await log_queue.put(f"API Request: /api/chat - Query: {query[:100]}...")
    try:
        # process_query in client.py now handles RAG vs. general LLM call
        ai_response = await app_mcp_client.process_query(query)
        await log_queue.put(f"API Response: /api/chat - AI Response: {ai_response[:100]}...")
        return JSONResponse(content={"response": ai_response})
    except Exception as e:
        error_detail = f"处理查询 '{query[:50]}...' 时发生错误: {str(e)}"
        await log_queue.put(f"API Error: /api/chat - {error_detail}")
        logging.error(f"Error processing general query '{query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理您的请求时发生内部错误。请查看服务器日志。")


@app.post("/trigger-research/{student_name}")
async def trigger_research(student_name: str):
    if not app_mcp_client:
        await log_queue.put("API Error: /trigger-research called but MCPClient not initialized.")
        raise HTTPException(status_code=503, detail="服务尚未准备就绪，请稍后重试。")
    if student_name not in STUDENT_EMAILS:
        await log_queue.put(f"API Error: /trigger-research - Student '{student_name}' not in STUDENT_EMAILS.")
        raise HTTPException(status_code=404, detail=f"学生 '{student_name}' 未在配置中找到。")

    await log_queue.put(f"API Request: /trigger-research - Student: {student_name}")
    try:
        # process_research_request in client.py calls the 'server' MCP tool
        status_message = await app_mcp_client.process_research_request(student_name)
        await log_queue.put(f"API Info: /trigger-research - Student '{student_name}' processing complete. Status: {status_message}")

        student_id_for_path = get_student_id_for_path(student_name)
        # 基础目录应该与 server.py 中 StudentResearchManager 使用的目录一致
        student_folder_base = Path("./student_papers") #  Assumes mcp_web.py runs from project root
        student_folder = student_folder_base / student_id_for_path
        report_filename = None

        if student_folder.exists() and student_folder.is_dir():
            # server.py's process_student_research saves files like "recommended_papers_{student_id}_{timestamp}.txt"
            txt_files = sorted(
                student_folder.glob("recommended_papers_*.txt"), 
                key=os.path.getmtime, 
                reverse=True
            )
            if txt_files:
                latest_file = txt_files[0]
                report_filename = latest_file.name
                await log_queue.put(f"API Info: /trigger-research - Found report for '{student_name}': {report_filename}")
            else:
                await log_queue.put(f"API Info: /trigger-research - No 'recommended_papers_*.txt' files found in {student_folder}")
        else:
            await log_queue.put(f"API Info: /trigger-research - Student folder {student_folder} does not exist.")
        
        return {
            "message": status_message,
            "report_filename": report_filename, # This can be None if no file found
            "student_id_for_path": student_id_for_path # Always return this for path construction
        }
    except Exception as e:
        error_detail = f"处理 '{student_name}' 的文献推荐时出错: {str(e)}"
        await log_queue.put(f"API Error: /trigger-research - {error_detail}")
        logging.error(f"Error processing research request for '{student_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理文献推荐请求时发生内部错误。请查看服务器日志。")

@app.get("/stream-logs")
async def stream_logs(request: Request): # Add request for client disconnect detection
    async def log_generator():
        # Send an initial message to confirm connection
        yield f"data: Successfully connected to log stream. Client: {request.client.host}\n\n"
        
        # Send existing logs in queue first (optional, up to a limit)
        # initial_logs_count = 0
        # while not log_queue.empty() and initial_logs_count < 50: # Send up to 50 old logs
        #     try:
        #         log_entry = log_queue.get_nowait()
        #         yield f"data: {log_entry}\n\n"
        #         log_queue.task_done()
        #         initial_logs_count += 1
        #     except asyncio.QueueEmpty:
        #         break
        # if initial_logs_count > 0:
        #    yield f"data: --- End of initial logs ---\n\n"

        while True:
            try:
                # Check for client disconnect
                if await request.is_disconnected():
                    logging.info(f"Log stream client {request.client.host} disconnected.")
                    await log_queue.put(f"Log stream client {request.client.host} disconnected.")
                    break
                
                log_entry = await asyncio.wait_for(log_queue.get(), timeout=25) # 25s timeout for keep-alive
                yield f"data: {log_entry}\n\n"
                log_queue.task_done()
            except asyncio.TimeoutError:
                # Send a keep-alive comment to prevent connection closure by intermediaries
                yield ": keep-alive\n\n"
            except asyncio.CancelledError:
                logging.info(f"Log stream task for {request.client.host} cancelled.")
                await log_queue.put(f"Log stream task for {request.client.host} cancelled.")
                break
            except Exception as e:
                # Log error, send error to client if possible, and continue
                logging.warning(f"Error in log_generator for {request.client.host}: {e}", exc_info=False)
                try:
                    yield f"data: [LOG_STREAM_ERROR] Error in log stream: {str(e)}\n\n"
                except Exception: # If yielding itself fails (e.g., client gone)
                    pass
                await asyncio.sleep(1) # Avoid tight loop on persistent errors
    
    return StreamingResponse(log_generator(), media_type="text/event-stream",
                             headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'X-Accel-Buffering': 'no'})


@app.get("/get-report/{student_id_for_path}/{filename}")
async def get_report_file(student_id_for_path: str, filename: str):
    decoded_filename = urllib.parse.unquote(filename)

    # Basic security check
    if ".." in decoded_filename or decoded_filename.startswith("/") or decoded_filename.startswith("\\"):
        await log_queue.put(f"API Warning: Invalid report filename pattern: {decoded_filename}")
        raise HTTPException(status_code=400, detail="无效的文件名。")
    
    # Base directory for student papers, relative to where mcp_web.py is running
    base_report_dir = Path("./student_papers") 
    
    # Construct the full path
    # student_id_for_path should be sanitized from get_student_id_for_path
    # decoded_filename is the actual file name
    file_path = (base_report_dir / student_id_for_path / decoded_filename).resolve()

    # Security: Ensure the resolved path is still within the intended base_report_dir
    if not file_path.is_relative_to(base_report_dir.resolve()):
        await log_queue.put(f"API Security Alert: Attempt to access file outside report directory: {file_path}")
        raise HTTPException(status_code=403, detail="禁止访问。")

    if not file_path.exists() or not file_path.is_file():
        await log_queue.put(f"API Warning: Report file not found: {file_path}")
        raise HTTPException(status_code=404, detail=f"报告文件 '{decoded_filename}' 未找到。")
    
    await log_queue.put(f"API Info: Serving report file: {file_path}")
    return FileResponse(
        path=str(file_path), # FileResponse prefers string path
        media_type='text/plain', 
        filename=decoded_filename # This suggests filename for download dialog
    )


# --- 运行 FastAPI 应用 ---
if __name__ == "__main__":
    # Ensure necessary directories exist relative to this script's location
    current_dir = Path(__file__).resolve().parent
    (current_dir / "templates").mkdir(parents=True, exist_ok=True)
    (current_dir / "student_papers").mkdir(parents=True, exist_ok=True)
    (current_dir / "llm_outputs").mkdir(parents=True, exist_ok=True) # Used by client.py (if it saves outputs)
    (current_dir / "chroma_db").mkdir(parents=True, exist_ok=True) # Used by server.py
    (current_dir / "chroma_db_rag").mkdir(parents=True, exist_ok=True) # Used by rag_processor.py
    
    print(f"Web UI should be accessible at: http://127.0.0.1:8001") # Ensure port matches uvicorn.run
    print(f"Ensure '{current_dir / 'templates' / 'index.html'}' exists.")

    uvicorn.run("mcp_web:app", host="127.0.0.1", port=8001, reload=False, log_level="info")