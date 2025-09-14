import asyncio
import logging
import os
import json
from pathlib import Path
import sys
from typing import Optional, List, Dict
from contextlib import AsyncExitStack
from datetime import datetime
import re
from openai import OpenAI
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import subprocess
import mcp.types as mcp_types
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()

STUDENT_EMAILS = {

}
RAG_KEYWORDS = []


class MCPClient:

    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.openai_api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")
        if not self.openai_api_key:
            raise ValueError("❌ 未找到 OpenAI API Key，请在 .env 文件中设置 DASHSCOPE_API_KEY")
        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)
        self.sessions = {} 
        
        self.mcp_config = self._load_mcp_config()

    def _load_mcp_config(self):
        try:
            mcp_json_path = Path(r"E:\实习\MCP\MCP_Demo\mcp-project\mcp.json") # 或者你的绝对路径
            with open(mcp_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 无法加载 mcp.json: {e}")
            return {"mcpServers": {}}

    async def connect_to_server(self, server_script_path: str):
        try:
            logging.info(f"正在连接到服务器: {server_script_path}")
            
            is_python = server_script_path.endswith('.py')
            is_js = server_script_path.endswith('.js')
            if not (is_python or is_js):
                raise ValueError("服务器脚本必须是 .py 或 .js 文件")

            command = sys.executable if is_python else "node" # 使用 sys.executable 保证是当前Python解释器
            logging.info(f"使用命令: {command}")

            effective_env = os.environ.copy()
            if is_python:
                effective_env["PYTHONIOENCODING"] = "utf-8"
                effective_env["PYTHONUTF8"] = "1" 
            
            script_dir = os.path.dirname(os.path.abspath(server_script_path))

            server_params = StdioServerParameters(
                command=command, 
                args=[server_script_path], 
                env=effective_env,
                cwd=script_dir
            )
            logging.info(f"正在启动服务进程... CWD: {script_dir}")
            
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            logging.info("服务进程已启动")

            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            logging.info("会话已初始化")

            server_name = os.path.splitext(os.path.basename(server_script_path))[0]
            self.sessions[server_name] = session

            response = await session.list_tools()
            tools = response.tools
            logging.info(f"已连接到服务器 {server_name}，可用工具: {[tool.name for tool in tools]}")

        except Exception as e:
            logging.error(f"连接服务器失败: {str(e)}", exc_info=True)
            raise

    async def connect_to_mcp_services(self):
        logging.info("开始连接MCP服务...")
        for name, config in self.mcp_config["mcpServers"].items():
            if not config.get("isActive", False):
                logging.info(f"跳过未激活的服务: {name}")
                continue

            try:
                logging.info(f"正在连接服务: {name}")

                effective_env = os.environ.copy()
                cmd_list = config["command"] if isinstance(config["command"], list) else [config["command"]]
                cmd_lower = cmd_list[0].lower() if cmd_list else ""
                
                if "python" in cmd_lower:
                    effective_env["PYTHONIOENCODING"] = "utf-8"
                    effective_env["PYTHONUTF8"] = "1"
                
                if config.get("env"):
                    effective_env.update(config["env"])

                mcp_json_dir = Path(r"E:\实习\MCP\MCP_Demo\mcp-project") # mcp.json 所在目录
                cwd = mcp_json_dir # 默认 cwd 为 mcp.json 所在目录
                if "cwd" in config:
                    config_cwd = Path(config["cwd"])
                    if not config_cwd.is_absolute():
                        cwd = (mcp_json_dir / config_cwd).resolve()
                    else:
                        cwd = config_cwd
                
                logging.info(f"服务 {name} 的 CWD 设置为: {cwd}")

                server_params = StdioServerParameters(
                    command=cmd_list[0], # command 是可执行文件
                    args=cmd_list[1:] + config.get("args", []), # args 是命令的参数
                    env=effective_env,
                    cwd=str(cwd) # 传递 cwd
                )

                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                stdio, write = stdio_transport
                logging.info(f"服务 {name} 进程已启动")

                session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
                await session.initialize()
                logging.info(f"服务 {name} 会话已初始化")

                self.sessions[name] = session

                response = await session.list_tools()
                mcp_tools = response.tools
                logging.info(f"已连接到服务 {name}，可用工具: {[tool.name for tool in mcp_tools]}")

            except Exception as e:
                logging.error(f"连接服务 {name} 失败: {str(e)}", exc_info=True)

    async def _run_rag_processor(self, student_name: str, query: str) -> str:
        try:
            rag_script_path = Path(__file__).resolve().parent / "rag_processor.py"
            if not rag_script_path.exists():
                logging.error(f"RAG处理脚本未找到: {rag_script_path}")
                return "抱歉，学习规划助手组件缺失，无法处理您的请求。"

            logging.info(f"调用RAG脚本: python {rag_script_path} \"{student_name}\" \"{query}\"")
            
            current_env = os.environ.copy()
            script_dir = str(Path(__file__).resolve().parent)
            if "PYTHONPATH" in current_env:
                current_env["PYTHONPATH"] = f"{script_dir}{os.pathsep}{current_env['PYTHONPATH']}"
            else:
                current_env["PYTHONPATH"] = script_dir
            current_env["PYTHONIOENCODING"] = "utf-8" # 确保子进程输出也是UTF-8
            current_env["PYTHONUTF8"] = "1"


            process = await asyncio.create_subprocess_exec(
                sys.executable, str(rag_script_path), student_name, query,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=current_env,
                cwd=script_dir
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_output = stderr.decode('utf-8', errors='replace').strip()
                logging.error(f"RAG脚本执行失败 (返回码 {process.returncode}):\n{error_output}")
                return f"抱歉，学习规划助手在处理时遇到错误。详情: {error_output[:200]}" # 只显示部分错误

            result = stdout.decode('utf-8', errors='replace').strip()
            logging.info(f"RAG脚本成功执行，输出: {result[:200]}...")
            return result

    async def process_query(self, query: str) -> str:
        extracted_student_name: Optional[str] = None
        is_rag_query = False

        for student_name_key in STUDENT_EMAILS.keys():
            if student_name_key in query:
                for rag_keyword in RAG_KEYWORDS:
                    if rag_keyword in query:
                        extracted_student_name = student_name_key
                        is_rag_query = True
                        break
            if is_rag_query:
                break
        
        if is_rag_query and extracted_student_name:
            logging.info(f"检测到RAG查询，学生: {extracted_student_name}，原始查询: {query}")
            return await self._run_rag_processor(extracted_student_name, query)
        
        messages = [{"role": "user", "content": query}]
        
        available_tools = []
        try:
            logging.info(f"向大模型发送普通查询: {query}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            
            final_output = response.choices[0].message.content
            logging.info(f"大模型普通问答回复: {final_output[:100]}...")
            return final_output

        except Exception as e:
            logging.error(f"处理普通查询时出错: {e}", exc_info=True)
            return f"抱歉，处理您的提问时发生错误: {str(e)}"


    async def process_research_request(self, student_name: str) -> str:
        try:
            email = STUDENT_EMAILS.get(student_name)
            if not email:
                return f"❌ 未找到 {student_name} 的邮箱配置，请先在 STUDENT_EMAILS 中添加"

            logging.info(f"处理 {student_name} 的研究请求，邮箱: {email}")

            server_session = self.sessions.get("server") # 'server' 是 server.py 脚本的文件名（不含.py）
            if not server_session:
                logging.error("未找到 'server' 服务会话。请检查 connect_to_server 是否正确连接并命名。")
                all_sessions = list(self.sessions.keys())
                logging.info(f"当前可用会话: {all_sessions}")
                return "❌ 'server' 服务未正确连接，无法处理文献推荐。请检查服务端。"

            logging.info("调用 server.process_student_research 工具...")
            result = await server_session.call_tool( # 使用获取到的 server_session
                "process_student_research",
                {
                    "student_name": student_name,
                    "email": email
                }
            )
            logging.info(f"工具调用完成，结果类型: {type(result.content[0])}")
            
            if result.content and isinstance(result.content[0], mcp_types.TextContent):
                response_text = result.content[0].text
                logging.info(f"来自 server.process_student_research 的文本响应: {response_text[:200]}...")
                return response_text
            else:
                logging.error(f"server.process_student_research 返回了意外的格式: {result.content}")
                return "❌ 处理文献推荐时收到意外的响应格式。"

        except AttributeError as ae: # 捕获特定错误，比如sessions中没有server
             logging.error(f"处理研究请求时发生属性错误 (可能是会话问题): {str(ae)}", exc_info=True)
             return f"❌ 处理研究请求失败 (属性错误): {str(ae)}"
        except Exception as e:
            logging.error(f"处理研究请求失败: {str(e)}", exc_info=True)
            return f"❌ 处理研究请求失败: {str(e)}"

    async def chat_loop(self):
        print("\n🤖 科研智能体已启动！输入 'quit' 退出")
        print("1. 发送论文推荐：'给xxx发论文'")
        print(f"2. 进行学习规划（RAG）：包含学生姓名（如“小明”）和关键词（如“规划”）的提问，例如：“帮小明规划一下接下来的学习重点”")
        print(f"\n📋 当前支持的学生列表：{', '.join(STUDENT_EMAILS.keys())}")

        while True:
            try:
                query = input("\n请输入命令或问题：").strip() # 修改提示
                if query.lower() == 'quit':
                    break

                paper_request_match = re.match(r'^给(\w+)发论文$', query)
                if paper_request_match:
                    student_name = paper_request_match.group(1)
                    print(f"\n📚 正在处理 {student_name} 的论文推荐...")
                    response = await self.process_research_request(student_name)
                else:
                    print(f"\n💬 正在处理您的问题: {query[:50]}...")
                    response = await self.process_query(query)

                print(f"\n🤖 AI: {response}")

            except Exception as e:
                logging.error(f"处理请求时出错: {str(e)}", exc_info=True)
                print(f"❌ 处理请求时出错: {str(e)}")


    async def plan_tool_usage(self, query: str, tools: List[dict]) -> List[dict]:
        logging.warning("plan_tool_usage。")
        return []


    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    client = MCPClient()
    try:
        server_script_path = str(Path(__file__).resolve().parent / "server.py")
        
        if not Path(server_script_path).exists():
            logging.error(f"主要服务脚本 server.py 未找到于: {server_script_path}")
        else:
            await client.connect_to_server(server_script_path)
        
        await client.chat_loop()
    except Exception as e:
        logging.critical(f"MCPClient main loop encountered an unrecoverable error: {e}", exc_info=True)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info >= (3,8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
