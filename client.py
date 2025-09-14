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
import subprocess # <<<< 新增subprocess导入
import mcp.types as mcp_types
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()

# 学生邮箱配置
STUDENT_EMAILS = {
    "小明": "1453301028@qq.com",
    "小红": "1453301028@qq.com",
    "小华": "xiaohua@example.com",
    "罗伟源": "1054633506@qq.com"
    # 可以添加更多学生
}
# RAG 触发关键词
RAG_KEYWORDS = ["规划", "学习情况", "总结", "下一步", "分析一下", "建议"]


class MCPClient:

    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.openai_api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")
        if not self.openai_api_key:
            raise ValueError("❌ 未找到 OpenAI API Key，请在 .env 文件中设置 DASHSCOPE_API_KEY")
        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)
        self.sessions = {}  # 存储多个服务会话
        
        # 加载 MCP 服务配置
        self.mcp_config = self._load_mcp_config()

    def _load_mcp_config(self):
        """加载 mcp.json 配置文件"""
        try:
            # 确保使用正确的相对路径或绝对路径
            # mcp_json_path = Path(__file__).resolve().parent / "mcp.json" # 如果 mcp.json 在同级
            mcp_json_path = Path(r"E:\实习\MCP\MCP_Demo\mcp-project\mcp.json") # 或者你的绝对路径
            with open(mcp_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 无法加载 mcp.json: {e}")
            return {"mcpServers": {}}

    async def connect_to_server(self, server_script_path: str):
        """连接到自定义 Python/JS 服务器"""
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
            
            # 获取 server_script_path 所在的目录
            script_dir = os.path.dirname(os.path.abspath(server_script_path))

            server_params = StdioServerParameters(
                command=command, 
                args=[server_script_path], 
                env=effective_env,
                cwd=script_dir # <<<< 关键：设置工作目录为脚本所在目录
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
        """连接到 mcp.json 中定义的服务"""
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

                # 处理 cwd
                mcp_json_dir = Path(r"E:\实习\MCP\MCP_Demo\mcp-project") # mcp.json 所在目录
                cwd = mcp_json_dir # 默认 cwd 为 mcp.json 所在目录
                if "cwd" in config:
                    # 如果 cwd 是相对路径，则相对于 mcp.json 目录
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
        """调用 rag_processor.py 脚本"""
        try:
            rag_script_path = Path(__file__).resolve().parent / "rag_processor.py"
            if not rag_script_path.exists():
                logging.error(f"RAG处理脚本未找到: {rag_script_path}")
                return "抱歉，学习规划助手组件缺失，无法处理您的请求。"

            # 使用 asyncio.to_thread 运行同步的 subprocess
            # 或者直接用 subprocess.run 如果你的事件循环允许
            logging.info(f"调用RAG脚本: python {rag_script_path} \"{student_name}\" \"{query}\"")
            
            # 设置PYTHONPATH确保rag_processor能找到同级模块 (如果需要)
            current_env = os.environ.copy()
            script_dir = str(Path(__file__).resolve().parent)
            if "PYTHONPATH" in current_env:
                current_env["PYTHONPATH"] = f"{script_dir}{os.pathsep}{current_env['PYTHONPATH']}"
            else:
                current_env["PYTHONPATH"] = script_dir
            current_env["PYTHONIOENCODING"] = "utf-8" # 确保子进程输出也是UTF-8
            current_env["PYTHONUTF8"] = "1"


            # 使用 asyncio.create_subprocess_exec 来更好地集成异步操作
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(rag_script_path), student_name, query,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=current_env,
                cwd=script_dir # 设置工作目录为脚本所在目录
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_output = stderr.decode('utf-8', errors='replace').strip()
                logging.error(f"RAG脚本执行失败 (返回码 {process.returncode}):\n{error_output}")
                return f"抱歉，学习规划助手在处理时遇到错误。详情: {error_output[:200]}" # 只显示部分错误

            result = stdout.decode('utf-8', errors='replace').strip()
            logging.info(f"RAG脚本成功执行，输出: {result[:200]}...")
            return result

        except FileNotFoundError:
            logging.error("RAG脚本的Python解释器未找到 (sys.executable)。")
            return "抱歉，无法启动学习规划助手。"
        except Exception as e:
            logging.error(f"调用RAG脚本时发生异常: {e}", exc_info=True)
            return f"抱歉，调用学习规划助手时发生系统错误: {str(e)}"

    async def process_query(self, query: str) -> str:
        # 检查是否为RAG查询
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
            # 调用RAG处理器
            return await self._run_rag_processor(extracted_student_name, query)
        
        # --- 原有的非RAG查询处理逻辑 ---
        messages = [{"role": "user", "content": query}]
        
        available_tools = []
        # ... (原有的工具列出逻辑，如果非RAG查询也需要工具调用的话)
        # 如果普通问答不需要工具，可以简化

        try:
            # 对于普通问答，直接调用大模型
            logging.info(f"向大模型发送普通查询: {query}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                # tools=available_tools if available_tools else None, # 如果有工具
                # tool_choice="auto" if available_tools else None,
            )
            
            # 这里简化，假设普通问答不直接调用MCP工具链，只做纯文本问答
            # 如果需要工具调用，需要加入tool_calls处理逻辑
            final_output = response.choices[0].message.content
            logging.info(f"大模型普通问答回复: {final_output[:100]}...")

            # (可选) 保存普通问答记录
            # ...

            return final_output

        except Exception as e:
            logging.error(f"处理普通查询时出错: {e}", exc_info=True)
            return f"抱歉，处理您的提问时发生错误: {str(e)}"


    async def process_research_request(self, student_name: str) -> str:
        """
        处理学生的研究文献分析请求
        """
        try:
            email = STUDENT_EMAILS.get(student_name)
            if not email:
                return f"❌ 未找到 {student_name} 的邮箱配置，请先在 STUDENT_EMAILS 中添加"

            logging.info(f"处理 {student_name} 的研究请求，邮箱: {email}")
            
            # 确保 server 会话存在
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
        # ... (原有 chat_loop 内容) ...
        print("\n🤖 科研智能体已启动！输入 'quit' 退出")
        print("1. 发送论文推荐：'给xxx发论文'")
        print(f"2. 进行学习规划（RAG）：包含学生姓名（如“小明”）和关键词（如“规划”）的提问，例如：“帮小明规划一下接下来的学习重点”")
        print(f"\n📋 当前支持的学生列表：{', '.join(STUDENT_EMAILS.keys())}")

        while True:
            try:
                query = input("\n请输入命令或问题：").strip() # 修改提示
                if query.lower() == 'quit':
                    break

                # 处理发送论文请求
                paper_request_match = re.match(r'^给(\w+)发论文$', query)
                if paper_request_match:
                    student_name = paper_request_match.group(1)
                    print(f"\n📚 正在处理 {student_name} 的论文推荐...")
                    response = await self.process_research_request(student_name)
                else:
                    # 处理其他类型的查询 (包括RAG或普通问答)
                    print(f"\n💬 正在处理您的问题: {query[:50]}...")
                    response = await self.process_query(query)

                print(f"\n🤖 AI: {response}")

            except Exception as e:
                logging.error(f"处理请求时出错: {str(e)}", exc_info=True)
                print(f"❌ 处理请求时出错: {str(e)}")


    async def plan_tool_usage(self, query: str, tools: List[dict]) -> List[dict]:
        # ... (原有 plan_tool_usage 内容，目前RAG不经过这里，普通问答也简化了) ...
        # 如果普通问答也需要工具链规划，则保留此方法并由 process_query 调用
        # 目前的简化版 process_query 不直接使用这个
        logging.warning("plan_tool_usage 当前未被 RAG 或简化版普通问答流程调用。")
        return []


    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    client = MCPClient()
    try:
        # 连接自定义服务器
        # 确保路径正确，最好使用相对于当前文件的路径或绝对路径
        server_script_path = str(Path(__file__).resolve().parent / "server.py")
        # 或者你的绝对路径: server_script_path = r"E:\实习\MCP\MCP_Demo\mcp-project\server.py"
        
        if not Path(server_script_path).exists():
            logging.error(f"主要服务脚本 server.py 未找到于: {server_script_path}")
            # return # 或者抛出异常
        else:
            await client.connect_to_server(server_script_path)
        
        # 连接 mcp.json 中定义的服务 (如果需要的话)
        # await client.connect_to_mcp_services()
        
        # 进入聊天循环
        await client.chat_loop()
    except Exception as e:
        logging.critical(f"MCPClient main loop encountered an unrecoverable error: {e}", exc_info=True)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info >= (3,8): # 兼容Windows环境下的asyncio策略
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())