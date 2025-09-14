import asyncio
import logging
import os
import json
from typing import Optional, List, Dict
from contextlib import AsyncExitStack
from datetime import datetime
import re
from openai import OpenAI # OpenAI 客户端在这里可能不是直接测试 search_works 所必需的，但保留以备其他功能
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters # StdioServerParameters 是需要的
from mcp.client.stdio import stdio_client

# 配置日志
logging.basicConfig(
    level=logging.DEBUG, # 设置为 DEBUG 以查看详细的 MCP 交互和测试日志
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - CLIENT - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# 学生邮箱配置 (这个测试中可能用不到，但保留)
STUDENT_EMAILS = {
    "小明": "1453301028@qq.com",
    "小红": "1453301028@qq.com",
    "小华": "xiaohua@example.com",
    "罗伟源": "1054633506@qq.com"
}

class MCPClient:

    def __init__(self):
        self.exit_stack = AsyncExitStack()
        # OpenAI 相关配置，对于直接测试 MCP 工具可能不是必需的，但保留
        self.openai_api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")
        if self.openai_api_key: # 只在 API key 存在时初始化
            self.openai_client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)
        else:
            logger.warning("OpenAI API Key (DASHSCOPE_API_KEY) not found. LLM-dependent features will not work.")
            self.openai_client = None
            
        self.sessions: Dict[str, ClientSession] = {}  # 存储多个服务会话
        self.mcp_config = self._load_mcp_config()

    def _load_mcp_config(self):
        """加载 mcp.json 配置文件"""
        try:
            # 确保路径正确，如果 client.py 和 mcp.json 在同一目录，或者调整路径
            # 使用相对路径可能更好，或者通过环境变量配置
            config_path = os.path.join(os.path.dirname(__file__), "..", "mcp.json") # 假设 mcp.json 在项目根目录
            # 如果 mcp.json 和 client.py 在同一个 mcp-project 目录，则用下面这个
            # config_path = os.path.join(os.path.dirname(__file__), "mcp.json") 
            # 或者使用绝对路径
            # config_path = r"E:\实习\MCP\MCP_Demo\mcp-project\mcp.json" # 你之前的路径
            
            # 尝试更通用的相对路径，假设 mcp.json 在 client.py 的上一层目录
            # 如: mcp-project/client.py 和 mcp-project/../mcp.json (即 mcp-project 的父目录下的 mcp.json)
            # 如果 mcp.json 和 client.py 在同一目录，则是:
            # config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp.json")
            # 根据你的项目结构调整
            # 为了演示，我假设 mcp.json 在 client.py 的上一级目录中的 mcp-project 文件夹下
            # 即 client.py 在 mcp-project/client/client.py, mcp.json 在 mcp-project/mcp.json
            # current_script_path = os.path.dirname(os.path.abspath(__file__))
            # project_root = os.path.dirname(current_script_path) # client 所在的目录
            # config_path = os.path.join(project_root, "mcp.json")

            # 使用你提供的原始路径，如果它确实是固定的
            config_path = r"E:\实习\MCP\MCP_Demo\mcp-project\mcp.json"
            logger.info(f"Attempting to load mcp.json from: {config_path}")

            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"⚠️ mcp.json not found at {config_path}. MCP server auto-discovery from config will not work.")
            return {"mcpServers": {}}
        except Exception as e:
            logger.error(f"⚠️ Error loading mcp.json: {e}")
            return {"mcpServers": {}}


    async def connect_to_server(self, server_script_path: str, server_name: str = "server"):
        """连接到自定义 Python/JS 服务器"""
        try:
            logger.info(f"Connecting to server script: {server_script_path} under name '{server_name}'")
            
            is_python = server_script_path.endswith('.py')
            is_js = server_script_path.endswith('.js')
            if not (is_python or is_js):
                raise ValueError("Server script must be a .py or .js file")

            command_executable = "python" if is_python else "node" # The executable itself
            logger.info(f"Using command executable: {command_executable}")

            effective_env = os.environ.copy()
            if is_python:
                effective_env["PYTHONIOENCODING"] = "utf-8"
                effective_env["PYTHONUTF8"] = "1"
            
            # CORRECTED StdioServerParameters:
            server_params = StdioServerParameters(
                command=command_executable,       # command is a string (the executable)
                args=[server_script_path],        # args is a list of strings (script path is the first arg)
                env=effective_env,
                working_directory=os.path.dirname(server_script_path) # Set working directory
            )
            # ---------------------------------

            logger.info(f"Starting server process with params: command='{server_params.command}', args={server_params.args}")
            
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            logger.info("Server process started.")

            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize() # Initialize session, handshake etc.
            logger.info(f"Session with '{server_name}' initialized.")

            self.sessions[server_name] = session

            response = await session.list_tools()
            tools = response.tools
            logger.info(f"Connected to server '{server_name}', available tools: {[tool.name for tool in tools]}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to server '{server_name}' at {server_script_path}: {str(e)}", exc_info=True)
            return False # Indicate connection failure
    async def connect_to_mcp_services(self):
        """连接到 mcp.json 中定义的服务 (如果需要)"""
        # ... (此函数可以保持不变，如果你的测试不依赖 mcp.json 定义的服务)
        logger.info("Attempting to connect to MCP services defined in mcp.json...")
        if not self.mcp_config or not self.mcp_config.get("mcpServers"):
            logger.info("No mcpServers defined in mcp.json or config not loaded. Skipping.")
            return

        for name, config in self.mcp_config["mcpServers"].items():
            if not config.get("isActive", False):
                logger.info(f"Skipping inactive service from mcp.json: {name}")
                continue
            try:
                logger.info(f"Connecting to service from mcp.json: {name}")
                effective_env = os.environ.copy()
                cmd_lower = ""
                command_config = config.get("command") #可以是字符串或列表
                if isinstance(command_config, str):
                    cmd_lower = command_config.lower()
                elif isinstance(command_config, list) and command_config:
                    cmd_lower = command_config[0].lower()
                
                if "python" in cmd_lower:
                    effective_env["PYTHONIOENCODING"] = "utf-8"
                    effective_env["PYTHONUTF8"] = "1"
                
                if config.get("env"):
                    effective_env.update(config["env"])

                # StdioServerParameters expects 'command' as list [executable, arg1, ...]
                if isinstance(command_config, str):
                    cmd_list = [command_config] + config.get("args", [])
                elif isinstance(command_config, list):
                    cmd_list = command_config + config.get("args", [])
                else:
                    logger.error(f"Invalid command configuration for service {name}")
                    continue
                
                server_params = StdioServerParameters(
                    command=cmd_list,
                    # args=[], # args are now part of command list
                    env=effective_env,
                    working_directory=config.get("workingDirectory") # 使用 mcp.json 中的 workingDirectory
                )
                logger.info(f"Starting server process for '{name}' with params: {server_params}")

                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                stdio, write = stdio_transport
                logger.info(f"Service process '{name}' started.")

                session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
                await session.initialize()
                logger.info(f"Session with service '{name}' initialized.")
                self.sessions[name] = session
                response = await session.list_tools()
                mcp_tools = response.tools
                logger.info(f"Connected to service '{name}', available tools: {[tool.name for tool in mcp_tools]}")
            except Exception as e:
                logger.error(f"Failed to connect to service '{name}' from mcp.json: {str(e)}", exc_info=True)


    async def call_server_tool(self, server_name: str, tool_name: str, tool_args: Dict) -> Optional[Dict]:
        """调用指定服务器上的指定工具"""
        if server_name not in self.sessions:
            logger.error(f"Not connected to server '{server_name}'. Cannot call tool '{tool_name}'.")
            return None
        
        session = self.sessions[server_name]
        try:
            logger.debug(f"Calling tool '{server_name}.{tool_name}' with args: {json.dumps(tool_args, indent=2, default=str)}")
            response = await session.call_tool(tool_name, tool_args)
            # MCP call_tool returns a ToolCallResult. Content is a list of ContentBlock.
            # We assume the tool returns a single JSON-compatible text block.
            if response and response.content and isinstance(response.content, list) and len(response.content) > 0:
                # Assuming the first content block is the primary output and is text
                # And that text is JSON decodable if it's a complex type.
                # For search_works, it should return a dict which gets JSON serialized.
                raw_text_output = response.content[0].text
                logger.debug(f"Raw text output from tool '{tool_name}': {raw_text_output[:500]}...") # Log snippet
                try:
                    # The server tool should return a JSON string if it's a complex object like a dict
                    # FastMCP typically handles this serialization for dicts/lists.
                    # If the server tool returns a string, it will be raw_text_output.
                    # If it returns a dict/list, FastMCP serializes it to JSON string,
                    # then ClientSession deserializes it. So response.content[0].text might already be a dict.
                    
                    # Let's check type of response.content[0]
                    # If it's mcp.types.ToolCallResultContentText, .text is string.
                    # If it's mcp.types.ToolCallResultContentJson, .json_content is dict/list.
                    content_block = response.content[0]
                    if hasattr(content_block, 'json_content') and content_block.json_content is not None:
                        logger.debug(f"Tool '{tool_name}' returned structured JSON content.")
                        return content_block.json_content # This is already a dict/list
                    elif hasattr(content_block, 'text'):
                        logger.debug(f"Tool '{tool_name}' returned text content. Attempting JSON parse.")
                        # If the server tool explicitly returned a JSON string:
                        return json.loads(content_block.text)
                    else:
                        logger.warning(f"Tool '{tool_name}' returned unknown content block type.")
                        return {"raw_output": str(content_block)} # Fallback

                except json.JSONDecodeError as je:
                    logger.error(f"Failed to parse JSON response from tool '{tool_name}': {je}. Raw text: {raw_text_output}")
                    return {"error": "JSONDecodeError", "raw_output": raw_text_output}
                except Exception as e:
                    logger.error(f"Error processing tool response for '{tool_name}': {e}", exc_info=True)
                    return {"error": str(e), "raw_output": raw_text_output if 'raw_text_output' in locals() else "Unknown"}

            else:
                logger.warning(f"Tool '{tool_name}' did not return any content or invalid content structure.")
                return None
        except Exception as e:
            logger.error(f"Error calling tool '{server_name}.{tool_name}': {str(e)}", exc_info=True)
            return {"error": str(e)}


    # process_query, chat_loop, plan_tool_usage etc. are not strictly needed for this specific test,
    # but can be kept if you use them for other purposes.

    async def cleanup(self):
        logger.info("Cleaning up MCPClient resources...")
        await self.exit_stack.aclose()
        logger.info("MCPClient cleanup complete.")


async def test_search_works_exclusion_from_client(client: MCPClient, server_instance_name: str):
    """
    在客户端测试 server.py 上的 search_works 工具的 ID 排除功能。
    """
    logger.info(f"--- Starting Client-Side Test for '{server_instance_name}.search_works' Exclusion ---")

    if server_instance_name not in client.sessions:
        logger.error(f"Server '{server_instance_name}' not connected. Test cannot proceed.")
        return

    test_query = "artificial intelligence ethics"
    # 这些ID应该是完整的OpenAlex URL格式
    # 你可以先运行一次不带exclude_ids的搜索，从结果中挑选几个ID用于此列表
    # 例如:
    # initial_search_args = {
    #     "search_query": test_query,
    #     "filters": {"publication_year": ">2022"},
    #     "per_page": 5,
    #     "summarize_results": True
    # }
    # initial_response = await client.call_server_tool(server_instance_name, "search_works", initial_search_args)
    # if initial_response and initial_response.get("results"):
    #     ids_to_exclude_from_initial = [w['id'] for w in initial_response["results"][:2]] # Exclude first 2
    # else:
    #     ids_to_exclude_from_initial = ["https://openalex.org/Wxxxxxxxxxx"] # Fallback, replace
    
    # 为了简单和可复现，我们硬编码一些ID。请用实际ID替换。
    # 确保这些ID确实可能出现在对 "artificial intelligence ethics" 的搜索结果中。
    ids_to_exclude = [
        "https://openalex.org/W4210185466", # Example: "Ethics of Artificial Intelligence" by Bostrom
        "https://openalex.org/W2755740951", # Example: "The ethics of artificial intelligence" by Floridi
        "https://openalex.org/W4307940037"  # Example: A recent paper on AI ethics
    ] 
    # 重要的: 确保这些ID和你在server.py的独立测试中使用的ID类型一致，
    # 并且和server.py中search_works工具期望的ID格式一致 (即完整的OpenAlex URL)。

    desired_new_papers = 3

    logger.info(f"\n[CLIENT_TEST] Phase 1: Searching for '{test_query}' WITHOUT exclusion (requesting {desired_new_papers + len(ids_to_exclude)} items initially to get IDs for exclusion)")
    
    # 为了获取用于排除的ID，我们先做一次搜索
    # 或者，如果你有其他方法填充 ids_to_exclude (例如从数据库读取)，可以跳过这一步
    args_for_baseline = {
        "search_query": test_query,
        "filters": {"publication_year": ">2022"}, # 确保能搜到东西
        "per_page": 5, # 获取比ids_to_exclude数量多一点，以确保能找到它们
        "summarize_results": True, # 摘要结果通常包含ID
        "sort": {"relevance_score": "desc"},
        "exclude_ids": None # 显式不排除
    }
    baseline_response = await client.call_server_tool(server_instance_name, "search_works", args_for_baseline)

    actual_ids_to_exclude_for_test = []
    if baseline_response and not baseline_response.get("error") and baseline_response.get("results"):
        baseline_works = baseline_response["results"]
        logger.info(f"[CLIENT_TEST] Phase 1: Got {len(baseline_works)} works. IDs: {[w.get('id') for w in baseline_works]}")
        # 从基线中挑选ID用于排除测试，或者合并预定义的ids_to_exclude
        predefined_ids_found_in_baseline = [w.get('id') for w in baseline_works if w.get('id') in ids_to_exclude]
        if predefined_ids_found_in_baseline:
            actual_ids_to_exclude_for_test.extend(predefined_ids_found_in_baseline)
            logger.info(f"[CLIENT_TEST] Predefined IDs found in baseline: {predefined_ids_found_in_baseline}")
        else: # 如果预定义的ID没在基线里，就从基线里随便取几个
            actual_ids_to_exclude_for_test.extend([w.get('id') for w in baseline_works[:len(ids_to_exclude)] if w.get('id')])
            logger.info(f"[CLIENT_TEST] Using IDs from baseline for exclusion: {actual_ids_to_exclude_for_test}")

        if not actual_ids_to_exclude_for_test: # 如果还是没有，就用硬编码的
             actual_ids_to_exclude_for_test = ids_to_exclude
             logger.info(f"[CLIENT_TEST] Falling back to hardcoded IDs for exclusion as none found/derived from baseline: {actual_ids_to_exclude_for_test}")
    else:
        logger.error(f"[CLIENT_TEST] Phase 1 search failed or returned no results: {baseline_response}")
        actual_ids_to_exclude_for_test = ids_to_exclude # 使用硬编码的ID进行后续测试
        logger.info(f"[CLIENT_TEST] Using hardcoded IDs for exclusion due to Phase 1 failure: {actual_ids_to_exclude_for_test}")

    if not actual_ids_to_exclude_for_test:
        logger.error("[CLIENT_TEST] No IDs available to exclude. Test cannot effectively check exclusion. Aborting.")
        return

    logger.info(f"[CLIENT_TEST] IDs that WILL BE EXCLUDED in Phase 2: {actual_ids_to_exclude_for_test}")

    logger.info(f"\n[CLIENT_TEST] Phase 2: Searching for '{test_query}' WITH exclusion (requesting {desired_new_papers} new items)")
    args_with_exclusion = {
        "search_query": test_query,
        "filters": {"publication_year": ">2022"},
        "per_page": desired_new_papers,
        "summarize_results": True,
        "sort": {"relevance_score": "desc"},
        "exclude_ids": actual_ids_to_exclude_for_test # <<<< 传递要排除的ID
    }
    excluded_response = await client.call_server_tool(server_instance_name, "search_works", args_with_exclusion)

    if excluded_response and not excluded_response.get("error") and "results" in excluded_response:
        excluded_works = excluded_response["results"]
        logger.info(f"[CLIENT_TEST] Phase 2: Got {len(excluded_works)} NEW works. IDs: {[w.get('id') for w in excluded_works]}")
        
        # 验证结果
        found_excluded_id_in_phase2 = False
        for work in excluded_works:
            work_id = work.get("id")
            if work_id in actual_ids_to_exclude_for_test:
                logger.error(f"[CLIENT_TEST] VALIDATION FAILED: ID '{work_id}' was supposed to be excluded but was found in Phase 2 results!")
                found_excluded_id_in_phase2 = True
        
        if not found_excluded_id_in_phase2:
            logger.info("[CLIENT_TEST] VALIDATION PASSED: No excluded IDs were found in the Phase 2 results.")
        else:
            logger.error("[CLIENT_TEST] VALIDATION FAILED: One or more excluded IDs were present in the Phase 2 results.")
        
        meta = excluded_response.get("meta", {})
        logger.info(f"[CLIENT_TEST] Phase 2 Meta: returned_new_count={meta.get('returned_new_count')}, requested_per_page={meta.get('requested_per_page')}")

    elif excluded_response and excluded_response.get("error"):
        logger.error(f"[CLIENT_TEST] Phase 2 FAILED with error from server: {excluded_response['error']}")
    else:
        logger.error(f"[CLIENT_TEST] Phase 2 FAILED: No results or unexpected response structure: {excluded_response}")

    logger.info(f"--- Client-Side Test for '{server_instance_name}.search_works' Exclusion Finished ---")


async def main():
    client = MCPClient()
    server_script_path = r"E:\实习\MCP\MCP_Demo\mcp-project\server.py" # 你的 server.py 路径
    server_instance_name = "myResearchServer" # 给这个服务器连接一个名字

    try:
        # 连接到你的 Python MCP 服务器
        connected = await client.connect_to_server(server_script_path, server_name=server_instance_name)
        
        if connected:
            # 如果连接成功，运行测试
            await test_search_works_exclusion_from_client(client, server_instance_name)
        else:
            logger.error(f"Could not connect to server at {server_script_path}. Aborting test.")
            
        # 你也可以在这里连接 mcp.json 中定义的服务（如果需要）
        # await client.connect_to_mcp_services()
        
        # 如果你想保留聊天循环或其他功能，可以在测试后取消注释
        # await client.chat_loop()

    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    # 确保在运行此客户端之前，server.py 没有在独立运行（例如，通过 `python server.py test_exclusion`）
    # server.py 应该由这个 client.py 通过 StdioServerParameters 启动。
    # 确保 .env 文件配置正确，尤其是 OPENALEX_EMAIL for server.py
    asyncio.run(main())