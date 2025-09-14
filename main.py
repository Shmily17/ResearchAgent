import asyncio
import logging
import os
import json
import sys
from typing import Optional, List, Dict
from contextlib import AsyncExitStack
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - CLIENT - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

STUDENT_EMAILS = {
    "小明": "xiaoming@example.com",
    "小红": "xiaohong@example.com",
    "小华": "xiaohua@example.com",
}

class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.openai_api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)
        else:
            logger.warning("OpenAI API Key (DASHSCOPE_API_KEY) not found. LLM-dependent features will not work.")
            self.openai_client = None
        self.sessions: Dict[str, ClientSession] = {}
        self.mcp_config = self._load_mcp_config()

    def _load_mcp_config(self):
        config_path = PROJECT_ROOT / "mcp.json"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"⚠️ mcp.json not found at {config_path}. MCP server auto-discovery from config will not work.")
            return {"mcpServers": {}}
        except Exception as e:
            logger.error(f"⚠️ Error loading mcp.json: {e}")
            return {"mcpServers": {}}

    async def connect_to_server(self, server_script_path: str, server_name: str = "server"):
        try:
            logger.info(f"Connecting to server script: {server_script_path} under name '{server_name}'")
            is_python = server_script_path.endswith('.py')
            if not is_python:
                raise ValueError("Server script must be a .py file for this client.")

            command_executable = sys.executable
            logger.info(f"Using command executable: {command_executable}")

            effective_env = os.environ.copy()
            effective_env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + effective_env.get("PYTHONPATH", "")
            effective_env["PYTHONIOENCODING"] = "utf-8"
            effective_env["PYTHONUTF8"] = "1"

            server_params = StdioServerParameters(
                command=command_executable,
                args=[server_script_path],
                env=effective_env,
                cwd=str(PROJECT_ROOT)
            )
            logger.info(f"Starting server process with params: command='{server_params.command}', args={server_params.args}")
            
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            logger.info("Server process started.")

            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            logger.info(f"Session with '{server_name}' initialized.")

            self.sessions[server_name] = session
            response = await session.list_tools()
            tools = response.tools
            logger.info(f"Connected to server '{server_name}', available tools: {[tool.name for tool in tools]}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to server '{server_name}' at {server_script_path}: {str(e)}", exc_info=True)
            return False

    async def call_server_tool(self, server_name: str, tool_name: str, tool_args: Dict) -> Optional[Dict]:
        if server_name not in self.sessions:
            logger.error(f"Not connected to server '{server_name}'. Cannot call tool '{tool_name}'.")
            return None
        
        session = self.sessions[server_name]
        try:
            logger.debug(f"Calling tool '{server_name}.{tool_name}' with args: {json.dumps(tool_args, indent=2, default=str)}")
            response = await session.call_tool(tool_name, tool_args)
            if response and response.content and isinstance(response.content, list) and len(response.content) > 0:
                content_block = response.content[0]
                if hasattr(content_block, 'json_content') and content_block.json_content is not None:
                    logger.debug(f"Tool '{tool_name}' returned structured JSON content.")
                    return content_block.json_content
                elif hasattr(content_block, 'text'):
                    logger.debug(f"Tool '{tool_name}' returned text content. Attempting JSON parse.")
                    return json.loads(content_block.text)
                else:
                    logger.warning(f"Tool '{tool_name}' returned unknown content block type.")
                    return {"raw_output": str(content_block)}
            else:
                logger.warning(f"Tool '{tool_name}' did not return any content or invalid content structure.")
                return None
        except json.JSONDecodeError as je:
            raw_text_output = response.content[0].text if response and response.content else "N/A"
            logger.error(f"Failed to parse JSON response from tool '{tool_name}': {je}. Raw text: {raw_text_output}")
            return {"error": "JSONDecodeError", "raw_output": raw_text_output}
        except Exception as e:
            logger.error(f"Error calling tool '{server_name}.{tool_name}': {str(e)}", exc_info=True)
            return {"error": str(e)}

    async def cleanup(self):
        logger.info("Cleaning up MCPClient resources...")
        await self.exit_stack.aclose()
        logger.info("MCPClient cleanup complete.")


async def test_search_works_exclusion_from_client(client: MCPClient, server_instance_name: str):
    logger.info(f"--- Starting Client-Side Test for '{server_instance_name}.search_works' Exclusion ---")
    if server_instance_name not in client.sessions:
        logger.error(f"Server '{server_instance_name}' not connected. Test cannot proceed.")
        return

    test_query = "artificial intelligence ethics"
    ids_to_exclude = [
        "https://openalex.org/W4210185466",
        "https://openalex.org/W2755740951",
        "https://openalex.org/W4307940037"
    ]
    desired_new_papers = 3

    logger.info(f"\n[CLIENT_TEST] Phase 1: Searching for '{test_query}' WITHOUT exclusion")
    args_for_baseline = {
        "search_query": test_query,
        "filters": {"publication_year": ">2022"},
        "per_page": 5,
        "summarize_results": True,
        "sort": {"relevance_score": "desc"},
        "exclude_ids": None
    }
    baseline_response = await client.call_server_tool(server_instance_name, "search_works", args_for_baseline)

    actual_ids_to_exclude_for_test = []
    if baseline_response and not baseline_response.get("error") and baseline_response.get("results"):
        baseline_works = baseline_response["results"]
        logger.info(f"[CLIENT_TEST] Phase 1: Got {len(baseline_works)} works. IDs: {[w.get('id') for w in baseline_works]}")
        predefined_ids_found = [w.get('id') for w in baseline_works if w.get('id') in ids_to_exclude]
        if predefined_ids_found:
            actual_ids_to_exclude_for_test.extend(predefined_ids_found)
        else:
            actual_ids_to_exclude_for_test.extend([w.get('id') for w in baseline_works[:len(ids_to_exclude)] if w.get('id')])
    else:
        logger.error(f"[CLIENT_TEST] Phase 1 search failed or returned no results: {baseline_response}")
        actual_ids_to_exclude_for_test = ids_to_exclude

    if not actual_ids_to_exclude_for_test:
        logger.error("[CLIENT_TEST] No IDs available to exclude. Test cannot effectively check exclusion. Aborting.")
        return

    logger.info(f"[CLIENT_TEST] IDs that WILL BE EXCLUDED in Phase 2: {actual_ids_to_exclude_for_test}")

    logger.info(f"\n[CLIENT_TEST] Phase 2: Searching for '{test_query}' WITH exclusion")
    args_with_exclusion = {
        "search_query": test_query,
        "filters": {"publication_year": ">2022"},
        "per_page": desired_new_papers,
        "summarize_results": True,
        "sort": {"relevance_score": "desc"},
        "exclude_ids": actual_ids_to_exclude_for_test
    }
    excluded_response = await client.call_server_tool(server_instance_name, "search_works", args_with_exclusion)

    if excluded_response and not excluded_response.get("error") and "results" in excluded_response:
        excluded_works = excluded_response["results"]
        logger.info(f"[CLIENT_TEST] Phase 2: Got {len(excluded_works)} NEW works. IDs: {[w.get('id') for w in excluded_works]}")
        
        found_excluded_id_in_phase2 = any(work.get("id") in actual_ids_to_exclude_for_test for work in excluded_works)
        
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
    server_script_path = str(PROJECT_ROOT / "src" / "server.py")
    server_instance_name = "myResearchServer"

    try:
        connected = await client.connect_to_server(server_script_path, server_name=server_instance_name)
        if connected:
            await test_search_works_exclusion_from_client(client, server_instance_name)
        else:
            logger.error(f"Could not connect to server at {server_script_path}. Aborting test.")
    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
