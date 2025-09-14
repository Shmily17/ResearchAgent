import asyncio
import logging
import os
import json
import re
import smtplib
from datetime import datetime
from email.message import EmailMessage
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from openai import OpenAI
# import os # 已在上面导入
# import json # 已在上面导入
# import logging # 已在上面导入
from typing import List, Dict, Any, Optional, Tuple, Union
from itertools import chain
from pathlib import Path

import pyalex
from pyalex import Works, Authors, Sources, Institutions, Topics, Publishers, Funders
from pyalex.api import Work
from mcp.server.fastmcp import FastMCP, Context # Corrected import
import mcp.types as mcp_types
import chromadb
from chromadb.config import Settings # 注意：如果使用新版 chromadb>0.4.22，可能是 Settings
# from chromadb.config import Settings as ChromaSettings # 旧版 ChromaDB <0.4.0 可能用 ChromaSettings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
import sys

from pypinyin import Style, pinyin
logging.basicConfig(
    level=logging.DEBUG, # 可以调整为 INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # 确保日志输出到标准输出
    ]
)
logger = logging.getLogger(__name__)

logger.debug("DEBUG_SERVER: Script execution started.")
print("PRINT_SERVER: Script execution started (direct print).", flush=True)
# 加载环境变量
load_dotenv()
# logging.basicConfig(level=logging.INFO) # 已在上面配置
# logger = logging.getLogger(__name__) # 已在上面获取

# 初始化 MCP 服务器
mcp = FastMCP("ResearchAssistant")

# 初始化 ChromaDB
CHROMA_DB_DIR = "./chroma_db" # 这是通用的 ChromaDB 目录

openai_key = os.getenv("DASHSCOPE_API_KEY")
model_name = os.getenv("MODEL") # 将变量名改为 model_name 以避免与模块名混淆
openai_client = OpenAI(api_key=openai_key, base_url=os.getenv("BASE_URL"))


class StudentResearchManager:
    def __init__(self, student_name: str):
        self.student_name_display = student_name

        pinyin_list = pinyin(student_name, style=Style.NORMAL)
        sanitized_name_parts = [item[0] for item in pinyin_list]
        sanitized_id = "_".join(sanitized_name_parts)
        sanitized_id = re.sub(r'[^a-zA-Z0-9_]', '', sanitized_id)
        sanitized_id = re.sub(r'_+', '_', sanitized_id).strip('_')
        if not sanitized_id:
            sanitized_id = "default_student"
        self.student_id = sanitized_id
        logger.info(f"StudentResearchManager: 原始名称 '{student_name}', 清理后的ID '{self.student_id}'")

        # 确保使用正确的 Settings 类
        try:
            chroma_settings = Settings(persist_directory=CHROMA_DB_DIR, anonymized_telemetry=False)
        except TypeError: # 兼容旧版 chromadb 可能没有 anonymized_telemetry
            chroma_settings = Settings(persist_directory=CHROMA_DB_DIR)
            logger.warning("ChromaDB Settings: 'anonymized_telemetry' not available in this version. Using default.")


        global_chroma_client = chromadb.Client(chroma_settings)
        self.collection = global_chroma_client.get_or_create_collection(name=f"research_papers_{self.student_id}")
        logger.info(f"学生 '{self.student_id}' 的 ChromaDB 集合 '{self.collection.name}' 已准备就绪。")

        self.student_folder: Path = Path(f"./student_papers/{self.student_id}")
        self.student_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"学生 '{self.student_id}' 的论文文件夹路径: {self.student_folder.resolve()}")

    def get_existing_paper_ids(self) -> List[str]:
        """从该学生的ChromaDB集合中获取所有已存储文献的ID。"""
        try:
            # .get() 方法在没有指定ids时，会返回集合中所有的数据项
            # 我们只需要ID，所以 include=[] 可以减少返回的数据量
            results = self.collection.get(include=[]) 
            existing_ids = results['ids'] if results and 'ids' in results else []
            logger.info(f"为学生 {self.student_id} 从ChromaDB获取到 {len(existing_ids)} 个已存在的文献ID。")
            return existing_ids
        except Exception as e:
            logger.error(f"为学生 {self.student_id} 获取已存在文献ID时出错: {e}", exc_info=True)
            return []

    async def scan_papers(self) -> str:
        # ... (scan_papers remains the same)
        try:
            supported_extensions = {'.pdf', '.txt'}
            paper_files = []
            
            for file_path_obj in self.student_folder.rglob("*"):
                if file_path_obj.is_file() and file_path_obj.suffix.lower() in supported_extensions:
                    paper_files.append(str(file_path_obj.resolve()))
            
            return f"✅ 扫描完成，在 {self.student_name_display} 的文件夹 {self.student_folder.resolve()} 中找到 {len(paper_files)} 个文件:\n" + "\n".join(paper_files)
        except Exception as e:
            return f"❌ 扫描文件夹失败: {str(e)}"

    async def analyze_direction(self) -> str:
        """
        分析该学生文件夹中的论文，提取核心关键词。
        返回值应该是一个逗号分隔的关键词字符串。
        """
        try:
            documents = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            
            has_files_to_analyze = False
            for file_path_obj in self.student_folder.rglob("*"):
                if not file_path_obj.is_file():
                    continue
                has_files_to_analyze = True
                if file_path_obj.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(file_path_obj.resolve()))
                    documents.extend(loader.load())
                elif file_path_obj.suffix.lower() == '.txt':
                    loader = TextLoader(str(file_path_obj.resolve()))
                    documents.extend(loader.load())
            
            if not documents:
                if not has_files_to_analyze:
                    return f"ERROR: 未在 {self.student_name_display} 的文件夹 ({self.student_folder.resolve()}) 中找到任何可分析的文档（PDF或TXT）。"
                else:
                    return f"ERROR: 在 {self.student_name_display} 的文件夹 ({self.student_folder.resolve()}) 中找到了文件，但未能成功加载任何文档内容进行分析。"

            texts = text_splitter.split_documents(documents)
            combined_text = "\n".join([doc.page_content for doc in texts]) 
            if len(combined_text) > 15000: 
                combined_text = combined_text[:15000]

            system_prompt_content = (
                "你是一个专业的科研文献关键词提取助手。"
                "你的任务是从提供的文本中提取出 **唯一一个最核心、最能代表研究主题的关键词或关键短语**。" 
                "请严格遵守以下规则："
                "1. **只输出一个关键词或一个关键短语。** 不要输出列表。" 
                "2. 这个关键词/短语应该是文本中最重要、最具概括性的概念。"
                "3. 不要添加任何前缀 (如 'Keywords:', '提取的关键词是:', '以下是关键词:')。"
                "4. 不要添加任何序号、点号、解释、标题或任何其他说明性文字。"
                "5. 不要输出任何 Markdown 格式。"
                "6. 关键词/短语应尽可能精确并具有代表性，最好是名词性短语。"
                "7. 如果无法提取到明确的单个核心关键词，请返回 'N/A'。"
                "例如，如果输入文本是关于用LLM改进MedVQA，一个好的输出示例是 'Medical Visual Question Answering' 或者 'MedVQA' (选择其中一个最核心的)。"
                "另一个例子，如果文本讨论多种技术但核心是 'Deep Learning for Image Recognition'，那么就输出 'Deep Learning for Image Recognition'。"
                "请确保你的回答只有这个关键词本身，没有其他任何多余字符。"
            )
            user_prompt_content = f"请从以下研究文献内容中提取唯一一个最核心的关键词或关键短语：\n\n{combined_text}"
            
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt_content},
                    {"role": "user", "content": user_prompt_content}
                ],
                temperature=0.1 
            )
            
            single_keyword_string = response.choices[0].message.content.strip()
            
            single_keyword_string = single_keyword_string.replace('"', '').replace("'", "")
            if single_keyword_string.startswith("- "):
                single_keyword_string = single_keyword_string[2:]
            prefixes_to_remove = ["keywords:", "keyword:", "extracted keyword:", "key term:", "以下是关键词：", "关键词：", "核心关键词："] 
            for prefix in prefixes_to_remove:
                if single_keyword_string.lower().startswith(prefix):
                    single_keyword_string = single_keyword_string[len(prefix):].strip()
                    break

            if single_keyword_string.endswith('.'):
                single_keyword_string = single_keyword_string[:-1].strip()
            
            if not single_keyword_string or \
               single_keyword_string.lower() == 'n/a' or \
               ',' in single_keyword_string or \
               len(single_keyword_string.split()) > 7: 
                logger.warning(f"未能从 {self.student_name_display} 的文献中提取到明确的单一核心关键词，LLM返回: '{response.choices[0].message.content}'. 返回的清理后结果: '{single_keyword_string}'")
                return "ERROR: 未能提取到明确的单一核心研究关键词。" 

            logger.info(f"为 {self.student_name_display} 提取的单一核心关键词: {single_keyword_string}")
            return single_keyword_string

        except Exception as e:
            logger.error(f"分析研究方向（提取关键词）时出错: {e}", exc_info=True)
            return f"ERROR: 分析研究方向（提取关键词）失败: {str(e)}"

    async def update_database(self, paper_info: Dict[str, Any]) -> str:
        """更新该学生的RAG数据库"""
        try:
            paper_id_str = str(paper_info["id"])

            existing_entry = self.collection.get(ids=[paper_id_str]) # ChromaDB get returns a dict with 'ids', 'documents', etc.
            
            if existing_entry and paper_id_str in existing_entry['ids']:
                return f"ℹ️ 文献 '{paper_info['title']}' (ID: {paper_id_str}) 已存在于 {self.student_name_display} 的数据库中"
            
            embeddings_model = OllamaEmbeddings(model="bge-m3") 
            embedding = embeddings_model.embed_query(paper_info["title"] + " " + paper_info.get("abstract", ""))
            
            metadata_to_add = {
                "id": paper_id_str, 
                "title": str(paper_info.get("title", "")),
                "authors": json.dumps(paper_info.get("authors", [])), 
                "doi": str(paper_info.get("doi", "")),
                "year": str(paper_info.get("year", "")) 
            }
            metadata_to_add = {k: v for k, v in metadata_to_add.items() if v is not None and v != ""}

            self.collection.add(
                documents=[str(paper_info["title"]) + "\n" + str(paper_info.get("abstract", ""))],
                metadatas=[metadata_to_add],
                ids=[paper_id_str] 
            )
            
            return f"✅ 已将文献 '{paper_info['title']}' (ID: {paper_id_str}) 添加到 {self.student_name_display} 的检索库"
        except Exception as e:
            logger.error(f"更新检索库失败: {e}", exc_info=True)
            return f"❌ 更新检索库失败 (文献ID: {paper_info.get('id', 'N/A')}): {str(e)}"


@mcp.tool()
async def send_email_with_attachment(to: str, subject: str, body: str, attachment_full_path: Optional[str] = None) -> str:
    """
    发送带附件的邮件。如果提供了 attachment_full_path，则会尝试附加该文件。

    参数:
        to: 收件人邮箱地址
        subject: 邮件标题
        body: 邮件正文
        attachment_full_path (Optional[str]): 要附加的文件的完整绝对路径。如果为 None 或空，则不发送附件。

    返回:
        邮件发送状态说明
    """
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port_str = os.getenv("SMTP_PORT", "465")
    sender_email = os.getenv("EMAIL_USER")
    sender_pass = os.getenv("EMAIL_PASS")

    if not all([smtp_server, smtp_port_str, sender_email, sender_pass]):
        missing_configs = [name for name, val in [("SMTP_SERVER", smtp_server), ("SMTP_PORT", smtp_port_str), ("EMAIL_USER", sender_email), ("EMAIL_PASS", sender_pass)] if not val]
        error_msg = f"❌ 邮件发送失败：SMTP配置不完整，请在 .env 文件中检查以下配置项: {', '.join(missing_configs)}"
        logger.error(error_msg)
        return error_msg

    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        error_msg = f"❌ 邮件发送失败：SMTP_PORT ('{smtp_port_str}') 不是一个有效的端口号。"
        logger.error(error_msg)
        return error_msg

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to
    msg.set_content(body)

    attachment_status_message = "（无附件）"

    if attachment_full_path and os.path.exists(attachment_full_path) and os.path.isfile(attachment_full_path):
        try:
            with open(attachment_full_path, "rb") as f:
                file_data = f.read()
                file_name_for_email = os.path.basename(attachment_full_path)
                msg.add_attachment(file_data, maintype="application", subtype="octet-stream", filename=file_name_for_email)
            attachment_status_message = f"（含附件: {file_name_for_email}）"
            logger.info(f"已成功准备附件: {attachment_full_path}")
        except Exception as e:
            logger.error(f"读取或添加附件 {attachment_full_path} 失败: {str(e)}", exc_info=True)
            attachment_status_message = f"（附件 {os.path.basename(attachment_full_path)} 读取失败）"
    elif attachment_full_path:
        logger.warning(f"提供的附件路径 {attachment_full_path} 无效或不是文件。邮件将不带此附件发送。")
        attachment_status_message = f"（附件 {os.path.basename(attachment_full_path)} 未找到或无效）"
    else:
        logger.info("未提供附件路径，邮件将不含附件发送。")

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_pass)
            server.send_message(msg)
        success_msg = f"✅ 邮件已成功发送给 {to} {attachment_status_message}"
        logger.info(success_msg)
        return success_msg
    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"❌ 邮件发送失败：SMTP认证失败。请检查 EMAIL_USER 和 EMAIL_PASS (授权码) 是否正确。错误: {e}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ 邮件发送失败，发生未知错误: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

def _select_fields(item: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    required_ids = {"id", "doi"} 
    selected_fields_set = set(fields) | required_ids
    return {k: v for k, v in item.items() if k in selected_fields_set}

def _process_results(
    results: List[Dict[str, Any]],
    select_fields: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    processed = []
    for item in results:
        if not isinstance(item, dict):
             if hasattr(item, 'to_dict'): 
                 item = item.to_dict()
             else:
                 logger.warning(f"Skipping non-dictionary item: {type(item)}")
                 continue
        is_pyalex_obj = hasattr(item, 'items') and callable(item.items)
        generated_abstract = None
        if select_fields is None and is_pyalex_obj:
            try:
                generated_abstract = item["abstract"] 
            except KeyError:
                logger.warning(f"Abstract could not be generated for {item.get('id')}, index likely missing.")
            except Exception as gen_err:
                logger.error(f"Error during abstract generation for {item.get('id')}: {gen_err}")
        if is_pyalex_obj:
            dict_item = dict(item.items())
        else:
            dict_item = item 
        if generated_abstract is not None:
             dict_item["abstract"] = generated_abstract
        elif select_fields is None and "abstract" not in dict_item:
             dict_item["abstract"] = None
        if select_fields:
            processed.append(_select_fields(dict_item, select_fields))
        else:
            processed.append(dict_item)
    return processed

def _summarize_work(work_dict: Dict[str, Any], select_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    summary_data = {}
    summary_data["id"] = work_dict.get("id")
    summary_data["doi"] = work_dict.get("doi")
    summary_data["title"] = work_dict.get("title")
    summary_data["publication_year"] = work_dict.get("publication_year")
    summary_data["authors"] = [
        authorship.get("author", {}).get("display_name")
        for authorship in work_dict.get("authorships", []) 
        if authorship.get("author", {}).get("display_name")
    ][:6] 
    summary_data["cited_by_count"] = work_dict.get("cited_by_count")
    venue_source = work_dict.get("primary_location", {}).get("source", {})
    summary_data["venue"] = venue_source.get("display_name") if venue_source else None
    summary_data["oa_url"] = None
    best_oa = work_dict.get("best_oa_location")
    if best_oa and best_oa.get("pdf_url"):
        summary_data["oa_url"] = best_oa.get("pdf_url")
    elif work_dict.get("open_access", {}).get("oa_url"):
         summary_data["oa_url"] = work_dict.get("open_access", {}).get("oa_url")
    summary_data["abstract"] = work_dict.get("abstract") 
    default_summary_keys = set(summary_data.keys())
    if select_fields is None:
        keys_to_include = default_summary_keys
    else:
        keys_to_include = default_summary_keys.intersection(set(select_fields))
    final_summary = {key: summary_data[key] for key in keys_to_include if summary_data[key] is not None}
    return final_summary

# --- MCP Tools ---

@mcp.tool()
async def search_works(
    search_query: str,
    filters: Optional[Dict[str, Any]] = None,
    search_field: str = "default",
    select_fields: Optional[List[str]] = None,
    sort: Optional[Dict[str, str]] = None,
    summarize_results: bool = True, 
    per_page: int = 25,
    cursor: Optional[str] = "*",
    exclude_ids: Optional[List[str]] = None, # <<<< 新增参数
) -> Dict[str, Any]:
    """
    Searches for OpenAlex works based on keywords and filters, returning selected fields.
    Supports boolean operators in the search query. Uses cursor pagination.
    Can exclude specified IDs from the results.

    Args:
        search_query: Search term(s).
        filters: Key-value pairs for filtering.
        search_field: Field to search within.
        select_fields: List of root-level fields to return.
        sort: Field to sort by and direction.
        summarize_results: If True, returns a condensed summary.
        per_page: Number of results desired after filtering exclusions.
        cursor: Cursor for pagination.
        exclude_ids: A list of OpenAlex work IDs (short form like WXXXXXXXX or full URL)
                     to exclude from the search results.
    Returns:
        A dictionary containing the search results and pagination metadata.
    """
    logger.info(f"search_works: 正在运行，查询='{search_query}', 期望每页={per_page}, 排除ID数量={len(exclude_ids or [])}")
    try:
        query = Works()

        if search_field == "default":
            query = query.search(search_query)
        elif search_field in ["title", "abstract", "fulltext", "title_and_abstract", "display_name"]:
            search_filter_dict = {search_field: search_query}
            if search_field == "display_name":
                search_filter_dict = {"title": search_query}
            query = query.search_filter(**search_filter_dict)
        else:
            raise ValueError(f"Invalid search_field: {search_field}")

        logger.debug(f"search_works: 预过滤器找到 {query.count()} 个结果。")

        if filters:
            processed_filters = {}
            for key, value in filters.items():
                if '.' in key:
                     logger.warning(f"嵌套过滤器键 '{key}'直接传递。请确保pyalex兼容性。")
                processed_filters[key] = value
            query = query.filter(**processed_filters)

        if sort:
            query = query.sort(**sort)

        # ---- 调整获取逻辑以处理 exclude_ids ----
        # 我们需要获取足够多的文献，以便在排除后仍有 per_page 数量。
        # 这是一个启发式方法，可能需要多次分页才能满足要求。
        # 为简单起见，我们先获取一个稍大的批次。
        # 如果 exclude_ids 很多，OpenAlex 的最大 per_page (200) 可能是限制因素。
        
        internal_fetch_per_page = per_page
        if exclude_ids:
            # 请求更多文献，为排除留出余地。这里用一个简单的buffer。
            # 更稳健的做法是循环获取页面，直到收集到 per_page 个新文献。
            internal_fetch_per_page = per_page + len(exclude_ids)
            if internal_fetch_per_page > 200: # OpenAlex API 上限
                internal_fetch_per_page = 200
                logger.warning(f"search_works: 由于排除ID过多，内部请求数量调整为最大值200。")
        
        logger.info(f"search_works: 内部将从OpenAlex请求 {internal_fetch_per_page} 篇文献。")

        if not summarize_results and select_fields:
            logger.info(f"search_works: 关闭摘要，选择字段: {select_fields}")
            query = query.select(select_fields)
        
        # pyalex paginate 返回页面迭代器
        pager = query.paginate(per_page=internal_fetch_per_page, cursor=cursor, n_max=internal_fetch_per_page) 

        page_results_raw = []
        metadata = {}
        try:
            first_page = next(pager)
            page_results_raw = first_page
            if hasattr(first_page, 'meta'):
                 metadata = first_page.meta
            else:
                 if hasattr(query, '_get_meta'):
                     try:
                         raw_meta_response = await query._get_meta() # pyalex 内部细节
                         metadata = raw_meta_response.get('meta', {})
                     except Exception as meta_err:
                         logger.warning(f"search_works: 无法通过内部回退检索元数据: {meta_err}")
                 else:
                     logger.warning("search_works: 无法检索分页元数据。")
        except StopIteration:
            logger.info("search_works: 当前页面未找到结果。")
            metadata = {'count': 0, 'page': 1, 'per_page': internal_fetch_per_page, 'next_cursor': None}

        # 预处理结果（例如，转换为字典）
        processed_page_results = _process_results(page_results_raw, None) # 总是获取完整对象进行过滤和摘要

        # ---- 在此过滤 exclude_ids ----
        final_filtered_results = []
        if exclude_ids:
            normalized_exclude_ids = set()
            for ex_id in exclude_ids:
                # OpenAlex ID可能是完整URL或短ID (如 W12345)
                if ex_id.startswith("https://openalex.org/"):
                    normalized_exclude_ids.add(ex_id)
                    normalized_exclude_ids.add(ex_id.split("/")[-1])
                else:
                    normalized_exclude_ids.add(ex_id)
                    normalized_exclude_ids.add(f"https://openalex.org/{ex_id}")
            
            for item in processed_page_results:
                item_id_full = item.get("id") # 通常是完整URL
                item_id_short = item_id_full.split("/")[-1] if item_id_full and "/" in item_id_full else item_id_full
                
                if not (item_id_full in normalized_exclude_ids or \
                        (item_id_short and item_id_short in normalized_exclude_ids)):
                    final_filtered_results.append(item)
            
            logger.info(f"search_works: 从 {len(processed_page_results)} 个获取结果中排除了 "
                        f"{len(processed_page_results) - len(final_filtered_results)} 个ID。保留 "
                        f"{len(final_filtered_results)} 个。")
        else:
            final_filtered_results = processed_page_results

        # 如果过滤后的结果多于期望的 per_page，则截断
        if len(final_filtered_results) > per_page:
            final_filtered_results = final_filtered_results[:per_page]
            logger.info(f"search_works: 将过滤后的结果截断为期望的 {per_page} 个。")

        # 应用摘要（如果需要）
        if summarize_results:
            final_results_for_output = [_summarize_work(item, select_fields) for item in final_filtered_results]
        else:
            # 如果不进行摘要，但之前已通过 query.select() 选择字段，则结果已经是筛选过的
            # 如果没有 select_fields，则 final_filtered_results 包含完整对象
            final_results_for_output = final_filtered_results


        # 构造响应
        # 注意: metadata.count 仍然是 OpenAlex 对原始查询的总计数，而不是过滤后的计数
        response = {
            "results": final_results_for_output,
            "meta": {
                "count": metadata.get("count"), # OpenAlex 的原始总数
                "returned_count": len(final_results_for_output), # 本次调用实际返回的数量
                "requested_per_page": per_page, # 用户请求的每页数量
                "next_cursor": metadata.get("next_cursor"),
            },
        }
        return response

    except Exception as e:
        logger.exception(f"search_works 错误: {e}")
        return {"error": str(e), "results": [], "meta": {}}

@mcp.tool()
async def get_work_details(
    ctx: Context,
    work_id: str,
    select_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Retrieves detailed information for a specific OpenAlex work by its ID
    (OpenAlex ID URL, DOI URL, PMID URL, MAG ID). Handles abstract generation, no need to request abstract_inverted_index, simple retrieve abstract.

    Args:
        work_id: Identifier for the work (OpenAlex ID URL, DOI URL, PMID URL, MAG ID).
        select_fields: List of root-level fields to return. See OpenAlex docs for options.
                       If omitted, returns the full object.

    Returns:
        A dictionary containing the work details or an error message.
    """
    try:
        final_result = {}
        fetch_full_object = not select_fields or (select_fields and "abstract" in select_fields)

        if fetch_full_object:
            logger.info("Fetching full work object for abstract generation or full details.")
            work_data: Work = Works()[work_id]
            if not work_data: return {"error": f"Work not found: {work_id}"}

            work_dict = dict(work_data.items())
            try:
                generated_abstract = work_data["abstract"]
                work_dict["abstract"] = generated_abstract 
            except KeyError:
                 logger.warning(f"Abstract could not be generated for {work_id}, index likely missing.")
                 work_dict["abstract"] = None 
            except Exception as gen_err:
                 logger.error(f"Error during abstract generation for {work_id}: {gen_err}")
                 work_dict["abstract"] = None 
            if select_fields:
                final_result = _select_fields(work_dict, select_fields)
            else:
                final_result = work_dict 
            should_remove_index = (
                "abstract" in final_result and final_result.get("abstract") is not None and
                "abstract_inverted_index" in final_result and
                (not select_fields or "abstract_inverted_index" not in select_fields)
            )
            if should_remove_index:
                del final_result["abstract_inverted_index"]
        else:
            logger.info("Abstract not requested, using pyalex select() for efficiency.")
            query_select = list(set(select_fields) | {"id", "doi"})
            work_data: Work = Works().select(query_select)[work_id]
            if not work_data: return {"error": f"Work not found: {work_id}"}
            final_result = dict(work_data.items())
        return final_result
    except Exception as e:
        logger.exception(f"Error in get_work_details for ID {work_id}: {e}")
        return {"error": str(e)}

@mcp.tool()
async def get_batch_work_details(
    ctx: Context,
    work_ids: List[str],
    select_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Retrieves detailed information for a list of OpenAlex works by their IDs.
    Limited to a maximum of 50 IDs per request due to API limitations.

    Args:
        work_ids: A list of OpenAlex work identifiers (max 50).
        select_fields: List of root-level fields to return for each work.
                       See OpenAlex docs for options. If omitted, returns full objects.

    Returns:
        A dictionary containing a list of work details under the 'works' key,
        or an error message.
    """
    MAX_IDS = 50
    if not work_ids:
        return {"error": "work_ids list cannot be empty.", "works": []}
    if len(work_ids) > MAX_IDS:
        return {"error": f"Too many work_ids provided. Maximum is {MAX_IDS}.", "works": []}

    try:
        cleaned_ids = [
            _id.split("/")[-1] if _id.startswith("https://openalex.org/") else _id
            for _id in work_ids
        ]
        id_filter = {"ids": {"openalex": "|".join(cleaned_ids)}}
        query = Works().filter(**id_filter)
        abstract_requested = False
        api_select_fields = None
        
        if select_fields:
            abstract_requested = "abstract" in select_fields
            api_select_fields = [field for field in select_fields if field != "abstract"]
            if abstract_requested and "abstract_inverted_index" not in api_select_fields:
                api_select_fields.append("abstract_inverted_index")
            if "id" not in api_select_fields:
                api_select_fields = ["id"] + api_select_fields
            query = query.select(api_select_fields)
        
        results_list = query.get() 
        processed_results = []
        for work in results_list:
            if not isinstance(work, dict):
                if hasattr(work, 'to_dict'):
                    work_dict = work.to_dict()
                elif hasattr(work, 'items') and callable(work.items):
                    work_dict = dict(work.items())
                else:
                    logger.warning(f"Skipping non-dictionary item: {type(work)}")
                    continue
            else:
                work_dict = work
            if abstract_requested or not select_fields:
                if "abstract_inverted_index" in work_dict:
                    try:
                        if hasattr(work, "__getitem__"): # Check if it's subscriptable like a pyalex Work object
                            generated_abstract = work["abstract"] # This access triggers generation in pyalex
                        else: # Fallback for plain dicts (should not happen if pyalex returns Work objects)
                            inverted_index = work_dict.get("abstract_inverted_index")
                            if inverted_index:
                                words = []
                                for word_val, positions in inverted_index.items():
                                    for pos in positions:
                                        while len(words) <= pos: words.append("")
                                        words[pos] = word_val
                                generated_abstract = " ".join(filter(None, words)) # Filter out empty strings
                            else: generated_abstract = None
                        work_dict["abstract"] = generated_abstract
                    except KeyError:
                        logger.warning(f"Abstract could not be generated for {work_dict.get('id')}, index likely missing.")
                        work_dict["abstract"] = None
                    except Exception as gen_err:
                        logger.error(f"Error during abstract generation for {work_dict.get('id')}: {gen_err}")
                        work_dict["abstract"] = None
                else:
                    logger.warning(f"No abstract_inverted_index available for {work_dict.get('id')}")
                    work_dict["abstract"] = None

            if select_fields:
                current_select_fields = list(api_select_fields) # start with API fields
                if abstract_requested: # if abstract was requested, add it to selection for _select_fields
                    if "abstract" not in current_select_fields:
                         current_select_fields.append("abstract")
                result = _select_fields(work_dict, current_select_fields)
            else:
                result = work_dict

            inverted_index_requested = select_fields and "abstract_inverted_index" in select_fields
            if "abstract_inverted_index" in result and not inverted_index_requested:
                del result["abstract_inverted_index"]
            processed_results.append(result)
        return {"works": processed_results}
    except Exception as e:
        logger.exception(f"Error in get_batch_work_details for IDs {work_ids}: {e}")
        return {"error": str(e), "works": []}

@mcp.tool()
async def get_referenced_works(
    ctx: Context, 
    work_id: str,
) -> Dict[str, Any]:
    """
    Retrieves the list of OpenAlex IDs cited *by* a specific OpenAlex work (outgoing citations).
    Returns only the list of IDs. Use get_work_details for more info on each reference.

    Args:
        work_id: OpenAlex ID of the *citing* work (the one whose references you want).

    Returns:
        A dictionary containing a list of referenced work IDs under the
        'referenced_work_ids' key, or an error message.
    """
    try:
        if work_id.startswith("https://openalex.org/"):
            work_id = work_id.split("/")[-1]
        work_data = Works().select(["referenced_works"])[work_id]
        referenced_ids = work_data.get("referenced_works", [])
        return {"referenced_work_ids": referenced_ids}
    except Exception as e:
        logger.exception(f"Error in get_referenced_works for ID {work_id}: {e}")
        return {"error": str(e), "referenced_work_ids": []}

@mcp.tool()
async def get_citing_works(
    ctx: Context, 
    work_id: str,
    select_fields: Optional[List[str]] = None,
    summarize_results: bool = True, 
    per_page: int = 25,
    cursor: Optional[str] = "*",
) -> Dict[str, Any]:
    """
    Retrieves the list of works that *cite* a specific OpenAlex work (incoming citations).
    Uses cursor pagination.

    Args:
        work_id: OpenAlex ID of the *cited* work (the one you want citations for).
        select_fields: List of root-level fields to return.
        summarize_results: If True (default), returns a condensed summary.
        per_page: Number of results per page.
        cursor: Cursor for pagination.

    Returns:
        A dictionary containing the citing works results and pagination metadata,
        or an error message.
    """
    try:
        if work_id.startswith("https://openalex.org/"):
            work_id = work_id.split("/")[-1]
        query = Works().filter(cites=work_id)
        if not summarize_results and select_fields:
            logger.info(f"Summarization off, selecting fields for citing works: {select_fields}")
            query = query.select(select_fields)
        elif not summarize_results and not select_fields:
            default_summary_fields = ["id", "doi", "title", "publication_year", "authorships", "cited_by_count", "primary_location", "open_access", "abstract"]
            logger.info(f"Summarization off, no fields specified, selecting default summary fields: {default_summary_fields}")
            query = query.select(default_summary_fields)
        pager = query.paginate(per_page=per_page, cursor=cursor, n_max=per_page)
        page_results = []
        metadata = {}
        try:
            first_page = next(pager)
            page_results = first_page
            if hasattr(first_page, 'meta'):
                 metadata = first_page.meta
            else:
                 if hasattr(query, '_get_meta'):
                     try:
                         raw_meta_response = await query._get_meta()
                         metadata = raw_meta_response.get('meta', {})
                     except Exception as meta_err:
                         logger.warning(f"Could not retrieve metadata via internal fallback: {meta_err}")
                 else:
                     logger.warning("Could not retrieve pagination metadata.")
        except StopIteration:
            logger.info(f"No citing works found for page with cursor {cursor}.")
            metadata = {'count': 0, 'page': 1, 'per_page': per_page, 'next_cursor': None} 
        processed_page_results = _process_results(page_results, None if summarize_results else select_fields)
        if summarize_results:
            final_results = [_summarize_work(item, select_fields) for item in processed_page_results]
        else:
            final_results = processed_page_results
        response = {
            "results": final_results, 
            "meta": {
                "count": metadata.get("count"),
                "per_page": metadata.get("per_page", per_page),
                "next_cursor": metadata.get("next_cursor"),
            },
        }
        return response
    except Exception as e:
        logger.exception(f"Error in get_citing_works for ID {work_id}: {e}")
        return {"error": str(e), "results": [], "meta": {}}

@mcp.tool()
async def get_work_ngrams(
    ctx: Context, 
    work_id: str,
) -> Dict[str, Any]:
    """
    Retrieves the N-grams (word proximity information) for a specific OpenAlex work's full text.

    Args:
        work_id: OpenAlex ID of the work.

    Returns:
        A dictionary containing the N-gram data or an error message.
    """
    try:
        if work_id.startswith("https://openalex.org/"):
            work_id = work_id.split("/")[-1]
        ngrams_data = Works()[work_id].ngrams()
        return ngrams_data 
    except Exception as e:
        logger.exception(f"Error in get_work_ngrams for ID {work_id}: {e}")
        if hasattr(e, 'response') and e.response.status_code == 404:
             return {"error": f"N-grams not found for work ID {work_id}."}
        return {"error": str(e)}

@mcp.tool()
async def scan_research_papers(folder_path: str) -> str:
    """
    扫描指定文件夹中的研究论文（PDF和文本文件），并返回文件列表

    参数:
        folder_path (str): 要扫描的文件夹路径

    返回:
        str: 扫描结果说明
    """
    try:
        supported_extensions = {'.pdf', '.txt'}
        paper_files = []
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    paper_files.append(os.path.join(root, file))
        
        return f"✅ 扫描完成，找到 {len(paper_files)} 个文件:\n" + "\n".join(paper_files)
    except Exception as e:
        return f"❌ 扫描文件夹失败: {str(e)}"

@mcp.tool()
async def analyze_research_direction(folder_path: str) -> str:
    """
    分析用户文件夹中的论文，确定研究方向

    参数:
        folder_path (str): 包含研究论文的文件夹路径

    返回:
        str: 分析结果，要求只输出主要的研究方向,以Keywords:研究方向 作为输出
    """
    try:
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif file_path.lower().endswith('.txt'):
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
        
        texts = text_splitter.split_documents(documents)
        combined_text = "\n".join([doc.page_content for doc in texts[:5]])  
        
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个研究方向分析专家。请分析给定文本中反映的主要研究方向和关键词,要求只输出主要的研究方向,以Keywords:研究方向 作为输出."},
                {"role": "user", "content": f"请分析以下研究文献内容，总结出主要研究方向和关键词\n\n{combined_text}"}
            ]
        )
        
        analysis_result = response.choices[0].message.content
        return f"✅ 研究方向分析完成：\n{analysis_result}"
    except Exception as e:
        return f"❌ 分析研究方向失败: {str(e)}"

@mcp.tool()
async def update_rag_database(paper_info: Dict[str, Any]) -> str:
    """
    将新的文献信息添加到RAG检索库中 (此函数目前未使用全局 client/collection, 而是由 StudentResearchManager 处理)

    参数:
        paper_info (Dict[str, Any]): 包含文献信息的字典，需要包含id、title、abstract等字段

    返回:
        str: 更新操作结果说明
    """
    # 注意: 这个函数当前是独立的，但 StudentResearchManager.update_database 提供了实际功能
    # 如果要使其成为一个通用的工具，需要传递 collection 对象或 client/collection_name
    return "ℹ️ 此独立工具 update_rag_database 当前未被直接使用。请通过 StudentResearchManager 实例更新数据库。"


# server.py
@mcp.tool()
async def process_student_research(student_name: str, email: str) -> str:
    """
    处理特定学生的研究方向分析和文献推荐。
    会从学生文献中提取关键词，进行文献搜索（排除已存文献），并将推荐列表作为邮件附件发送。
    """
    try:
        logger.info(f"PROCESS_STUDENT_RESEARCH: 开始处理学生 '{student_name}', 邮箱 '{email}'")

        manager = StudentResearchManager(student_name)
        logger.info(f"PROCESS_STUDENT_RESEARCH: StudentResearchManager 已为 '{student_name}' 初始化。学生文件夹: {manager.student_folder.resolve()}")

        scan_result = await manager.scan_papers()
        logger.info(f"PROCESS_STUDENT_RESEARCH: '{student_name}' 的扫描结果: {scan_result}")

        keywords_or_error = await manager.analyze_direction()
        
        keywords_for_search = ""
        analysis_summary_for_email = "" 

        if keywords_or_error.startswith("ERROR:"):
            logger.error(f"PROCESS_STUDENT_RESEARCH: 提取关键词失败 for '{student_name}': {keywords_or_error}")
            if "未在" in keywords_or_error and "找到任何可分析的文档" in keywords_or_error:
                analysis_summary_for_email = (
                    "我们注意到您的个人文献库中当前没有可供分析的文档。因此，"
                    "我们暂时无法为您提取针对性的研究关键词。"
                )
                keywords_for_search = "general recent scientific publications" 
                logger.warning(f"PROCESS_STUDENT_RESEARCH: 学生 '{student_name}' 文件夹为空，使用通用关键词: '{keywords_for_search}'")
            else: 
                analysis_summary_for_email = (
                    f"在尝试分析您的研究方向（提取关键词）时遇到问题：\n{keywords_or_error}\n"
                    "因此，我们将使用通用关键词进行文献推荐。"
                )
                keywords_for_search = "general recent scientific publications" 
                logger.warning(f"PROCESS_STUDENT_RESEARCH: 学生 '{student_name}' 关键词提取出错，使用通用关键词: '{keywords_for_search}'")
        else:
            keywords_for_search = keywords_or_error 
            analysis_summary_for_email = (
                f"根据对您文献的分析，我们提取了以下核心关键词用于文献检索：\n'{keywords_for_search}'"
            )
            logger.info(f"PROCESS_STUDENT_RESEARCH: 为 '{student_name}' 成功提取的OpenAlex搜索关键词: '{keywords_for_search}'")

        # <<<< 新增：获取已存在的文献ID >>>>
        existing_paper_ids_in_db = manager.get_existing_paper_ids()
        logger.info(f"PROCESS_STUDENT_RESEARCH: 学生 '{student_name}' 的数据库中已有 {len(existing_paper_ids_in_db)} 篇文献将被排除在新的搜索之外。")

        search_results_response = None
        desired_new_papers = 5 # 我们希望最终推荐5篇新文献

        if keywords_for_search:
            logger.info(f"PROCESS_STUDENT_RESEARCH: 调用 search_works，查询='{keywords_for_search}', 期望获取 {desired_new_papers} 篇新文献，排除 {len(existing_paper_ids_in_db)} 篇已有文献。")
            search_results_response = await search_works(
                search_query=keywords_for_search,
                filters={"from_publication_date": "2022-01-01", "language": "en"}, # 近期英文文献
                per_page=desired_new_papers, # 期望得到的新文献数量
                summarize_results=True,
                sort={"relevance_score": "desc"}, # 按相关性排序
                exclude_ids=existing_paper_ids_in_db # <<<< 传递要排除的ID
            )
        else: 
            logger.error(f"PROCESS_STUDENT_RESEARCH: 关键词为空，无法为 '{student_name}' 进行OpenAlex搜索。这是一个意外情况。")
            search_results_response = {"results": [], "meta": {}, "error": "内部错误：关键词缺失未能执行搜索。"}

        recommended_papers_txt_content = f"为 {student_name} ({datetime.now().strftime('%Y-%m-%d %H:%M')}) 推荐的文献：\n"
        if keywords_for_search != "general recent scientific publications":
             recommended_papers_txt_content += f"基于关键词: {keywords_for_search}\n\n"
        else:
            recommended_papers_txt_content += f"基于通用科研领域推荐\n\n"

        attachment_file_path = None

        # search_works 返回的 "results" 已经是过滤和数量调整后的
        if search_results_response and "results" in search_results_response and search_results_response["results"]:
            actual_new_papers_found = len(search_results_response['results'])
            logger.info(f"PROCESS_STUDENT_RESEARCH: 从搜索中为 '{student_name}' 找到 {actual_new_papers_found} 篇 *新* 文献。")
            
            for i, paper in enumerate(search_results_response["results"], 1):
                recommended_papers_txt_content += f"--- 文献 {i} ---\n"
                recommended_papers_txt_content += f"标题: {paper.get('title', 'N/A')}\n"
                recommended_papers_txt_content += f"年份: {paper.get('publication_year', 'N/A')}\n"
                authors_list = paper.get('authors', [])
                if isinstance(authors_list, list):
                    authors_str = ", ".join(filter(None, authors_list))
                else: 
                    authors_str = str(authors_list)
                recommended_papers_txt_content += f"作者: {authors_str if authors_str else 'N/A'}\n"
                recommended_papers_txt_content += f"DOI: {paper.get('doi', 'N/A')}\n"
                recommended_papers_txt_content += f"摘要: {paper.get('abstract', 'N/A')}\n"
                openalex_id = paper.get('id', '')
                if openalex_id and not openalex_id.startswith("https://openalex.org/"):
                    recommended_papers_txt_content += f"OpenAlex链接: https://openalex.org/{openalex_id}\n"
                elif openalex_id:
                     recommended_papers_txt_content += f"OpenAlex链接: {openalex_id}\n"

                if paper.get('oa_url'):
                     recommended_papers_txt_content += f"开放获取链接: {paper.get('oa_url')}\n"
                recommended_papers_txt_content += "\n"

                paper_data_for_db = { 
                    "id": paper.get("id"),
                    "title": paper.get("title"),
                    "abstract": paper.get("abstract"), 
                    "authors": paper.get("authors"), 
                    "doi": paper.get("doi"),
                    "year": paper.get("publication_year")
                }
                if paper_data_for_db.get("id") and paper_data_for_db.get("title"): 
                    # update_database 内部会检查是否已存在，但由于我们这里是新文献，理论上不会重复
                    update_status = await manager.update_database(paper_data_for_db)
                    logger.info(f"数据库更新状态 ({paper.get('id')}): {update_status}")
                else:
                    logger.warning(f"PROCESS_STUDENT_RESEARCH: 因缺少ID或标题，跳过更新数据库的文献: {paper.get('id')}")
            
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            txt_filename = f"recommended_papers_{manager.student_id}_{timestamp_str}.txt"
            attachment_file_path_obj = manager.student_folder.joinpath(txt_filename)
            
            try:
                with open(attachment_file_path_obj, "w", encoding="utf-8") as f:
                    f.write(recommended_papers_txt_content)
                attachment_file_path = str(attachment_file_path_obj.resolve())
                logger.info(f"PROCESS_STUDENT_RESEARCH: 推荐文献已保存到 TXT 文件: {attachment_file_path}")
            except Exception as e:
                logger.error(f"PROCESS_STUDENT_RESEARCH: 保存推荐文献 TXT 文件失败: {e}", exc_info=True)
                recommended_papers_txt_content += "\n\n错误：未能将此列表保存到文件。\n"
                attachment_file_path = None
        else: 
            message = f"PROCESS_STUDENT_RESEARCH: 未能从OpenAlex为 '{student_name}' 搜索到 *新* 文献。"
            if search_results_response and "meta" in search_results_response and search_results_response["meta"].get("returned_count", 0) == 0 :
                 message += " (可能是所有找到的文献都已存在于您的库中，或没有符合条件的文献)。"
            if search_results_response and "info" in search_results_response: # 假设 search_works 可能返回 info
                message += f" ({search_results_response['info']})"
            if search_results_response and "error" in search_results_response:
                message += f" 错误: {search_results_response['error']}"
            logger.warning(message)
            recommended_papers_txt_content += "本次未能检索到相关的最新研究文献（可能均已存在于您的库中或无新文献）。\n"

        email_body = f"亲爱的 {student_name}：\n\n"
        email_body += f"{analysis_summary_for_email}\n\n" 

        if keywords_for_search != "general recent scientific publications":
            email_body += f"基于这些关键词，我们为您找到了一些相关的最新研究文献（已排除您库中已有的文献）。\n"
        else:
             email_body += f"因此，我们为您推荐了一些近期通用的科研动态（已排除您库中已有的文献）。\n"
        
        if attachment_file_path:
            email_body += f"详细的推荐文献列表已保存在附件 TXT 文件中 ({os.path.basename(attachment_file_path)})。\n"
        elif search_results_response and "results" in search_results_response and search_results_response["results"]:
             email_body += "推荐的文献列表未能成功生成附件，请查看日志。\n"
        else:
            email_body += "本次未能检索到需要作为附件发送的推荐文献。\n"

        email_body += "\n祝研究顺利！\n"
        
        logger.info(f"PROCESS_STUDENT_RESEARCH: 准备为 '{student_name}' 发送邮件至 '{email}'。附件路径: {attachment_file_path}")
        email_result = await send_email_with_attachment(
            to=email,
            subject=f"研究文献推荐 - {student_name} - {datetime.now().strftime('%Y-%m-%d')}",
            body=email_body,
            attachment_full_path=attachment_file_path
        )
        logger.info(f"PROCESS_STUDENT_RESEARCH: '{student_name}' 的邮件发送结果: {email_result}")
        
        final_message_parts = ["✅ 处理完成！"]
        if keywords_or_error.startswith("ERROR:") and "未在" in keywords_or_error:
             final_message_parts.append("提示：您的文献库为空，本次推荐基于通用科研领域。")
        elif keywords_or_error.startswith("ERROR:"):
             final_message_parts.append("提示：提取您的研究关键词时遇到问题，本次推荐基于通用科研领域。")

        if attachment_file_path:
            final_message_parts.append(f"推荐文献已准备好并尝试作为邮件附件发送。")
        else:
            final_message_parts.append("未生成或未能成功保存推荐文献TXT文件作为附件。")
        
        final_message_parts.append(f"邮件发送状态：{email_result}")

        final_message = "\n".join(final_message_parts)
        logger.info(f"PROCESS_STUDENT_RESEARCH: 给 '{student_name}' 的最终响应: {final_message}")
        return final_message

    except Exception as e:
        logger.error(f"PROCESS_STUDENT_RESEARCH: 为学生 '{student_name}' 处理时发生未捕获的错误: {str(e)}", exc_info=True)
        return f"❌ 处理失败 (PSR 主流程错误): {str(e)}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # 或者 DEBUG
    logger = logging.getLogger(__name__) # 确保 logger 在 __main__ 中也被正确获取
    logger.info("Starting MCP server via __main__ block...")
    mcp.run(transport='stdio')
    logger.info("MCP server run finished (should only see this if it exits cleanly or crashes before blocking).")