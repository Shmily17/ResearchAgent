import argparse
import asyncio
import logging
import os
import re
import sys
from pathlib import Path

from pypinyin import Style, pinyin

# langchain imports
from langchain_community.embeddings import OllamaEmbeddings # 嵌入模型仍然可以使用 Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama # 用于直接调用 Ollama LLM
from langchain.prompts import PromptTemplate

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # 输出到 stderr，方便 subprocess 捕获 stdout
)
logger = logging.getLogger(__name__)

# --- 配置区域 ---

# Ollama 嵌入模型配置
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")

# LLM 后端选择 ('ollama' 或 'vllm' 或其他 OpenAI 兼容 API)
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower() # 默认为 ollama

# Ollama LLM 配置 (如果 LLM_BACKEND == 'ollama')
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "cnshenyang/qwen3-nothink:14b") # 例如 llama3, qwen:7b-chat

# vLLM (或其他 OpenAI 兼容 API) 配置 (如果 LLM_BACKEND != 'ollama')
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL", "http://localhost:8000/v1") # 例如 vLLM 服务地址
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "Qwen/Qwen1.5-7B-Chat") # 模型名，会传给API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY") # 对于本地vLLM，通常不重要

# ChromaDB RAG 目录配置
CHROMA_DB_RAG_DIR_BASE = Path("./chroma_db_rag")

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

def get_student_chroma_path(student_id: str) -> Path:
    """获取指定学生ID的ChromaDB持久化路径"""
    return CHROMA_DB_RAG_DIR_BASE / student_id

# --- RAG 处理核心逻辑 ---

async def process_rag_query(student_name: str, query: str) -> str:
    logger.info(f"RAG处理开始：学生='{student_name}', 查询='{query}', LLM后端='{LLM_BACKEND}'")
    student_id = get_student_id_for_path(student_name)
    
    chroma_persist_dir = get_student_chroma_path(student_id)
    collection_name = f"rag_collection_{student_id}" # 与 build_rag_index.py 中用的集合名一致

    # 检查索引是否存在 (更可靠的检查是检查SQLite文件是否存在)
    db_file_path = chroma_persist_dir / "chroma.sqlite3"
    if not chroma_persist_dir.exists() or not db_file_path.exists():
        logger.warning(f"未找到学生 '{student_name}' (ID: {student_id}) 的持久化向量数据库: {chroma_persist_dir}。")
        error_msg = (f"抱歉，学生 {student_name} 的学习资料索引尚未建立或不完整。 "
                     f"请先运行索引程序 (例如: python build_rag_index.py --student \"{student_name}\")。")
        return error_msg

    logger.info(f"从 '{chroma_persist_dir}' 加载学生 '{student_name}' 的向量库，集合名 '{collection_name}'。")
    
    try:
        # 1. 初始化嵌入模型 (用于加载ChromaDB)
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        
        # 2. 初始化LLM客户端
        llm_client = None
        if LLM_BACKEND == 'ollama':
            logger.info(f"使用 Ollama LLM: {OLLAMA_LLM_MODEL} @ {OLLAMA_BASE_URL}")
            llm_client = ChatOllama(
                model=OLLAMA_LLM_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.3 # 可调
            )
        # 3. 加载现有的ChromaDB向量库
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=str(chroma_persist_dir),
            embedding_function=embeddings
        )
        
        db_count = vectorstore._collection.count()
        logger.info(f"向量库加载完毕。集合 '{collection_name}' 中条目数: {db_count}")

        if db_count == 0:
            logger.warning(f"学生 '{student_name}' 的向量数据库为空。这可能意味着原始文档为空或索引过程有问题。")
            return f"抱歉，学生 {student_name} 的学习资料库中没有内容，无法进行规划。"

        # 4. 创建检索器
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # k值可以调整，影响检索文档数量

        # 5. 定义Prompt模板
        prompt_template_str = """
        作为一名专业的学习规划导师，你的任务是根据下面提供的“学习资料和背景信息”，为学生“{student_name}”回答“学生的问题/请求”。

        规则：
        1. 你的回答必须完全基于提供的“学习资料和背景信息”。
        2. **请直接给出最终的、简洁的答案。**
        3. **不要包含任何思考过程、分析、解释、前缀或任何形式的元文本（如<think>标签）。**
        4. 如果资料不足以回答，请直接回答“根据所提供资料，无法回答此问题”。

        ---
        学习资料和背景信息（上下文）:
        {context}
        ---
        学生的问题/请求:
        {question}
        ---
        你的回答:
        """
        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template_str)
        
        # 6. 检索相关文档
        logger.info(f"正在为查询 '{query[:50]}...' 检索相关文档...")
        # retriever.get_relevant_documents 是同步的，用 to_thread 包装以在异步上下文中使用
        docs_retrieved = await asyncio.to_thread(retriever.get_relevant_documents, query)
        
        if not docs_retrieved:
            logger.info(f"未能检索到与查询 '{query}' 相关的文档。")
            # 可以选择直接返回，或者让LLM尝试基于无上下文回答（但通常效果不好）
            return f"抱歉，在学生 {student_name} 的学习资料中未能找到与您查询“{query}”直接相关的内容，无法给出具体规划。"
        
        logger.info(f"检索到 {len(docs_retrieved)} 个相关文档片段。")
        context_str = "\n\n---\n\n".join([doc.page_content for doc in docs_retrieved])
        
        # 7. 构建最终的Prompt并调用LLM
        final_prompt_for_llm = QA_CHAIN_PROMPT.format(
            student_name=student_name,
            context=context_str,
            question=query
        )
        
        logger.info(f"向 LLM ({LLM_BACKEND}) 发送最终提示进行生成...")
        
        # 使用异步调用
        retrieved_contexts_list = [doc.page_content for doc in docs_retrieved]

        response_obj = await llm_client.ainvoke(final_prompt_for_llm)
        answer = response_obj.content

        logger.info(f"LLM 生成的回答 (前200字符): {answer[:200]}...")
        
        # 返回一个字典，包含答案和上下文
        return {
            "answer": answer,
            "contexts": retrieved_contexts_list
        }

    except Exception as e:
        logger.error(f"RAG处理过程中发生错误: {e}", exc_info=True)
        # 避免在返回给用户的错误信息中暴露过多细节
        return f"抱歉，在为 {student_name} 进行学习规划时遇到技术问题。请稍后再试或联系管理员。"

# --- 命令行接口 ---
async def main():
    parser = argparse.ArgumentParser(description="为学生提供基于RAG的学习规划。")
    parser.add_argument("student_name", type=str, help="学生姓名")
    parser.add_argument("query", type=str, help="用户查询或规划请求")
    
    # 可以添加一个参数来覆盖环境变量中的 LLM_BACKEND
    parser.add_argument(
        "--llm_backend", 
        type=str, 
        choices=['ollama', 'vllm', 'openai_api'], # 'vllm' 和 'openai_api' 都用 ChatOpenAI
        help="覆盖LLM后端选择 (ollama, vllm, openai_api)"
    )

    args = parser.parse_args()

    # 如果通过命令行参数指定了后端，则覆盖环境变量
    global LLM_BACKEND 
    if args.llm_backend:
        if args.llm_backend == 'vllm' or args.llm_backend == 'openai_api':
            LLM_BACKEND = 'openai_compatible_api' # 内部统一标识
        else:
            LLM_BACKEND = args.llm_backend
        logger.info(f"通过命令行参数将 LLM 后端设置为: {LLM_BACKEND}")


    result = await process_rag_query(args.student_name, args.query)
    sys.stdout.write(result) # 将结果输出到标准输出，由 client.py 捕获
    sys.stdout.flush()


if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info >= (3, 8):
         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())