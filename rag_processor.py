import argparse
import asyncio
import logging
import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils import get_student_chroma_path, get_student_id_for_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "")
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "")
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL", "")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

async def process_rag_query(student_name: str, query: str) -> dict:
    logger.info(f"RAG处理开始：学生='{student_name}', 查询='{query}', LLM后端='{LLM_BACKEND}'")
    student_id = get_student_id_for_path(student_name)
    
    chroma_persist_dir = get_student_chroma_path(student_id)
    collection_name = f"rag_collection_{student_id}"

    db_file_path = chroma_persist_dir / "chroma.sqlite3"
    if not db_file_path.exists():
        logger.warning(f"未找到学生 '{student_name}' (ID: {student_id}) 的持久化向量数据库: {chroma_persist_dir}。")
        error_msg = (f"抱歉，学生 {student_name} 的学习资料索引尚未建立或不完整。 "
                     f"请先运行索引程序。")
        return {"answer": error_msg, "contexts": []}

    logger.info(f"从 '{chroma_persist_dir}' 加载学生 '{student_name}' 的向量库，集合名 '{collection_name}'。")
    
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        
        llm_client = None
        if LLM_BACKEND == 'ollama':
            logger.info(f"使用 Ollama LLM: {OLLAMA_LLM_MODEL} @ {OLLAMA_BASE_URL}")
            llm_client = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.3)
        else:
            # This part can be expanded to support other backends like vLLM via ChatOpenAI
            logger.error(f"LLM backend '{LLM_BACKEND}' is not implemented in this version.")
            raise NotImplementedError(f"LLM backend '{LLM_BACKEND}' is not supported.")

        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=str(chroma_persist_dir),
            embedding_function=embeddings
        )
        
        db_count = vectorstore._collection.count()
        logger.info(f"向量库加载完毕。集合 '{collection_name}' 中条目数: {db_count}")

        if db_count == 0:
            logger.warning(f"学生 '{student_name}' 的向量数据库为空。")
            return {"answer": f"抱歉，学生 {student_name} 的学习资料库中没有内容，无法进行规划。", "contexts": []}

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

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
        
        logger.info(f"正在为查询 '{query[:50]}...' 检索相关文档...")
        docs_retrieved = await asyncio.to_thread(retriever.get_relevant_documents, query)
        
        if not docs_retrieved:
            logger.info(f"未能检索到与查询 '{query}' 相关的文档。")
            return {
                "answer": f"抱歉，在学生 {student_name} 的学习资料中未能找到与您查询“{query}”直接相关的内容。",
                "contexts": []
            }
        
        logger.info(f"检索到 {len(docs_retrieved)} 个相关文档片段。")
        context_str = "\n\n---\n\n".join([doc.page_content for doc in docs_retrieved])
        
        final_prompt_for_llm = QA_CHAIN_PROMPT.format(
            student_name=student_name,
            context=context_str,
            question=query
        )
        
        logger.info(f"向 LLM ({LLM_BACKEND}) 发送最终提示进行生成...")
        
        retrieved_contexts_list = [doc.page_content for doc in docs_retrieved]
        response_obj = await llm_client.ainvoke(final_prompt_for_llm)
        answer = response_obj.content

        logger.info(f"LLM 生成的回答 (前200字符): {answer[:200]}...")
        
        return {
            "answer": answer,
            "contexts": retrieved_contexts_list
        }

    except Exception as e:
        logger.error(f"RAG处理过程中发生错误: {e}", exc_info=True)
        return {
            "answer": f"抱歉，在为 {student_name} 进行学习规划时遇到技术问题。",
            "contexts": []
        }

async def main():
    parser = argparse.ArgumentParser(description="为学生提供基于RAG的学习规划。")
    parser.add_argument("student_name", type=str, help="学生姓名")
    parser.add_argument("query", type=str, help="用户查询或规划请求")
    
    parser.add_argument(
        "--llm_backend", 
        type=str, 
        choices=['ollama', 'vllm', 'openai_api'],
        help="覆盖LLM后端选择"
    )

    args = parser.parse_args()

    global LLM_BACKEND 
    if args.llm_backend:
        LLM_BACKEND = args.llm_backend
        logger.info(f"通过命令行参数将 LLM 后端设置为: {LLM_BACKEND}")

    result = await process_rag_query(args.student_name, args.query)
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    sys.stdout.flush()


if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info >= (3, 8):
         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
