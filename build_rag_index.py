import asyncio
import logging
import os
import re
import sys
from pathlib import Path
import argparse

from pypinyin import Style, pinyin
from chromadb import Client as ChromaClient

# langchain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3") 

STUDENT_EMAILS = {
    "小明": "",
    "小红": "",
}


CHROMA_DB_RAG_DIR_BASE = Path("./chroma_db_rag")
STUDENT_PAPERS_DIR_BASE = Path("./student_papers")

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
    return CHROMA_DB_RAG_DIR_BASE / student_id

async def create_or_update_one_student_index(student_name: str, force_reindex: bool = False) -> bool:
    logger.info(f"开始为学生 '{student_name}' 创建/更新RAG索引...")
    student_id = get_student_id_for_path(student_name)
    
    student_folder_path = STUDENT_PAPERS_DIR_BASE / student_id
    chroma_persist_dir = get_student_chroma_path(student_id)
    collection_name = f"rag_collection_{student_id}" # 每个学生一个集合，在各自的持久化目录里

    if not student_folder_path.exists() or not student_folder_path.is_dir():
        logger.warning(f"未找到学生 '{student_name}' (ID: {student_id}) 的论文文件夹: {student_folder_path}。跳过索引。")
        return False

    pdf_files = list(student_folder_path.rglob("*.pdf"))
    txt_files = list(student_folder_path.rglob("*.txt"))

    if not pdf_files and not txt_files:
        logger.info(f"学生 '{student_name}' (ID: {student_id}) 的论文文件夹 {student_folder_path} 中没有 .pdf 或 .txt 文件。跳过索引。")
        return False

    logger.info(f"为学生 '{student_name}' (ID: {student_id}) 从文件夹 '{student_folder_path}' 加载文档...")

    try:
        documents = []
        pdf_loader_kwargs = {'extract_images': False}
        txt_loader_kwargs = {'encoding': 'utf-8'}

        if pdf_files:
            logger.debug(f"加载 {len(pdf_files)} 个 PDF 文件...")
            pdf_loader = DirectoryLoader(
                str(student_folder_path),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                loader_kwargs=pdf_loader_kwargs,
                recursive=True,
                show_progress=False,
                use_multithreading=True,
                silent_errors=True
            )
            try:
                loaded_pdfs = pdf_loader.load()
                if loaded_pdfs:
                    documents.extend(loaded_pdfs)
                    logger.debug(f"成功从 PDF 加载了 {len(loaded_pdfs)} 个文档片段。")
            except Exception as e:
                logger.error(f"加载学生 '{student_name}' 的 PDF 文件过程中发生错误: {e}", exc_info=True)
        
        if txt_files:
            logger.debug(f"加载 {len(txt_files)} 个 TXT 文件...")
            txt_loader = DirectoryLoader(
                str(student_folder_path),
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs=txt_loader_kwargs,
                recursive=True,
                show_progress=False,
                silent_errors=True
            )
            try:
                loaded_txts = txt_loader.load()
                if loaded_txts:
                    documents.extend(loaded_txts)
                    logger.debug(f"成功从 TXT 加载了 {len(loaded_txts)} 个文档片段。")
            except Exception as e:
                logger.error(f"加载学生 '{student_name}' 的 TXT 文件过程中发生错误: {e}", exc_info=True)

        if not documents:
            logger.warning(f"未能从学生 '{student_name}' 的文件夹 {student_folder_path} 加载任何文档内容。索引未创建/更新。")
            return False
        
        logger.info(f"为 '{student_name}' 总共加载了 {len(documents)} 个文档片段进行索引。")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=25) 
        texts = text_splitter.split_documents(documents)
        logger.info(f"文档切分为 {len(texts)} 个文本块。")

        if not texts:
            logger.warning(f"学生 '{student_name}' 的文档切分后没有文本块。索引未创建。")
            return False

        logger.info(f"使用嵌入模型: {EMBEDDING_MODEL} @ {OLLAMA_BASE_URL}")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


        chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"正在为学生 '{student_name}' (ID: {student_id}) 在目录 '{chroma_persist_dir}' 使用集合名 '{collection_name}' 创建/更新向量库...")
        
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=str(chroma_persist_dir) 
        )
        
        count = vectorstore._collection.count()
        logger.info(f"学生 '{student_name}' 的RAG索引创建/更新完毕。集合 '{collection_name}' 中现有 {count} 个条目。")
        return True

    except Exception as e:
        logger.error(f"为学生 '{student_name}' 创建/更新RAG索引时发生严重错误: {e}", exc_info=True)
        return False

async def main():
    parser = argparse.ArgumentParser(description="为学生文档创建或更新RAG索引。")
    parser.add_argument(
        "--student", 
        type=str, 
        help="指定要处理的学生姓名。如果未提供，且未指定 --all，则不执行任何操作。"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help=f"处理 STUDENT_EMAILS 字典中定义的所有学生。学生论文应位于 '{STUDENT_PAPERS_DIR_BASE}/<student_id>/'。"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="强制重新索引，即使认为索引已存在 (当前实现总是覆盖，此参数主要用于语义)。"
    )
    
    args = parser.parse_args()

    if not CHROMA_DB_RAG_DIR_BASE.exists():
        logger.info(f"基础RAG ChromaDB目录 '{CHROMA_DB_RAG_DIR_BASE}' 不存在，正在创建...")
        CHROMA_DB_RAG_DIR_BASE.mkdir(parents=True, exist_ok=True)
    
    if not STUDENT_PAPERS_DIR_BASE.exists():
        logger.error(f"错误：学生论文基础目录 '{STUDENT_PAPERS_DIR_BASE}' 不存在。请创建此目录并放入学生文档。")
        return

    students_to_process = []
    if args.all:
        if not STUDENT_EMAILS:
            logger.warning("指定了 --all 但 STUDENT_EMAILS 字典为空。无法确定要处理哪些学生。")
        else:
            students_to_process = list(STUDENT_EMAILS.keys())
            logger.info(f"将为以下所有学生处理索引: {', '.join(students_to_process)}")
    elif args.student:
        students_to_process.append(args.student)
        logger.info(f"将为指定学生处理索引: {args.student}")
    else:
        logger.info("未指定 --student 或 --all。请提供学生姓名或使用 --all 选项。")
        parser.print_help()
        return

    if not students_to_process:
        logger.info("没有需要处理的学生。退出。")
        return

    successful_indexes = 0
    failed_indexes = 0

    for student_name in students_to_process:
        logger.info(f"--- 开始处理学生: {student_name} ---")
        try:
            success = await create_or_update_one_student_index(student_name, force_reindex=args.force)
            if success:
                logger.info(f"+++ 学生 {student_name} 的索引成功处理。 +++")
                successful_indexes += 1
            else:
                logger.warning(f"--- 学生 {student_name} 的索引处理未成功或被跳过。 ---")
                failed_indexes +=1 # 也可能是跳过，但我们都算作未完全成功
        except Exception as e:
            logger.error(f"处理学生 {student_name} 时发生意外异常: {e}", exc_info=True)
            failed_indexes += 1
        logger.info("-" * 50)


    logger.info("===== 索引处理总结 =====")
    logger.info(f"成功处理/更新索引的学生数量: {successful_indexes}")
    logger.info(f"未能成功处理/更新索引或被跳过的学生数量: {failed_indexes}")
    logger.info("=========================")

if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info >= (3, 8):
         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())

