import os
import asyncio
import logging
from typing import List, Dict, Any

from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_openai import ChatOpenAI
from build_rag_index import create_or_update_one_student_index
from rag_processor import process_rag_query

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OpenAI API Key not found. Please set it in your .env file.")
llm_for_evaluation = ChatOpenAI(model="qwen-plus")

async def run_true_rag_pipeline(student_name: str, questions: List[str]) -> List[Dict[str, Any]]:
    results = []
    logger.info(f"正在为学生'{student_name}'的RAG系统处理 {len(questions)} 个问题...")
    
    for i, question in enumerate(questions, 1):
        logger.info(f"--- 处理问题 {i}/{len(questions)}: '{question[:50]}...' ---")
        try:
            rag_output = await process_rag_query(student_name, question)
            
            if isinstance(rag_output, dict) and "answer" in rag_output and "contexts" in rag_output:
                results.append({
                    "answer": rag_output["answer"],
                    "contexts": rag_output["contexts"]
                })
            else:
                logger.error(f"process_rag_query for question '{question}' did not return the expected dictionary format. Returned: {rag_output}")
                results.append({
                    "answer": "Error: RAG pipeline returned unexpected format.",
                    "contexts": []
                })
        except Exception as e:
            logger.error(f"调用RAG管道处理问题 '{question}' 时发生错误: {e}", exc_info=True)
            results.append({
                "answer": f"Error during RAG processing: {e}",
                "contexts": []
            })
            
    logger.info("所有问题的RAG处理已完成。")
    return results

def get_golden_dataset() -> List[Dict[str, Any]]:
    return [

    ]

async def main():
    student_to_evaluate = "小明"
    
    logger.info(f"===== 步骤 1: 开始为学生 '{student_to_evaluate}' 构建/更新RAG索引 =====")
    try:
        index_success = await create_or_update_one_student_index(student_to_evaluate, force_reindex=True)
        if not index_success:
            logger.error(f"索引构建失败或被跳过。请检查 `student_papers/{student_to_evaluate}` 文件夹是否存在且包含文档。评估无法继续。")
            return
    except Exception as e:
        logger.error(f"构建索引时发生严重错误: {e}", exc_info=True)
        logger.error("请确保您的Ollama服务正在运行，并且嵌入模型可用。")
        return
    logger.info("===== 步骤 1: RAG索引构建/更新完成 =====")

    logger.info("===== 步骤 2: 准备黄金评估数据集 =====")
    golden_dataset = get_golden_dataset()
    questions = [item["question"] for item in golden_dataset]
    ground_truths = [item["ground_truth"] for item in golden_dataset]
    logger.info(f"已加载 {len(questions)} 个评估问答对。")

    logger.info("===== 步骤 3: 调用RAG管道获取模型输出 =====")
    try:
        rag_results = await run_true_rag_pipeline(student_to_evaluate, questions)
    except Exception as e:
        logger.error(f"调用RAG管道时发生严重错误: {e}", exc_info=True)
        logger.error("请确保您的Ollama服务正在运行，并且RAG处理所需的模型可用。")
        return
        
    answers = [result["answer"] for result in rag_results]
    contexts = [result["contexts"] for result in rag_results]
    
    logger.info("===== 步骤 4: 整合数据并执行RAGAS评估 =====")
    data_for_ragas = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    
    dataset = Dataset.from_dict(data_for_ragas)
    
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    
    logger.info(f"评估将使用大语言模型 '{llm_for_evaluation.model_name}' 进行打分。")
    
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm_for_evaluation,
    )
    
    logger.info("===== 步骤 5: RAGAS评估结果 =====")
    print("\n评估结果摘要:")
    print(result)
    
    print("\n详细评估结果 (Pandas DataFrame):")
    df = result.to_pandas()
    print(df.to_string())

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
