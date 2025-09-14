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

# --- 导入您想要使用的LLM ---
# 注意：我们不再从 ragas.llms 导入任何东西
from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama # 如果你使用Ollama，取消这行注释

# --- 导入您项目中的核心逻辑 ---
from build_rag_index import create_or_update_one_student_index
from rag_processor import process_rag_query

# --- 1. 环境与模型配置 ---

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 配置RAGAS评估时使用的语言模型
# 请确保已设置 OPENAI_API_KEY
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OpenAI API Key not found. Please set it in your .env file.")
# 直接创建LangChain模型实例，不再需要LangchainLLM包装
llm_for_evaluation = ChatOpenAI(model="qwen-plus")
# 如果使用Ollama，请替换为:
# llm_for_evaluation = ChatOllama(model="your_ollama_model")


# --- 2. RAG管道调用封装 ---

async def run_true_rag_pipeline(student_name: str, questions: List[str]) -> List[Dict[str, Any]]:
    """
    封装并调用您真实的RAG系统来处理一系列问题。
    """
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

# --- 3. 完整的“黄金”测试数据集 ---

def get_golden_dataset() -> List[Dict[str, Any]]:
    """返回包含20个问答对的完整测试集"""
    # (内容同上一版，这里为了简洁省略，实际代码中请包含完整的20个条目)
    return [
        {"question": "M2I2框架中包含了哪四种预训练目标？", "ground_truth": "M2I2框架包含四种预训练目标：遮蔽图像建模（MIM）、遮蔽语言建模（MLM）、图像文本匹配（ITM）和图像文本对比学习（ITC）。"},
        {"question": "在M³AE模型中，为什么对图像和文本使用不同的遮蔽率？", "ground_truth": "M³AE对图像和文本使用不同的遮蔽率（图像通常为75%，文本为15%）是因为两者的信息密度不同。图像具有空间冗余性，高遮蔽率能迫使模型学习更鲁棒的视觉特征；而语言是信息密集的，较低的遮蔽率足以让模型进行有效的语言理解。"},
        {"question": "PubMedCLIP是如何被创建的？它解决了什么问题？", "ground_truth": "PubMedCLIP是通过在PubMed文章中的医疗图像-文本对上微调（fine-tune）原始的CLIP模型而创建的。它旨在解决通用领域预训练模型（如CLIP）在迁移到医疗领域时可能存在的领域偏移（domain shift）问题，从而提升模型在医疗VQA任务上的性能。"},
        {"question": "UniDCP模型的核心创新是什么？它如何实现对多任务的适应性？", "ground_truth": "UniDCP模型的核心创新是引入了“动态跨模态可学习提示”。它通过一个可共享的提示空间（shareable prompt space）和查询功能，为不同的下游任务动态选择和优化最相关的提示，从而实现一个统一且可塑的框架来适应多种医疗视觉语言任务。"},
        {"question": "什么是选择性视觉问答（Selective VQA），LYP方法是如何解决这个问题的？", "ground_truth": "选择性视觉问答（Selective VQA）是指模型在面对不确定或无法正确回答的问题时，可以选择“弃权”（abstain）而不是强行给出一个错误答案。LYP（Learning from Your Peers）方法通过将训练数据分成多个子集，让在不同数据子集上训练的“同伴模型”对未知数据进行预测，从而为训练选择器（Selector）提供关于样本难度的泛化信号，帮助主模型学习何时应该弃权。"},
        {"question": "M2I2模型在VQA-RAD数据集上的总体性能相较于之前的SOTA方法（如FITS）提升了多少？", "ground_truth": "M2I2模型在VQA-RAD数据集上的总体（Overall）性能达到了73.7%，相较于FITS方法的72.4%，实现了1.3%的绝对提升。"},
        {"question": "在M2I2的消融实验中，移除哪一个预训练目标（MIM或ITC）对VQA-RAD数据集的性能影响最大？", "ground_truth": "在M2I2的消融实验中，移除图像文本对比学习（ITC）对VQA-RAD数据集的性能影响更大，性能从73.7%下降到62.3%，下降了超过11个百分点，而移除MIM则下降到65.3%。"},
        {"question": "PubMedCLIP相比原始CLIP，在MedVQA任务上的性能提升了多少？", "ground_truth": "论文指出，PubMedCLIP相较于原始CLIP在MedVQA任务上可以带来高达3%的整体准确率提升。"},
        {"question": "PMC-VQA模型与其他模型（如BLIP-2, Open-Flamingo）在零样本（zero-shot）设置下进行比较，哪个表现更好？", "ground_truth": "论文表格显示，在零样本设置下，所有模型表现都不佳，准确率接近随机猜测。这表明通用的视觉语言模型很难直接应用于专业的MedVQA任务。"},
        {"question": "在M³AE的实验中，仅使用MLM预训练和同时使用MLM+MIM预训练，VQA-RAD测试集的准确率有何不同？", "ground_truth": "在VQA-RAD测试集上，仅使用MLM预训练的模型准确率为76.05%，而同时使用MLM和MIM的模型准确率提升至77.01%。"},
        {"question": "总结一下，小明研究的这些模型（M2I2, M³AE, UniDCP）在处理多模态医疗数据时，共同的预训练策略是什么？", "ground_truth": "小明研究的模型普遍采用自监督学习作为核心预训练策略。它们都利用了大规模的图像-文本对，并通过多种预训练任务（如遮蔽语言/图像建模、图文匹配、对比学习）来学习统一的视觉-语言表示，最后再将这些表示微调到下游的医疗VQA等任务中。"},
        {"question": "从PMC-VQA和M2I2等论文来看，构建高质量的医疗VQA数据集面临哪些挑战？", "ground_truth": "构建高质量医疗VQA数据集面临的主要挑战包括：1) 数据规模有限，标注成本高昂，需要医学专家知识；2) 现有数据集的多样性不足，问题类型和答案形式受限；3) 存在捷径学习（shortcut learning）风险，即模型可能仅靠文本信息就能回答问题，而忽略了图像。"},
        {"question": "为什么在医疗VQA领域，简单的分类模型逐渐被生成式模型（如PMC-VQA中的模型）所取代？", "ground_truth": "生成式模型被认为更具优势，因为它们能提供更自然、自由形式的答案，更贴近真实的人机交互场景。而分类模型受限于一个预定义的有限答案词汇表，无法处理词汇表之外的新问题或需要详细解释的场景，灵活性较差。"},
        {"question": "小明的研究中，哪项工作明确地解决了模型“知之为知之，不知为不知”的问题？", "ground_truth": "论文《Improving Selective Visual Question Answering by Learning from Your Peers》明确解决了这个问题。它提出的LYP方法旨在让VQA模型在面对不确定或可能答错的问题时，能够智能地选择“弃权”。"},
        {"question": "根据M³AE和M2I2的研究，对比学习（Contrastive Learning）在多模态预训练中扮演了什么关键角色？", "ground_truth": "对比学习（如M2I2中的ITC）在多模态预训练中扮演着对齐（align）视觉和文本两种模态表示的关键角色。它通过拉近匹配的图文对在表示空间中的距离、推远不匹配的图文对，来学习两种模态间的深层语义关联，这对于后续的跨模态理解任务至关重要。"},
        {"question": "UniDCP模型中提到的“动态提示”与传统的静态提示（如CoOp, VPT）有何不同？", "ground_truth": "传统的静态提示为每个下游任务学习一组固定的提示，任务间不共享。而UniDCP的“动态提示”则是从一个共享的、可学习的提示池中，通过一个查询函数为每个输入样本动态地选择和组合最合适的提示，因此更加灵活和高效。"},
        {"question": "在PMC-VQA的数据生成流程中，他们是如何过滤掉那些仅凭文本常识就能回答的问题的？", "ground_truth": "他们使用了一个纯文本的问答模型（LLaMA-7B），只输入问题和选项，不给图像信息。如果模型能正确回答，就认为这个问题可能存在语言偏见或捷径，并将其从数据集中剔除。"},
        {"question": "在M2I2的微调阶段，模型的哪些部分被冻结，哪些部分被微调？", "ground_truth": "在M2I2的微调阶段，图像编码器、文本编码器和多模态编码器都从端到端进行微调。而预训练时使用的图像解码器则不用于下游任务，被丢弃了。"},
        {"question": "M³AE论文中提到，为了进行低层级的图像重构（MIM任务），它从多模态融合模块的哪一层选择视觉输出来进行解码？", "ground_truth": "M³AE论文提到，它采用多模态融合模块的中间层（k-th Transformer layer）的视觉输出来执行低层级的图像重构任务（MIM），而不是使用最终的、语义层次更高的输出。"},
        {"question": "LYP方法是如何生成用于训练“选择器”的标签的，而不需要额外的人工标注？", "ground_truth": "LYP方法通过交叉验证的方式生成标签。它将训练集分成N份，训练N个“同伴模型”，每个模型都在N-1份数据上训练。然后，每个模型在它没见过的那一份数据上做预测。通过比较这些预测和真实标签，就可以自动生成“正确”或“错误”的标签，用于训练选择器。"}
    ]

# --- 4. 评估主程序 ---

async def main():
    """主函数，组织并执行完整的端到端评估流程"""
    
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
        llm=llm_for_evaluation,      # <-- 将LLM实例直接传递给evaluate函数
        # embeddings=... # 如果需要指定嵌入模型，也可以在这里传递
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