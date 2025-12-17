"""
RAG主流程模块 - 使用 LangChain

功能：
- 使用 LangChain 的 RAG 组件
- 检索 → 生成流程
- 支持流式和批量处理
"""
import logging
from typing import List, Dict, Any, Optional, Generator
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config.config_manager import config
from config.prompt_templates import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE
from src.data_loader import QADataLoader
from src.embedding import EmbeddingModel
from src.retriever import VectorRetriever
from src.llm_client import LLMClient
from src.reranker import Reranker
from src.query_enhancer import MultiQueryRetriever, QueryPreprocessor
from src.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG问答系统核心流程 - LangChain 实现"""

    def __init__(
        self,
        data_path: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        use_cache: Optional[bool] = None
    ):
        """
        初始化RAG Pipeline

        Args:
            data_path: 数据文件路径（可选，默认使用配置文件）
            embedding_model_name: 向量化模型名称（可选，默认使用配置文件）
            top_k: 检索返回的文档数量（可选，默认使用配置文件）
            similarity_threshold: 相似度阈值（可选，默认使用配置文件）
            use_cache: 是否使用缓存的向量（可选，默认使用配置文件）
        """
        # 使用配置管理器获取默认值
        retrieval_config = config.retrieval_config

        self.data_path = data_path
        self.top_k = top_k if top_k is not None else retrieval_config.get('top_k', 5)
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else retrieval_config.get('similarity_threshold', 0.7)
        self.use_cache = use_cache if use_cache is not None else retrieval_config.get('use_cache', True)

        # 重排序配置
        rerank_config = retrieval_config.get('rerank', {})
        self.rerank_enabled = rerank_config.get('enabled', False)
        self.rerank_strategy = rerank_config.get('strategy', 'mmr')
        self.rerank_top_k = rerank_config.get('top_k', 3)

        # 查询增强配置
        query_enhancement_config = retrieval_config.get('query_enhancement', {})
        self.query_enhancement_enabled = query_enhancement_config.get('enabled', False)
        multi_query_config = query_enhancement_config.get('multi_query', {})
        self.multi_query_enabled = multi_query_config.get('enabled', False)
        self.num_queries = multi_query_config.get('num_queries', 3)

        # 混合检索配置
        hybrid_config = retrieval_config.get('hybrid_search', {})
        self.hybrid_enabled = hybrid_config.get('enabled', False)
        self.hybrid_strategy = hybrid_config.get('strategy', 'weighted')
        self.vector_weight = hybrid_config.get('vector_weight', 0.7)
        self.bm25_weight = hybrid_config.get('bm25_weight', 0.3)

        # 初始化组件
        logger.info("正在初始化RAG Pipeline...")
        self.data_loader = QADataLoader(data_path)
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.retriever = VectorRetriever(embedding_model=self.embedding_model)
        self.llm_client = LLMClient()

        # 初始化查询增强器
        if self.query_enhancement_enabled and self.multi_query_enabled:
            self.multi_query_retriever = MultiQueryRetriever(
                llm_client=self.llm_client,
                num_queries=self.num_queries
            )
            logger.info(f"✅ 多查询生成已启用，生成 {self.num_queries} 个查询变体")
        else:
            self.multi_query_retriever = None
            logger.info("多查询生成未启用")

        # 初始化混合检索器
        if self.hybrid_enabled:
            self.hybrid_retriever = HybridRetriever(
                vector_retriever=self.retriever,
                vector_weight=self.vector_weight,
                bm25_weight=self.bm25_weight
            )
            logger.info(f"✅ 混合检索已启用，策略: {self.hybrid_strategy}")
        else:
            self.hybrid_retriever = None
            logger.info("混合检索未启用")

        # 初始化重排序器
        if self.rerank_enabled:
            self.reranker = Reranker(strategy=self.rerank_strategy)
            logger.info(f"✅ 重排序已启用，策略: {self.rerank_strategy}")
        else:
            self.reranker = None
            logger.info("重排序未启用")

        self._initialized = False
        self.rag_chain = None

    def initialize(self):
        """初始化Pipeline（加载/构建向量索引）"""
        if self._initialized:
            logger.info("Pipeline已初始化")
            return

        try:
            # 尝试加载缓存的向量索引
            if self.use_cache:
                try:
                    logger.info("尝试加载缓存的向量索引...")
                    # 先加载 embedding 模型
                    self.embedding_model.load_model()
                    # 再加载索引
                    self.retriever.load_index(filename="faiss_index")

                    # 如果启用混合检索，需要构建 BM25 索引
                    if self.hybrid_enabled and self.hybrid_retriever:
                        logger.info("混合检索已启用，正在构建 BM25 索引...")
                        # 加载数据用于构建 BM25 索引
                        self.data_loader.load_data()
                        qa_pairs = self.data_loader.preprocess_data()
                        self.hybrid_retriever.build_bm25_index(qa_pairs)
                        logger.info("✅ BM25 索引构建完成")

                    self._initialized = True
                    logger.info("✅ 从缓存加载向量索引成功")
                    return
                except Exception as e:
                    logger.warning(f"加载缓存失败: {e}，将重新构建索引")

            # 重新构建索引
            logger.info("开始构建向量索引...")

            # 1. 加载数据
            logger.info("步骤1: 加载数据...")
            self.data_loader.load_data()
            qa_pairs = self.data_loader.preprocess_data()

            # 2. 加载 embedding 模型
            logger.info("步骤2: 加载 Embedding 模型...")
            self.embedding_model.load_model()

            # 3. 构建 FAISS 索引（LangChain 会自动生成向量）
            logger.info("步骤3: 构建 FAISS 索引...")
            self.retriever.build_index(
                embeddings=None,  # LangChain 会自动处理
                qa_pairs=qa_pairs
            )

            # 4. 保存索引
            self.retriever.save_index(filename="faiss_index")

            # 5. 如果启用混合检索，构建 BM25 索引
            if self.hybrid_enabled and self.hybrid_retriever:
                logger.info("步骤5: 构建 BM25 索引...")
                self.hybrid_retriever.build_bm25_index(qa_pairs)

            self._initialized = True
            logger.info("✅ RAG Pipeline初始化完成")

        except Exception as e:
            logger.error(f"❌ Pipeline初始化失败: {e}")
            raise

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_context: bool = False
    ) -> Dict[str, Any]:
        """
        回答问题（非流式）

        Args:
            question: 用户问题
            top_k: 检索数量（None则使用默认值）
            return_context: 是否返回检索到的上下文

        Returns:
            包含答案和元数据的字典
        """
        if not self._initialized:
            self.initialize()

        try:
            # 1. 查询预处理
            question = QueryPreprocessor.preprocess(question)
            logger.info(f"正在检索: {question}")

            # 2. 生成多查询（如果启用）
            queries = [question]
            if self.multi_query_enabled and self.multi_query_retriever:
                logger.info("正在生成查询变体...")
                queries = self.multi_query_retriever.generate_queries(question)
                logger.info(f"生成了 {len(queries)} 个查询变体")

            # 3. 检索相关文档
            k = top_k or self.top_k
            all_retrieved_docs = []

            for query in queries:
                # 选择检索器
                if self.hybrid_enabled and self.hybrid_retriever:
                    retrieved_docs = self.hybrid_retriever.search(
                        query=query,
                        top_k=k,
                        strategy=self.hybrid_strategy
                    )
                else:
                    retrieved_docs = self.retriever.search(query, top_k=k)

                all_retrieved_docs.extend(retrieved_docs)

            # 4. 合并去重（基于文档ID）
            unique_docs = {}
            for doc in all_retrieved_docs:
                doc_id = doc['qa_pair']['id']
                if doc_id not in unique_docs:
                    unique_docs[doc_id] = doc
                else:
                    # 保留更高的分数
                    if self.hybrid_enabled:
                        if doc.get('hybrid_score', 0) > unique_docs[doc_id].get('hybrid_score', 0):
                            unique_docs[doc_id] = doc
                    else:
                        if doc.get('similarity', 0) > unique_docs[doc_id].get('similarity', 0):
                            unique_docs[doc_id] = doc

            # 转换回列表并按分数排序
            merged_docs = list(unique_docs.values())
            if self.hybrid_enabled:
                merged_docs.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
            else:
                merged_docs.sort(key=lambda x: x.get('similarity', 0), reverse=True)

            # 取 Top-K
            merged_docs = merged_docs[:k]

            # 5. 过滤低相似度文档
            filtered_docs = [
                doc for doc in merged_docs
                if doc.get('similarity', 0) >= self.similarity_threshold
            ]

            # 6. 重排序
            if self.rerank_enabled and self.reranker and filtered_docs:
                logger.info(f"正在重排序，策略: {self.rerank_strategy}")
                filtered_docs = self.reranker.rerank(
                    query=question,
                    documents=filtered_docs,
                    top_k=self.rerank_top_k
                )
                logger.info(f"重排序后文档数量: {len(filtered_docs)}")

            if not filtered_docs:
                logger.warning("未找到相似度足够高的文档")
                if merged_docs:
                    top_score = merged_docs[0].get('similarity', 0) or merged_docs[0].get('hybrid_score', 0)
                    logger.info(f"最高分数: {top_score:.4f} (阈值: {self.similarity_threshold})")

                return {
                    'answer': "抱歉，我在知识库中没有找到相关信息来回答您的问题。",
                    'context': [],
                    'retrieved_count': 0,
                    'top_similarity': top_score if merged_docs else 0.0
                }

            logger.info(f"检索到 {len(filtered_docs)} 个相关文档")

            # 7. 使用LLM生成回答
            logger.info("正在生成回答...")
            answer = self.llm_client.generate_with_context(
                question=question,
                context=filtered_docs
            )

            # 8. 构建结果
            result = {
                'answer': answer,
                'retrieved_count': len(filtered_docs),
                'top_similarity': filtered_docs[0].get('similarity', 0) if filtered_docs else 0.0
            }

            if return_context:
                result['context'] = filtered_docs

            logger.info("✅ 回答生成完成")
            return result

        except Exception as e:
            logger.error(f"❌ 回答问题失败: {e}")
            raise

    def answer_stream(
        self,
        question: str,
        top_k: Optional[int] = None
    ) -> Generator[str, None, None]:
        """
        回答问题（流式）

        Args:
            question: 用户问题
            top_k: 检索数量

        Yields:
            答案文本片段
        """
        if not self._initialized:
            self.initialize()

        try:
            # 1. 查询预处理
            question = QueryPreprocessor.preprocess(question)
            logger.info(f"正在检索: {question}")

            # 2. 生成多查询（如果启用）
            queries = [question]
            if self.multi_query_enabled and self.multi_query_retriever:
                logger.info("正在生成查询变体...")
                queries = self.multi_query_retriever.generate_queries(question)
                logger.info(f"生成了 {len(queries)} 个查询变体")

            # 3. 检索相关文档
            k = top_k or self.top_k
            all_retrieved_docs = []

            for query in queries:
                # 选择检索器
                if self.hybrid_enabled and self.hybrid_retriever:
                    retrieved_docs = self.hybrid_retriever.search(
                        query=query,
                        top_k=k,
                        strategy=self.hybrid_strategy
                    )
                else:
                    retrieved_docs = self.retriever.search(query, top_k=k)

                all_retrieved_docs.extend(retrieved_docs)

            # 4. 合并去重（基于文档ID）
            unique_docs = {}
            for doc in all_retrieved_docs:
                doc_id = doc['qa_pair']['id']
                if doc_id not in unique_docs:
                    unique_docs[doc_id] = doc
                else:
                    # 保留更高的分数
                    if self.hybrid_enabled:
                        if doc.get('hybrid_score', 0) > unique_docs[doc_id].get('hybrid_score', 0):
                            unique_docs[doc_id] = doc
                    else:
                        if doc.get('similarity', 0) > unique_docs[doc_id].get('similarity', 0):
                            unique_docs[doc_id] = doc

            # 转换回列表并按分数排序
            merged_docs = list(unique_docs.values())
            if self.hybrid_enabled:
                merged_docs.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
            else:
                merged_docs.sort(key=lambda x: x.get('similarity', 0), reverse=True)

            # 取 Top-K
            merged_docs = merged_docs[:k]

            # 5. 过滤低相似度文档
            filtered_docs = [
                doc for doc in merged_docs
                if doc.get('similarity', 0) >= self.similarity_threshold
            ]

            # 6. 重排序
            if self.rerank_enabled and self.reranker and filtered_docs:
                logger.info(f"正在重排序，策略: {self.rerank_strategy}")
                filtered_docs = self.reranker.rerank(
                    query=question,
                    documents=filtered_docs,
                    top_k=self.rerank_top_k
                )

            if not filtered_docs:
                yield "抱歉，我在知识库中没有找到相关信息来回答您的问题。"
                return

            logger.info(f"检索到 {len(filtered_docs)} 个相关文档")

            # 7. 构建上下文和 Prompt
            context_text = self.llm_client._format_context(filtered_docs)
            prompt = self.llm_client._build_rag_prompt(question, context_text)

            # 8. 流式生成
            logger.info("正在生成回答（流式）...")
            for chunk in self.llm_client.generate_stream(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT
            ):
                yield chunk

        except Exception as e:
            logger.error(f"❌ 流式回答失败: {e}")
            raise

    def batch_answer(
        self,
        questions: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        批量回答问题

        Args:
            questions: 问题列表
            top_k: 检索数量

        Returns:
            答案列表
        """
        if not self._initialized:
            self.initialize()

        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"正在处理第 {i}/{len(questions)} 个问题")
            try:
                result = self.answer(question, top_k=top_k)
                result['question'] = question
                results.append(result)
            except Exception as e:
                logger.error(f"处理问题失败: {question}, 错误: {e}")
                results.append({
                    'question': question,
                    'answer': f"处理失败: {str(e)}",
                    'error': True
                })

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取Pipeline统计信息

        Returns:
            统计信息字典
        """
        if not self._initialized:
            self.initialize()

        # 确保数据已加载
        if not self.data_loader.qa_pairs:
            try:
                logger.info("加载数据以获取统计信息...")
                self.data_loader.load_data()
            except Exception as e:
                logger.warning(f"无法加载数据统计: {e}")
                return {
                    'data_stats': {},
                    'vector_count': 0,
                    'embedding_dim': self.retriever.embedding_dim,
                    'model_name': self.embedding_model.model_name,
                    'llm_model': self.llm_client.model
                }

        # 获取向量数量
        vector_count = 0
        if self.retriever.vectorstore:
            try:
                # LangChain FAISS 的 vectorstore 有 index 属性
                if hasattr(self.retriever.vectorstore, 'index'):
                    vector_count = self.retriever.vectorstore.index.ntotal
            except:
                pass

        stats = {
            'data_stats': self.data_loader.get_statistics(),
            'vector_count': vector_count,
            'embedding_dim': self.retriever.embedding_dim,
            'model_name': self.embedding_model.model_name,
            'llm_model': self.llm_client.model
        }

        return stats
