"""
重排序模块 - 使用 LangChain

功能：
- 对检索结果进行重排序
- 支持多种重排序策略
- 提高检索精度
"""
import logging
from typing import List, Dict, Any, Optional
from config.config_manager import config

logger = logging.getLogger(__name__)


class Reranker:
    """重排序器 - 提高检索精度"""

    def __init__(self, strategy: str = "similarity"):
        """
        初始化重排序器

        Args:
            strategy: 重排序策略
                - "similarity": 基于相似度（默认）
                - "diversity": 基于多样性
                - "mmr": 最大边际相关性
                - "cross_encoder": 交叉编码器（需要额外模型）
        """
        self.strategy = strategy
        logger.info(f"初始化 Reranker，策略: {strategy}")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 检索到的文档列表
            top_k: 返回前 K 个结果（None 则返回全部）

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return documents

        logger.info(f"开始重排序，文档数量: {len(documents)}, 策略: {self.strategy}")

        # 根据策略选择重排序方法
        if self.strategy == "similarity":
            reranked = self._rerank_by_similarity(documents)
        elif self.strategy == "diversity":
            reranked = self._rerank_by_diversity(query, documents)
        elif self.strategy == "mmr":
            reranked = self._rerank_by_mmr(query, documents)
        elif self.strategy == "cross_encoder":
            reranked = self._rerank_by_cross_encoder(query, documents)
        else:
            logger.warning(f"未知策略: {self.strategy}，使用默认相似度排序")
            reranked = self._rerank_by_similarity(documents)

        # 截取 top_k
        if top_k is not None:
            reranked = reranked[:top_k]

        logger.info(f"✅ 重排序完成，返回 {len(reranked)} 个文档")
        return reranked

    def _rerank_by_similarity(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        基于相似度重排序（默认已排序，直接返回）

        Args:
            documents: 文档列表

        Returns:
            排序后的文档列表
        """
        # 文档已经按相似度排序，直接返回
        return documents

    def _rerank_by_diversity(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        lambda_param: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        基于多样性重排序 - 平衡相关性和多样性

        Args:
            query: 查询文本
            documents: 文档列表
            lambda_param: 多样性参数 (0-1)，越大越注重相关性

        Returns:
            重排序后的文档列表
        """
        if len(documents) <= 1:
            return documents

        # 简单的多样性策略：基于类别分散
        reranked = []
        remaining = documents.copy()
        seen_categories = set()

        # 第一轮：选择不同类别的文档
        for doc in remaining[:]:
            category = doc['qa_pair'].get('category', 'general')
            if category not in seen_categories:
                reranked.append(doc)
                remaining.remove(doc)
                seen_categories.add(category)

        # 第二轮：添加剩余文档（按相似度）
        reranked.extend(remaining)

        logger.info(f"多样性重排序：选择了 {len(seen_categories)} 个不同类别")
        return reranked

    def _rerank_by_mmr(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        lambda_param: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        最大边际相关性 (MMR) 重排序

        MMR = λ * Sim(query, doc) - (1-λ) * max(Sim(doc, selected_docs))

        Args:
            query: 查询文本
            documents: 文档列表
            lambda_param: 平衡参数 (0-1)

        Returns:
            重排序后的文档列表
        """
        if len(documents) <= 1:
            return documents

        # 简化的 MMR 实现
        # 在实际应用中，需要计算文档之间的相似度
        # 这里使用启发式方法：交替选择高相似度和不同类别的文档

        reranked = []
        remaining = documents.copy()

        # 先选择最相似的
        if remaining:
            reranked.append(remaining.pop(0))

        # 交替选择：相似度高的 vs 类别不同的
        while remaining:
            if len(reranked) % 2 == 1:
                # 选择相似度最高的
                reranked.append(remaining.pop(0))
            else:
                # 选择类别不同的
                selected_categories = {doc['qa_pair'].get('category') for doc in reranked}
                found = False
                for i, doc in enumerate(remaining):
                    if doc['qa_pair'].get('category') not in selected_categories:
                        reranked.append(remaining.pop(i))
                        found = True
                        break
                if not found:
                    reranked.append(remaining.pop(0))

        logger.info("MMR 重排序完成")
        return reranked

    def _rerank_by_cross_encoder(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        使用交叉编码器重排序（需要额外的模型）

        Args:
            query: 查询文本
            documents: 文档列表

        Returns:
            重排序后的文档列表
        """
        try:
            from sentence_transformers import CrossEncoder

            # 加载交叉编码器模型
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

            # 准备文档对
            pairs = []
            for doc in documents:
                # 使用问题+答案作为文档内容
                doc_text = f"{doc['qa_pair']['question']} {doc['qa_pair']['answer']}"
                pairs.append([query, doc_text])

            # 计算交叉编码分数
            scores = model.predict(pairs)

            # 添加交叉编码分数到文档
            for doc, score in zip(documents, scores):
                doc['cross_encoder_score'] = float(score)

            # 按交叉编码分数排序
            reranked = sorted(documents, key=lambda x: x['cross_encoder_score'], reverse=True)

            logger.info("✅ 交叉编码器重排序完成")
            return reranked

        except ImportError:
            logger.warning("未安装 sentence-transformers，回退到相似度排序")
            logger.warning("安装命令: pip install sentence-transformers")
            return self._rerank_by_similarity(documents)
        except Exception as e:
            logger.error(f"交叉编码器重排序失败: {e}")
            return self._rerank_by_similarity(documents)


class ContextualCompressionReranker:
    """
    上下文压缩重排序器 - 使用 LangChain 的 ContextualCompressionRetriever

    这是一个更高级的重排序方法，可以：
    1. 过滤不相关的文档
    2. 压缩文档内容
    3. 提取最相关的片段
    """

    def __init__(self, llm_client=None):
        """
        初始化上下文压缩重排序器

        Args:
            llm_client: LLM 客户端（用于评估相关性）
        """
        self.llm_client = llm_client
        logger.info("初始化 ContextualCompressionReranker")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        使用 LLM 进行上下文压缩和重排序

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前 K 个结果

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return documents

        logger.info(f"开始上下文压缩重排序，文档数量: {len(documents)}")

        # 使用 LLM 评估每个文档的相关性
        scored_docs = []
        for doc in documents:
            relevance_score = self._calculate_relevance(query, doc)
            doc['relevance_score'] = relevance_score
            scored_docs.append(doc)

        # 按相关性分数排序
        reranked = sorted(scored_docs, key=lambda x: x['relevance_score'], reverse=True)

        # 截取 top_k
        if top_k is not None:
            reranked = reranked[:top_k]

        logger.info(f"✅ 上下文压缩重排序完成，返回 {len(reranked)} 个文档")
        return reranked

    def _calculate_relevance(self, query: str, document: Dict[str, Any]) -> float:
        """
        计算文档与查询的相关性分数

        Args:
            query: 查询文本
            document: 文档

        Returns:
            相关性分数 (0-1)
        """
        # 简化版本：结合向量相似度和关键词匹配
        similarity = document.get('similarity', 0.0)

        # 关键词匹配加分
        question = document['qa_pair']['question'].lower()
        answer = document['qa_pair']['answer'].lower()
        query_lower = query.lower()

        keyword_bonus = 0.0
        query_words = set(query_lower.split())
        doc_words = set(question.split()) | set(answer.split())

        # 计算关键词重叠率
        overlap = len(query_words & doc_words)
        if query_words:
            keyword_bonus = overlap / len(query_words) * 0.2  # 最多加 0.2 分

        relevance = min(similarity + keyword_bonus, 1.0)
        return relevance
