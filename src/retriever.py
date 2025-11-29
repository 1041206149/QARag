"""
向量检索模块 - 使用 LangChain VectorStore

功能：
- 使用 LangChain 的 FAISS VectorStore
- 实现语义相似度搜索
- 返回 Top-K 相关文档
"""
import logging
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config.config_manager import config

logger = logging.getLogger(__name__)


class VectorRetriever:
    """基于 LangChain FAISS 的向量检索器"""

    def __init__(self, embedding_model=None):
        """
        初始化检索器

        Args:
            embedding_model: EmbeddingModel 实例（包含 LangChain embeddings）
        """
        self.embedding_model = embedding_model
        self.vectorstore = None
        self.qa_pairs = []

        # 使用配置管理器获取索引路径
        self.index_path = config.get_path('vector_store.index_path')
        self.index_path.mkdir(parents=True, exist_ok=True)

        logger.info("初始化 VectorRetriever")

    def build_index(self, embeddings: Any, qa_pairs: List[Dict[str, Any]]):
        """
        构建 FAISS 索引

        Args:
            embeddings: 向量矩阵（兼容旧接口，实际使用 embedding_model）
            qa_pairs: 问答对列表
        """
        if self.embedding_model is None or self.embedding_model.embeddings is None:
            raise ValueError("需要 embedding_model 来构建索引")

        try:
            self.qa_pairs = qa_pairs
            logger.info(f"正在构建 FAISS 索引，文档数量: {len(qa_pairs)}")

            # 将 QA 对转换为 LangChain Document
            documents = []
            for i, qa in enumerate(qa_pairs):
                doc = Document(
                    page_content=qa.get('combined_text', qa['question']),
                    metadata={
                        'id': qa.get('id', i),
                        'question': qa['question'],
                        'answer': qa['answer'],
                        'category': qa.get('category', 'general'),
                        'keywords': qa.get('keywords', [])
                    }
                )
                documents.append(doc)

            # 使用 LangChain FAISS 创建向量存储
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_model.embeddings
            )

            logger.info(f"✅ FAISS 索引构建完成，包含 {len(documents)} 个文档")

        except Exception as e:
            logger.error(f"❌ 构建索引失败: {e}")
            raise

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        语义搜索

        Args:
            query: 查询文本
            top_k: 返回前 K 个结果

        Returns:
            相关文档列表，包含相似度分数
        """
        if self.vectorstore is None:
            raise ValueError("索引未构建，请先调用 build_index()")

        try:
            # 使用 LangChain 的相似度搜索（带分数）
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query=query,
                k=top_k
            )

            # 构建结果
            results = []
            for i, (doc, score) in enumerate(docs_with_scores):
                # FAISS 返回的是距离，需要转换为相似度
                # 对于归一化向量，L2距离和余弦相似度的关系：similarity = 1 - distance^2 / 2
                similarity = 1 - (score / 2)

                result = {
                    'rank': i + 1,
                    'qa_pair': {
                        'id': doc.metadata.get('id'),
                        'question': doc.metadata.get('question'),
                        'answer': doc.metadata.get('answer'),
                        'category': doc.metadata.get('category'),
                        'keywords': doc.metadata.get('keywords', [])
                    },
                    'distance': float(score),
                    'similarity': float(similarity)
                }
                results.append(result)

            logger.info(f"✅ 检索到 {len(results)} 个相关文档")
            return results

        except Exception as e:
            logger.error(f"❌ 检索失败: {e}")
            raise

    def save_index(self, filename: str = "faiss_index"):
        """
        保存 FAISS 索引

        Args:
            filename: 索引目录名
        """
        if self.vectorstore is None:
            raise ValueError("索引未构建")

        try:
            index_dir = self.index_path / filename
            self.vectorstore.save_local(str(index_dir))
            logger.info(f"✅ 索引已保存到: {index_dir}")

        except Exception as e:
            logger.error(f"❌ 保存索引失败: {e}")
            raise

    def load_index(self, filename: str = "faiss_index"):
        """
        加载 FAISS 索引

        Args:
            filename: 索引目录名
        """
        if self.embedding_model is None or self.embedding_model.embeddings is None:
            raise ValueError("需要 embedding_model 来加载索引")

        try:
            index_dir = self.index_path / filename
            if not index_dir.exists():
                raise FileNotFoundError(f"索引目录不存在: {index_dir}")

            # 加载 LangChain FAISS 索引
            self.vectorstore = FAISS.load_local(
                folder_path=str(index_dir),
                embeddings=self.embedding_model.embeddings,
                allow_dangerous_deserialization=True  # 允许加载本地索引
            )

            logger.info(f"✅ 索引已加载")

        except Exception as e:
            logger.error(f"❌ 加载索引失败: {e}")
            raise

    @property
    def index(self):
        """兼容旧接口：返回 vectorstore"""
        return self.vectorstore

    @property
    def embedding_dim(self):
        """兼容旧接口：返回向量维度"""
        if self.embedding_model and self.embedding_model.embeddings:
            # 获取一个测试向量来确定维度
            test_embedding = self.embedding_model.embeddings.embed_query("test")
            return len(test_embedding)
        return 384  # 默认维度
