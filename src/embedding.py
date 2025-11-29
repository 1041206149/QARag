"""
文本向量化模块 - 使用 LangChain Embeddings

功能：
- 使用 LangChain 的 HuggingFaceEmbeddings 进行文本向量化
- 支持中文语义编码
- 缓存 embedding 结果
"""
import logging
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.config_manager import config

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """文本向量化模型 - LangChain 实现"""

    def __init__(self, model_name: Optional[str] = None):
        """
        初始化向量化模型

        Args:
            model_name: 使用的模型名称（可选，默认使用配置文件）
        """
        embedding_config = config.embedding_config
        self.model_name = model_name or embedding_config.get('model_name')
        self.embeddings = None

        # 使用配置管理器获取缓存目录
        self.cache_dir = config.get_path('embedding.cache_dir')
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"初始化 Embedding 模型: {self.model_name}")

    def load_model(self):
        """加载向量化模型"""
        if self.embeddings is not None:
            return

        try:
            logger.info(f"正在加载模型: {self.model_name}")

            # 使用 LangChain 的 HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            logger.info("✅ 模型加载完成")
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise

    def encode_text(self, text: str) -> np.ndarray:
        """
        编码单个文本

        Args:
            text: 文本字符串

        Returns:
            向量数组
        """
        if self.embeddings is None:
            self.load_model()

        try:
            embedding = self.embeddings.embed_query(text)
            return np.array(embedding)
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            raise

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        批量编码文本

        Args:
            texts: 文本列表

        Returns:
            向量矩阵 (n_texts, embedding_dim)
        """
        if self.embeddings is None:
            self.load_model()

        try:
            logger.info(f"开始编码 {len(texts)} 条文本...")
            embeddings = self.embeddings.embed_documents(texts)
            embeddings_array = np.array(embeddings)
            logger.info(f"✅ 编码完成，向量维度: {embeddings_array.shape}")
            return embeddings_array
        except Exception as e:
            logger.error(f"批量编码失败: {e}")
            raise

    def encode_qa_pairs(self, qa_pairs: List[Dict[str, Any]], use_combined: bool = True) -> Dict[str, Any]:
        """
        编码问答对

        Args:
            qa_pairs: 问答对列表
            use_combined: 是否使用合并的问题+答案文本

        Returns:
            包含向量和元数据的字典
        """
        # 提取文本
        if use_combined:
            texts = [qa['combined_text'] for qa in qa_pairs]
        else:
            texts = [qa['question'] for qa in qa_pairs]

        # 生成向量
        embeddings = self.encode_batch(texts)

        # 构建结果
        result = {
            'embeddings': embeddings,
            'qa_pairs': qa_pairs,
            'model_name': self.model_name,
            'embedding_dim': embeddings.shape[1],
            'total_count': len(qa_pairs)
        }

        return result

    def save_embeddings(self, embeddings_data: Dict[str, Any], filename: str = "embeddings.pkl"):
        """
        保存向量到文件

        Args:
            embeddings_data: 向量数据字典
            filename: 保存的文件名
        """
        filepath = self.cache_dir / filename
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(embeddings_data, f)
            logger.info(f"✅ 向量已保存到: {filepath}")
        except Exception as e:
            logger.error(f"❌ 保存向量失败: {e}")
            raise

    def load_embeddings(self, filename: str = "embeddings.pkl") -> Dict[str, Any]:
        """
        从文件加载向量

        Args:
            filename: 文件名

        Returns:
            向量数据字典
        """
        filepath = self.cache_dir / filename
        try:
            if not filepath.exists():
                raise FileNotFoundError(f"向量文件不存在: {filepath}")

            with open(filepath, 'rb') as f:
                embeddings_data = pickle.load(f)

            logger.info(f"✅ 已加载 {embeddings_data['total_count']} 条向量")
            return embeddings_data
        except Exception as e:
            logger.error(f"❌ 加载向量失败: {e}")
            raise
