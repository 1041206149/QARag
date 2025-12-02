"""
整合的测试文件 - 包含所有模块的测试

使用方法：
    pytest tests/test_all.py -v
    pytest tests/test_all.py::test_data_loader -v
    pytest tests/test_all.py::TestRAGPipeline -v
"""
import pytest
from pathlib import Path
from src.data_loader import QADataLoader
from src.embedding import EmbeddingModel
from src.retriever import VectorRetriever
from src.llm_client import LLMClient
from src.rag_pipeline import RAGPipeline


# ============================================
# 数据加载测试
# ============================================
class TestDataLoader:
    """数据加载模块测试"""

    def test_load_data(self):
        """测试数据加载"""
        loader = QADataLoader()
        qa_pairs = loader.load_data()

        assert len(qa_pairs) > 0, "应该加载到数据"
        assert isinstance(qa_pairs, list), "返回应该是列表"

    def test_preprocess_data(self):
        """测试数据预处理"""
        loader = QADataLoader()
        loader.load_data()
        processed = loader.preprocess_data()

        assert len(processed) > 0, "预处理后应有数据"
        assert 'combined_text' in processed[0], "应包含combined_text字段"
        assert 'question' in processed[0], "应包含question字段"
        assert 'answer' in processed[0], "应包含answer字段"

    def test_statistics(self):
        """测试统计信息"""
        loader = QADataLoader()
        loader.load_data()
        stats = loader.get_statistics()

        assert 'total_count' in stats, "应包含总数"
        assert 'categories' in stats, "应包含分类信息"
        assert stats['total_count'] > 0, "总数应大于0"


# ============================================
# 向量化测试
# ============================================
class TestEmbedding:
    """向量化模块测试"""

    @pytest.fixture
    def embedding_model(self):
        """创建向量化模型实例"""
        return EmbeddingModel()

    def test_encode_text(self, embedding_model):
        """测试文本向量化"""
        text = "这是一个测试文本"
        embedding = embedding_model.encode_text(text)

        assert embedding is not None, "应返回向量"
        assert len(embedding.shape) == 1, "应返回一维向量"
        assert embedding.shape[0] == 384, "向量维度应为384"

    def test_encode_batch(self, embedding_model):
        """测试批量向量化"""
        texts = ["文本1", "文本2", "文本3"]
        embeddings = embedding_model.encode_batch(texts)

        assert len(embeddings) == 3, "应返回3个向量"
        assert embeddings.shape[1] == 384, "向量维度应为384"


# ============================================
# 检索测试
# ============================================
class TestRetriever:
    """检索模块测试"""

    @pytest.fixture
    def retriever(self):
        """创建检索器实例"""
        embedding_model = EmbeddingModel()
        return VectorRetriever(embedding_model=embedding_model)

    @pytest.fixture
    def sample_data(self):
        """准备测试数据"""
        loader = QADataLoader()
        loader.load_data()
        return loader.preprocess_data()

    def test_build_index(self, retriever, sample_data):
        """测试构建索引"""
        embedding_model = EmbeddingModel()
        embeddings_data = embedding_model.encode_qa_pairs(sample_data)

        retriever.build_index(
            embeddings=embeddings_data['embeddings'],
            qa_pairs=embeddings_data['qa_pairs']
        )

        assert retriever.index is not None, "索引应被创建"
        assert retriever.index.ntotal > 0, "索引应包含向量"

    def test_search(self, retriever, sample_data):
        """测试搜索功能"""
        # 构建索引
        embedding_model = EmbeddingModel()
        embeddings_data = embedding_model.encode_qa_pairs(sample_data)
        retriever.build_index(
            embeddings=embeddings_data['embeddings'],
            qa_pairs=embeddings_data['qa_pairs']
        )

        # 执行搜索
        results = retriever.search("如何申请退款", top_k=3)

        assert len(results) > 0, "应返回搜索结果"
        assert len(results) <= 3, "结果数量不应超过top_k"
        assert 'similarity' in results[0], "结果应包含相似度"
        assert 'qa_pair' in results[0], "结果应包含QA对"


# ============================================
# LLM客户端测试
# ============================================
class TestLLMClient:
    """LLM客户端测试"""

    @pytest.fixture
    def llm_client(self):
        """创建LLM客户端实例"""
        return LLMClient()

    def test_build_prompt(self, llm_client):
        """测试Prompt构建"""
        question = "如何申请退款？"
        context = [
            {
                'similarity': 0.9,
                'qa_pair': {
                    'question': '退款流程是什么？',
                    'answer': '在订单页面点击退款按钮'
                }
            }
        ]

        prompt = llm_client._build_rag_prompt(question, llm_client._format_context(context))

        assert question in prompt, "Prompt应包含问题"
        assert '退款' in prompt, "Prompt应包含上下文内容"

    @pytest.mark.skip(reason="需要真实的API key才能测试")
    def test_generate(self, llm_client):
        """测试LLM生成（需要API key）"""
        context = []
        response = llm_client.generate_with_context("测试问题", context)
        assert isinstance(response, str), "应返回字符串"


# ============================================
# RAG Pipeline集成测试
# ============================================
class TestRAGPipeline:
    """RAG Pipeline集成测试"""

    @pytest.fixture(scope="class")
    def pipeline(self):
        """创建Pipeline实例（类级别，只初始化一次）"""
        pipeline = RAGPipeline(top_k=3, similarity_threshold=0.7, use_cache=True)
        pipeline.initialize()
        return pipeline

    def test_initialization(self, pipeline):
        """测试初始化"""
        assert pipeline._initialized, "Pipeline应该被初始化"
        assert pipeline.retriever.index is not None, "索引应该被创建"

    def test_statistics(self, pipeline):
        """测试统计信息"""
        stats = pipeline.get_statistics()

        assert 'vector_count' in stats, "应包含向量数量"
        assert 'embedding_dim' in stats, "应包含向量维度"
        assert stats['embedding_dim'] == 384, "向量维度应为384"

    def test_answer(self, pipeline):
        """测试单个问题回答"""
        question = "如何申请退款？"
        result = pipeline.answer(question, return_context=True)

        assert 'answer' in result, "应返回答案"
        assert 'retrieved_count' in result, "应返回检索数量"
        assert 'top_similarity' in result, "应返回最高相似度"
        assert isinstance(result['answer'], str), "答案应为字符串"

    def test_answer_low_similarity(self, pipeline):
        """测试低相似度问题"""
        question = "今天天气怎么样？"  # 不相关的问题
        result = pipeline.answer(question)

        # 应返回未找到信息的提示
        assert '未找到相关信息' in result['answer'] or result['retrieved_count'] == 0

    def test_batch_answer(self, pipeline):
        """测试批量回答"""
        questions = [
            "如何申请退款？",
            "贷款审批需要多长时间？"
        ]
        results = pipeline.batch_answer(questions, top_k=2)

        assert len(results) == len(questions), "结果数量应与问题数量一致"
        for result in results:
            assert 'question' in result, "每个结果应包含问题"
            assert 'answer' in result, "每个结果应包含答案"

    @pytest.mark.skip(reason="需要真实的LLM API才能完整测试")
    def test_answer_stream(self, pipeline):
        """测试流式输出"""
        question = "如何申请退款？"
        chunks = list(pipeline.answer_stream(question))

        assert len(chunks) > 0, "应该产生输出块"
        assert all(isinstance(chunk, str) for chunk in chunks), "所有块应为字符串"


# ============================================
# 运行所有测试
# ============================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
