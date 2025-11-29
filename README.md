# RAG 智能问答系统

基于检索增强生成（RAG）技术的智能问答系统，通过向量检索 + LLM 生成，提供准确的问答服务。

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境变量
创建 `.env` 文件：
```env
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=1.0
```

### 3. 运行程序
```bash
python main.py
```

## 项目结构

```
QARag/
├── config/                  # 配置管理
│   ├── config.yaml         # 主配置文件
│   ├── config_manager.py   # 配置管理器
│   └── prompt_templates.py # Prompt 模板
├── src/                    # 核心代码
│   ├── data_loader.py     # 数据加载
│   ├── embedding.py       # 文本向量化
│   ├── retriever.py       # 向量检索
│   ├── llm_client.py      # LLM 调用
│   └── rag_pipeline.py    # RAG 主流程
├── tests/                 # 测试文件
├── data/                  # 数据目录
│   └── raw/qa_pairs_rag.json
├── main.py               # 程序入口
└── requirements.txt      # 依赖列表
```

## 核心功能

- **语义检索**：基于 sentence-transformers 的向量相似度检索
- **向量存储**：使用 FAISS 进行高效索引
- **上下文增强**：整合多个相关文档生成回答
- **流式输出**：支持实时流式响应
- **配置管理**：统一的配置文件和环境变量管理

## 使用示例

### 交互式问答
```bash
python main.py

# 可用命令：
# - 直接输入问题
# - help: 显示帮助
# - stats: 显示统计信息
# - exit: 退出
```

### Python API
```python
from src.rag_pipeline import RAGPipeline

# 初始化（使用配置文件默认值）
pipeline = RAGPipeline()
pipeline.initialize()

# 单个问题
result = pipeline.answer("如何申请退款？")
print(result['answer'])

# 流式输出
for chunk in pipeline.answer_stream("贷款额度如何提升？"):
    print(chunk, end="")
```

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 测试覆盖率
pytest tests/ --cov=src --cov-report=html
```

## 配置说明

### config/config.yaml
```yaml
# 检索配置
retrieval:
  top_k: 3                    # 检索文档数量
  similarity_threshold: 0.7   # 相似度阈值

# LLM 配置
llm:
  model: "gpt-4o-mini"
  temperature: 1.0
  max_tokens: 1000
```

环境变量优先级高于配置文件。

## 技术栈

| 组件 | 技术 |
|------|------|
| 向量化 | sentence-transformers |
| 向量库 | FAISS |
| LLM | OpenAI API |
| 语言 | Python 3.8+ |

## 性能

- 检索速度: < 100ms（521 条数据）
- 向量维度: 384 维
- LLM 响应: 2-4 秒

## 开发

```bash
# 安装测试依赖
pip install pytest pytest-cov

# 运行测试
pytest

# 修改配置
vi config/config.yaml

# 查看日志
tail -f logs/app.log
```

## License

MIT License
