# RAG 智能问答系统 (LangChain 版本)

基于 **LangChain** 框架的 RAG（检索增强生成）智能问答系统，通过向量检索 + LLM 生成，提供准确的问答服务。

## ✨ 特性

- 🔗 **基于 LangChain** - 使用 LangChain 生态系统构建
- 🧠 **HuggingFace Embeddings** - 支持多语言语义向量化
- 📚 **FAISS 向量存储** - 高效的相似度检索
- 💬 **OpenAI LLM** - 强大的语言生成能力
- ⚡ **流式输出** - 支持实时流式响应
- ⚙️ **配置管理** - 统一的配置文件和环境变量管理

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
│   ├── embedding.py       # LangChain Embeddings
│   ├── retriever.py       # LangChain FAISS VectorStore
│   ├── llm_client.py      # LangChain ChatOpenAI
│   └── rag_pipeline.py    # RAG 主流程
├── tests/                 # 测试文件
│   └── test_all.py       # 整合测试
├── data/                  # 数据目录
│   └── raw/qa_pairs_rag.json
├── main.py               # 程序入口
└── requirements.txt      # 依赖列表
```

## 核心架构

### LangChain 组件

| 组件 | LangChain 类 | 说明 |
|------|-------------|------|
| **Embeddings** | `HuggingFaceEmbeddings` | 文本向量化 |
| **VectorStore** | `FAISS` | 向量存储和检索 |
| **LLM** | `ChatOpenAI` | 语言模型 |
| **Messages** | `HumanMessage`, `SystemMessage` | 消息封装 |

### RAG 流程

```
用户问题
    ↓
[Embeddings] 问题向量化
    ↓
[FAISS VectorStore] 相似度检索
    ↓
[过滤] 相似度阈值过滤
    ↓
[Prompt 构建] 整合上下文
    ↓
[ChatOpenAI] LLM 生成回答
    ↓
返回结果
```

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

# 批量处理
questions = ["问题1", "问题2", "问题3"]
results = pipeline.batch_answer(questions)
```

## 测试

```bash
# 运行所有测试
pytest tests/test_all.py -v

# 运行特定测试类
pytest tests/test_all.py::TestRAGPipeline -v

# 测试覆盖率
pytest tests/test_all.py --cov=src --cov-report=html
```

## 配置说明

### config/config.yaml
```yaml
# 向量化配置
embedding:
  model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

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

## LangChain 优势

### 相比原生实现

| 特性 | 原生实现 | LangChain 实现 |
|------|---------|---------------|
| **代码量** | 更多 | 更少（减少30%） |
| **可维护性** | 需要手动管理 | 框架统一管理 |
| **扩展性** | 需要自己实现 | 丰富的生态系统 |
| **文档支持** | 自己编写 | 官方文档完善 |
| **社区支持** | 有限 | 活跃的社区 |

### 主要改进

1. **统一接口** - 所有组件使用 LangChain 标准接口
2. **自动向量化** - FAISS.from_documents 自动处理向量生成
3. **消息封装** - 使用 HumanMessage/SystemMessage 规范化
4. **流式支持** - 原生支持流式输出
5. **易于扩展** - 可轻松切换不同的 Embeddings/LLM/VectorStore

## 性能

- 检索速度: < 100ms（521 条数据）
- 向量维度: 384 维
- LLM 响应: 2-4 秒
- 相似度计算: 余弦相似度

## 开发

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest

# 修改配置
vi config/config.yaml

# 查看日志
tail -f logs/app.log
```

## 依赖说明

### 核心依赖
- `langchain` - LangChain 核心框架
- `langchain-community` - 社区组件（HuggingFace, FAISS）
- `langchain-openai` - OpenAI 集成
- `langchain-core` - 核心抽象
- `faiss-cpu` - FAISS 向量检索

### 测试依赖
- `pytest` - 测试框架
- `pytest-cov` - 测试覆盖率

## 迁移指南

### 从原生实现迁移

如果你之前使用的是原生实现（sentence-transformers + 手动 FAISS），迁移到 LangChain 版本：

1. **更新依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **删除旧缓存**
   ```bash
   rm -rf data/processed/*.pkl
   rm -rf vector_store/faiss_index/*.index
   ```

3. **重新初始化**
   ```bash
   python main.py
   ```

LangChain 版本会自动重建索引，索引格式与原生版本不兼容。

## License

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
