"""
LLM API调用模块 - 使用 LangChain

功能：
- 使用 LangChain 的 ChatOpenAI
- 支持流式和非流式输出
- 集成 Prompt 模板
"""
import logging
from typing import List, Dict, Any, Optional, Generator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from config.config_manager import config
from config.prompt_templates import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE, CONTEXT_TEMPLATE

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM客户端封装 - LangChain 实现"""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        """
        初始化LLM客户端

        Args:
            model: 模型名称
            api_key: API密钥
            base_url: API基础URL
            temperature: 温度参数
        """
        llm_config = config.llm_config

        self.model = model or llm_config.get('model')
        self.api_key = api_key or llm_config.get('api_key')
        self.base_url = base_url or llm_config.get('base_url')
        self.temperature = temperature if temperature is not None else llm_config.get('temperature', 1.0)

        # 初始化 LangChain ChatOpenAI
        self.llm = ChatOpenAI(
            model=self.model,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            temperature=self.temperature,
            streaming=False
        )

        # 流式 LLM
        self.llm_stream = ChatOpenAI(
            model=self.model,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            temperature=self.temperature,
            streaming=True
        )

        logger.info(f"✅ LLM客户端初始化完成，模型: {self.model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        生成回复（非流式）

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            max_tokens: 最大token数

        Returns:
            生成的文本
        """
        try:
            messages = []

            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))

            messages.append(HumanMessage(content=prompt))

            logger.info(f"正在调用LLM生成回复...")

            # 使用 LangChain 调用
            response = self.llm.invoke(messages)
            answer = response.content

            logger.info(f"✅ LLM回复生成完成，长度: {len(answer)} 字符")
            return answer

        except Exception as e:
            logger.error(f"❌ LLM调用失败: {e}")
            raise

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> Generator[str, None, None]:
        """
        生成回复（流式）

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            max_tokens: 最大token数

        Yields:
            生成的文本片段
        """
        try:
            messages = []

            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))

            messages.append(HumanMessage(content=prompt))

            logger.info(f"正在调用LLM生成回复（流式）...")

            # 使用 LangChain 流式调用
            for chunk in self.llm_stream.stream(messages):
                if chunk.content:
                    yield chunk.content

        except Exception as e:
            logger.error(f"❌ LLM流式调用失败: {e}")
            raise

    def generate_with_context(
        self,
        question: str,
        context: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        基于检索到的上下文生成回复

        Args:
            question: 用户问题
            context: 检索到的相关文档列表
            system_prompt: 系统提示词
            max_tokens: 最大token数

        Returns:
            生成的回答
        """
        # 构建上下文文本
        context_text = self._format_context(context)

        # 构建提示词
        prompt = self._build_rag_prompt(question, context_text)

        # 使用默认系统提示词
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens
        )

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """
        格式化上下文文档

        Args:
            context: 检索到的文档列表

        Returns:
            格式化后的上下文文本
        """
        context_parts = []
        for i, doc in enumerate(context, 1):
            qa_pair = doc['qa_pair']
            similarity = doc['similarity']

            # 使用配置的模板
            context_part = CONTEXT_TEMPLATE.format(
                index=i,
                similarity=similarity,
                question=qa_pair['question'],
                answer=qa_pair['answer']
            )
            context_parts.append(context_part.strip())

        return "\n\n".join(context_parts)

    def _build_rag_prompt(self, question: str, context: str) -> str:
        """
        构建RAG提示词

        Args:
            question: 用户问题
            context: 上下文文本

        Returns:
            完整的提示词
        """
        # 使用配置的 RAG 模板
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
        return prompt
