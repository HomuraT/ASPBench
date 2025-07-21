"""
使用 LangChain 调用 Google PaLM API (Vertex AI PaLM)，
暂不支持 openai-style function calling，但可自定义指令来实现特定功能。
"""

import os
from langchain.chat_models import ChatGooglePalm
from typing import Optional, Any

def call_google_palm_api(
    messages: list[dict[str, str]],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,  # Google PaLM 大多数情况下不支持自定义 base url
    model_name: str = "models/chat-bison-001",
    temperature: float = 0.7,
    top_p: float = 1.0,
    candidate_count: int = 1,
    response_format: str = "text",
    **kwargs
) -> Any:
    """
    调用LangChain对Google PaLM/Vertex AI PaLM模型的封装。
    :param message: 用户输入
    :param api_key: Google Cloud API Key, 或者使用ADC
    :param base_url: PaLM一般不支持自定义host, 保留作扩展
    :param model_name: "models/chat-bison-001" 等
    :param temperature: ...
    :param top_p: ...
    :param candidate_count: 返回候选回复数
    :param response_format: "json"/"text"
    :return: 根据format返回生成的内容
    """
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY", "")

    chat = ChatGooglePalm(
        google_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        candidate_count=candidate_count
    )
    response = chat(messages)

    if response_format == "json":
        return response.dict()
    else:
        return response.content