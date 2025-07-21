"""
使用 LangChain 调用 Azure OpenAI API，支持:
- base_url (openai_api_base)
- function calling
- 自定义参数 (model_name, temperature, top_p, ... )
- 返回自定义格式
"""

import os
from langchain.chat_models import ChatOpenAI
from typing import Optional, List, Dict, Any

def call_azure_openai_api(
    messages: list[dict[str, str]],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,   # 替换默认 Azure OpenAI endpoint
    model_name: str = "gpt-35-turbo",
    temperature: float = 0.7,
    top_p: float = 1.0,
    functions: Optional[List[Dict[str, Any]]] = None,
    function_call: str = "auto",
    response_format: str = "text",
    **kwargs
) -> Any:
    """
    Azure OpenAI 调用示例。Azure OpenAI 也可使用 function calling，具体看 Azure 版本支持情况。
    :param message: 用户输入文本
    :param api_key: 如果不传，就尝试从环境变量AZURE_OPENAI_KEY中获取
    :param base_url: Azure OpenAI Endpoint, 例如 https://YOUR-RESOURCE-NAME.openai.azure.com/
    :param model_name: 部署的 Azure OpenAI 模型名称
    :param temperature: ...
    :param top_p: ...
    :param functions: Official function calling
    :param function_call: function调用策略
    :param response_format: "json"/"text"
    :return: 生成的内容
    """
    if not api_key:
        api_key = os.getenv("AZURE_OPENAI_KEY", "")

    # Azure 还需指定 openai_api_type="azure"，以及openai_api_version等
    chat = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=base_url,   # 如果传 None 则使用默认
        openai_api_type="azure",
        openai_api_version="2023-05-15",  # 示例版本，可按需修改
        model_name=model_name,
        temperature=temperature,
        top_p=top_p
    )

    response = chat(messages, functions=functions, function_call=function_call)

    if response_format == "json":
        return response.dict()
    else:
        return response.content