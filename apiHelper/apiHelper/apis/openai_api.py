"""
使用 LangChain 调用 OpenAI API，支持:
- base_url (openai_api_base)
- function calling
- 自定义参数 (model_name, temperature, top_p, ... )
- 返回自定义格式
"""
import json
import os
from langchain_openai import ChatOpenAI
from typing import Optional, List, Dict, Any, Type, Union

from pydantic import BaseModel
from openai import OpenAI, NOT_GIVEN


def call_openai_api(
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,           # 如果指定，则覆盖默认的 openai_api_base
    model_name: str = None,
    temperature: float = 0.5,
    top_p: float = 1.0,
    tools: Optional[List[Dict[str, Any]]] = None,   # Official function calling
    tool_choice: Union[str, Dict[str, str]] = "auto", # "none" | "auto" | {"name": "..."}
    response_format: Union[str, Type[BaseModel]] = "text", # "text" 或 Pydantic模型
    client: OpenAI = None,
    return_str: bool = True,
    return_full_completion: bool = False,
    **kwargs
) -> Any:
    """
    使用 Python 官方 openai 库封装大模型调用，支持：
      1) 普通对话 + function calling
      2) 结构化输出(Pydantic) => 调用 parse(...)
      3) 自定义 base_url、model_name、temperature、top_p 等

    :param messages: 对话消息列表, e.g. [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    :param api_key: OpenAI API Key，不传则尝试从环境变量 OPENAI_API_KEY 获取
    :param base_url: 替换默认的 openai_api_base，比如 "https://api.openai.com/v1"
    :param model_name: 模型名称, e.g. "gpt-4" or "gpt-3.5-turbo"
    :param temperature: 生成文本的发散度
    :param top_p: nucleus sampling
    :param tools: Official function calling 的 function 列表
    :param tool_choice: Union[str, Dict[str, str]] = "auto", # "none" | "auto" | {"name": "..."}
    :param response_format: "text" 表示返回文本字符串, 如果传入一个 Pydantic模型类则进行结构化解析
    :param client: OpenAI = None,
    :param return_str: bool = True,
    :param return_full_completion: bool = False,
    :param kwargs: 传给 API 的更多参数 (如 `max_tokens`, `n` 等)
    :return: 返回结果根据参数组合而不同:
             1. **如果 `response_format` 是 Pydantic 模型类 AND (`return_full_completion` 为 `True` OR `return_str` 为 `False`):**
                - 函数调用 OpenAI `parse()` 方法。
                - 返回值是解析后的 Pydantic 模型实例。
             2. **否则 (即 `response_format` 不是 Pydantic，或虽是 Pydantic 但 `return_str` 为 `True` 且 `return_full_completion` 为 `False`):**
                - 函数调用 OpenAI `create()` 方法，得到 `ChatCompletion` 对象。
                - **如果 `return_full_completion` 为 `True`:**
                  - 返回原始的 `ChatCompletion` 对象。
                - **如果 `return_full_completion` 为 `False`:**
                  - **若 `return_str` 为 `True`:**
                    - 如果 `ChatCompletion` 对象含多个 `choices` (例如 `n > 1`): 返回 `List[str]` (消息内容列表)。
                    - 否则 (单个 `choice`): 返回 `str` (消息内容，或工具调用的JSON/Llama3格式)。
                  - **若 `return_str` 为 `False`:**
                    - 返回第一个 `choice` 的 `ChatCompletionMessage` 对象 (或 `None`)。
    """

    assert not (return_full_completion and return_str), 'return_full_completion and return_str can not be True together'

    if 'api_custom_name' in kwargs: 
        del kwargs['api_custom_name']
    if 'params' in kwargs:
        del kwargs['params']

    if 'temperature' in kwargs:
        temperature = kwargs['temperature']
        del kwargs['temperature']
    if 'top_p' in kwargs:
        top_p = kwargs['top_p']
        del kwargs['top_p']
        

    if client:
        _client = client
    else:
        _client = OpenAI(api_key=api_key, base_url=base_url)

    # Determine if we should use the parse() method
    # We use parse() if response_format is Pydantic and the user wants the Pydantic object itself
    # (either as the full_completion result or as the specific non-string result when not full_completion).
    should_use_parse = (
        isinstance(response_format, type) and
        issubclass(response_format, BaseModel)
    )

    if should_use_parse:
        # Path 1: Use OpenAI's parse() method
        completion_obj = _client.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            response_format=response_format,
            temperature=temperature if temperature is not None else NOT_GIVEN,
            top_p=top_p if top_p is not None else NOT_GIVEN,
            tools=tools if tools else NOT_GIVEN,
            tool_choice=tool_choice if tools else NOT_GIVEN,
            **kwargs
        )
    else:
        completion_obj = _client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            tools=tools if tools else NOT_GIVEN,
            tool_choice=tool_choice if tools else NOT_GIVEN,
            **kwargs 
        )

    if return_full_completion:
        # This implies create() was called and user wants the full ChatCompletion object.
        return completion_obj

    # From here, return_full_completion is False, and we are processing the ChatCompletion object.
    if return_str:
        if completion_obj.choices and len(completion_obj.choices) > 1:
            contents = []
            for choice in completion_obj.choices:
                if choice.message and choice.message.content is not None:
                    contents.append(choice.message.content)
            return contents
        else:
            first_choice_message = completion_obj.choices[0].message if completion_obj.choices else None

            if tools is not None and first_choice_message:
                tool_call_content = first_choice_message.content # Keep a reference
                if tool_call_content and '<|python_tag|>' in tool_call_content:
                    content_after_tag = tool_call_content.split('<|python_tag|>', 1)[1]
                    if '<|eom_id|>' in content_after_tag:
                        content_after_tag = content_after_tag.split('<|eom_id|>', 1)[0]
                    try:
                        tool_call_obj = eval(content_after_tag)
                        if isinstance(tool_call_obj, dict) and 'parameters' in tool_call_obj:
                            if 'messages' in tool_call_obj['parameters'] and \
                               isinstance(tool_call_obj['parameters']['messages'], str):
                                tool_call_obj['parameters']['messages'] = eval(tool_call_obj['parameters']['messages'])
                            tool_call_obj['arguments'] = tool_call_obj['parameters']
                            del tool_call_obj['parameters']
                            return json.dumps({'function_call': tool_call_obj})
                    except Exception:
                        pass # Fall through if eval or dict manipulation fails

            if first_choice_message:
                # If tools are active AND not handled by Llama3 format, return JSON of message
                # Note: tool_call_content might be undefined if tools is None or no first_choice_message
                # So check first_choice_message.content directly here for the Llama3 tag check.
                if tools is not None and not (first_choice_message.content and '<|python_tag|>' in first_choice_message.content):
                    return first_choice_message.to_json()
                else: # No tools, or Llama3 format handled (or failed & fell through)
                    return first_choice_message.content
            else:
                return None # No choices, return None for string
    else: # return_str is False (and not Pydantic/not_return_str path, and not full_completion path)
        # User wants the Message object from the first choice
        return completion_obj.choices[0].message if completion_obj.choices else None
