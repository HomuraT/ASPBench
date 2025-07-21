import os
import http.client
import json
from typing import Optional, List, Dict, Any, Union, Type

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from anthropic import Anthropic

def call_anthropic_api(
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: str = "claude-2",
    temperature: float = 0.7,
    top_p: float = 1.0,
    functions: Optional[List[Dict[str, Any]]] = None,
    function_call: Union[str, Dict[str, str]] = "auto",
    response_format: Union[str, Type[BaseModel]] = "text",
    return_str: bool = True,
    return_full_completion: bool = False,
    max_tokens: int = 1024,
    client: Anthropic = None,
    **kwargs
) -> Any:
    """
    调用 Anthropic API 并返回生成结果，可选择返回纯文本或结构化输出。

    :param messages: 多轮对话的列表，每个元素是包含 'role' 和 'content' 的字典
    :type messages: List[Dict[str, str]]
    :param api_key: Anthropic API Key，若不提供则需自行在环境或配置中指定
    :type api_key: Optional[str]
    :param base_url: Anthropic API 的基础 URL，通常以 https:// 开头
    :type base_url: Optional[str]
    :param model_name: 模型名称，默认为 "claude-2"
    :type model_name: str
    :param temperature: 生成文本的“创意度”控制
    :type temperature: float
    :param top_p: nucleus sampling 参数，小于 1.0 可限制采样范围
    :type top_p: float
    :param functions: 用于描述可调用工具函数的 JSONSchema 信息
    :type functions: Optional[List[Dict[str, Any]]]
    :param function_call: 表示是否及如何调用上面函数的指示，可为 "auto" 或带配置的字典
    :type function_call: Union[str, Dict[str, str]]
    :param response_format: 返回结果的格式，可为 "text" 或继承自 BaseModel 的类
    :type response_format: Union[str, Type[BaseModel]]
    :param return_str: 是否以字符串形式返回结果，默认 True
    :type return_str: bool
    :param return_full_completion: 是否返回完整的 Completion 对象，默认 False
    :type return_full_completion: bool
    :param max_tokens: 本次调用允许生成的最大 tokens 数
    :type max_tokens: int
    :param client: Anthropic 客户端对象，若不提供则函数内部自动创建
    :type client: Anthropic
    :keyword kwargs: 其他扩展参数
    :return: 若指定结构化输出，则返回模型对应实例；否则返回字符串或完整对象
    :rtype: Any
    :raises AssertionError: 当 return_full_completion 和 return_str 同时为 True 时
    """

    assert not (return_full_completion and return_str), 'return_full_completion and return_str can not be True together'

    if client:
        _client = client
    else:
        # 如果还没有创建过 client，则创建并持有(或你也可每次都新建)
        if '/v1' in base_url:
            base_url = base_url.split('/v1')[0]
        _client = Anthropic(api_key=api_key, base_url=base_url)

    # 如果是结构化输出，则调用 parse(...)，否则调用 ChatCompletion.create
    # 注意 parse(...) 目前只支持部分参数，不一定包括 temperature、top_p、functions 等，
    # 官方接口还在 beta 中。若 parse(...) 不支持的参数，会被忽略或报错。
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        functions = [
            {
                "name": "build_text_analysis_result",
                "description": "build the text analysis object",
                "input_schema": response_format.model_json_schema()
            }
        ]

        last_content = messages[-1]['content']
        last_content = f'''
        {last_content}
        please build a json object according to the json schema.
        '''
        messages[-1]['content'] = last_content

        return call_anthropic_api(
            messages,
            api_key,
            base_url,
            model_name,
            temperature,
            top_p,
            functions,
            function_call,
            None,
            return_str,
            return_full_completion,
            max_tokens,
            client,
            **kwargs
        )
    else:
        # 普通文本输出 => ChatCompletion.create
        # 在这里可以传递 temperature, top_p, functions, function_call, max_tokens, etc.

        if functions is not None:
            last_content = messages[-1]['content']
            last_content = f'''
            In this environment you have access to a set of tools you can use to answer the user's question.
            {last_content}
            String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.
            Here are the functions available in JSONSchema format:
            {functions}
            You must build a json object according to the json schema according to the follows format:
              "function_call":{{
                 "name":"function name",
                 "arguments":{{
                     "attribute name":"value",
                     "attribute name":"value",
                     ...
                  }}
              }}
            '''
            messages[-1]['content'] = last_content

        completion = _client.messages.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            tools=functions,
            max_tokens=max_tokens
        )
        if return_full_completion:
            return completion
        else:
            if return_str:
                if functions is not None:
                    json_parser = JsonOutputParser()
                    json_obj = json_parser.invoke(completion.content[0].text)
                    return json.dumps(json_obj, ensure_ascii=False)
                else:
                    return completion.content[0].text
            else:
                return completion.content[0]