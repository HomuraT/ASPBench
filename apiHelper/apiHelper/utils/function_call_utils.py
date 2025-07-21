import inspect
import re
from typing import Callable, Dict, Any


def parse_function_to_schema(func: Callable[..., Any]) -> Dict[str, Any]:
    """
    将指定的 Python 函数转换为可用于调用的 JSON Schema 格式，包括函数名称、描述以及参数的类型与说明。

    :param func: 目标 Python 函数，通常含有自身的 docstring，可提供函数功能说明与参数描述
    :type func: function
    :return: JSON Schema 格式的描述信息，包括 name、description、parameters 等字段
    :rtype: dict
    """
    spec = inspect.getfullargspec(func)
    doc = func.__doc__ or ""
    doc_lines = [line.strip() for line in doc.split("\n") if line.strip()]

    # 函数整体描述(首行)
    description = doc_lines[0] if doc_lines else ""

    # 解析 :param
    param_desc_pattern = re.compile(r":param\s+(\w+)\s*:\s*(.*)")
    param_desc_map = {}
    for line in doc_lines:
        match = param_desc_pattern.search(line)
        if match:
            var_name, var_desc = match.groups()
            param_desc_map[var_name] = var_desc.strip()

    schema = {
        "name": func.__name__,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }

    # 处理有默认值的参数
    defaults = spec.defaults or ()
    num_required = len(spec.args) - len(defaults)

    for i, arg in enumerate(spec.args):
        # 根据注解决定 type
        annotation = spec.annotations.get(arg, None)
        if annotation == int:
            param_type = "integer"
        elif annotation == float:
            param_type = "number"
        elif annotation == bool:
            param_type = "boolean"
        else:
            param_type = "string"

        # property 信息
        schema["parameters"]["properties"][arg] = {
            "type": param_type,
            "description": param_desc_map.get(arg, "")
        }

        # 无默认值才放进 required
        if i < num_required:
            schema["parameters"]["required"].append(arg)

    return schema
