# apiHelper

一个llm的便捷调用工具

- 实现了openai和athropic的接口适配 `from src.apis.openai_api import call_openai_api, call_anthropic_api`
- 实现了openai，athropic的统一调用，使用 `from src.langchain.custom_llm import CustomLLM`

# 支持功能

## Function转换

支持直接把方法转成function call需要的格式

```python
from apiHelper.apis.openai_api import call_openai_api
from apiHelper.utils.function_call_utils import parse_function_to_schema

print(parse_function_to_schema(call_openai_api))
```

# 安装

```shell
bash install.sh
```

## 开发环境设置 (Development Environment Setup)

如果你想为此项目贡献代码或在本地运行示例，建议设置一个独立的 conda 环境：

```shell
# 1. 创建并激活 conda 环境
conda create -n apiHelperDev python=3.11 -y
conda activate apiHelperDev

# 2. 安装项目依赖
pip install -r requirement.txt
```

# Todo

- 实现多function调用（感觉这个不一定有用）
- 提示那边的功能可以简单实现一下
- 结构输出
- function call的iteral类型可以看看，非常有用
