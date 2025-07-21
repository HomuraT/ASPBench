# Default JSON Parsing Model Name (added)

DEFAULT_JSON_PARSING_MODEL_NAME="local_qwen2_5_7b"



api_names=(
   "local_qwen2_5_14b"
   "local_qwen2_5_7b"
)



# 定义 API 对应的线程数

declare -A api_thread_counts=(
    ["ppinfra_deepseek_v3"]=4
    ["ppinfra_deepseek_r1"]=4
    ["local_qwen3_8b"]=16
    ["local_qwen2_5_7b"]=16
)



# 默认线程数
default_threads=16
