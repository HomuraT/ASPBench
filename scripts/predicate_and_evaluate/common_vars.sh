# Default JSON Parsing Model Name (added)

DEFAULT_JSON_PARSING_MODEL_NAME="local_qwen2_5_7b"



api_names=(
    "local_qwen2_5_7b"
    "local_qwen3_8b"
    "local_qwen3_14b"
#    "ppinfra_deepseek_v3"
#    "mmm_claude_3_5_haiku"
#    "ppinfra_deepseek_r1"
#    "mmmfz_o3_mini"
#    "mmmfz_o4_mini"
#    "mmm_gpt_4o_mini"
#    "mmm_glm_4_flash"
)


# 定义 API 对应的线程数

declare -A api_thread_counts=(
    ["local_qwen3_8b"]=4
    ["local_qwen3_14b"]=4
    ["local_qwen2_5_7b"]=8
)

# 默认线程数

default_threads=16