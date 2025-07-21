# Default JSON Parsing Model Name (added)

DEFAULT_JSON_PARSING_MODEL_NAME="mmmfz_gpt_4o_mini"



api_names=(
#    "local_qwen2_5_7b"
#    "local_qwen3_8b"
#    "ppinfra_deepseek_v3"
#    "mmm_gemini_25_flash_thinking"
    "mmmfz_o4_mini"
#    "ppinfra_deepseek_r1"
#    "mmmfz_o3_mini"
#    "mmmfz_gpt_4o"
#    "mmm_gpt_4o_mini"
#    "mmm_glm_4_flash"
)



# 定义 API 对应的线程数

declare -A api_thread_counts=(
    ["mmmfz_o4_mini"]=4
    ["ppinfra_deepseek_r1"]=4
    ["mmm_gemini_25_flash_nothinking"]=64
    ["mmm_gemini_25_flash_thinking"]=64
    ["local_qwen2_5_7b"]=8
    ["mmm_claude_3_5_haiku"]=16
)



# 默认线程数

default_threads=16