import json
import os
from typing import Dict, Any

RESOURCES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'resources')
API_CONFIGS_PATH = os.path.join(RESOURCES_DIR, 'api_configs.json')
MODELS_CONFIGS_PATH = os.path.join(RESOURCES_DIR, 'models_configs.json')

api_configs: Dict[str, Dict[str, Any]] = {}
model_configs: Dict[str, Dict[str, Any]] = {}

def load_db() -> None:
    if not os.path.exists(RESOURCES_DIR):
        os.makedirs(RESOURCES_DIR)
    global api_configs, model_configs
    if os.path.isfile(API_CONFIGS_PATH):
        with open(API_CONFIGS_PATH, 'r', encoding='utf-8') as f:
            api_configs = json.load(f)
    if os.path.isfile(MODELS_CONFIGS_PATH):
        with open(MODELS_CONFIGS_PATH, 'r', encoding='utf-8') as f:
            model_configs = json.load(f)

def save_db() -> None:
    if not os.path.exists(RESOURCES_DIR):
        os.makedirs(RESOURCES_DIR)
    with open(API_CONFIGS_PATH, 'w', encoding='utf-8') as f:
        json.dump(api_configs, f, ensure_ascii=False, indent=2)
        f.flush()
    with open(MODELS_CONFIGS_PATH, 'w', encoding='utf-8') as f:
        json.dump(model_configs, f, ensure_ascii=False, indent=2)
        f.flush()

def get_model_and_api_info(model_name: str):
    global model_configs, api_configs
    model_info = model_configs.get(model_name)
    if not model_info:
        return None  # or raise an exception

    api_name = model_info.get('api_custom_name')
    api_info = api_configs.get(api_name, {})

    return {
        **model_info,
        **api_info,
        **model_info['params']
    }

# load once on import
load_db()