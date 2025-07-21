# managers/api_config_manager.py
from typing import List, Dict, Any
from .db import api_configs, load_db, save_db

def create_api_config(api_custom_name: str, url: str, api_key_value: str) -> bool:
    if api_custom_name in api_configs:
        return False
    api_configs[api_custom_name] = {"base_url": url, "api_key": api_key_value}
    save_db()
    return True

def read_api_config(api_custom_name: str) -> Dict[str, Any]:
    return api_configs.get(api_custom_name, {})

def update_api_config(api_custom_name: str, url: str = None, api_key_value: str = None) -> bool:
    if api_custom_name not in api_configs:
        return False
    if url is not None:
        api_configs[api_custom_name]["base_url"] = url
    if api_key_value is not None:
        api_configs[api_custom_name]["api_key"] = api_key_value
    save_db()
    return True

def delete_api_config(api_custom_name: str) -> bool:
    if api_custom_name not in api_configs:
        return False
    del api_configs[api_custom_name]
    save_db()
    return True

def list_api_configs() -> List[Dict[str, Any]]:
    return [
        {
            "api_custom_name": name,
            "base_url": data["base_url"],
            "api_key": data["api_key"],
        }
        for name, data in api_configs.items()
    ]