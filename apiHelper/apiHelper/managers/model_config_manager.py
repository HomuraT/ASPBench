# managers/model_config_manager.py

from typing import List, Dict, Any
from .db import model_configs, save_db


def create_model_config(
    model_custom_name: str,
    api_custom_name: str,
    model_name: str,
    params: Dict[str, Any]
) -> bool:
    """
    创建新的模型配置，如已存在同名 model_custom_name 则返回False。
    """
    if model_custom_name in model_configs:
        return False
    model_configs[model_custom_name] = {
        "api_custom_name": api_custom_name,
        "model_name": model_name,
        "params": params
    }
    save_db()
    return True


def read_model_config(model_custom_name: str) -> Dict[str, Any]:
    """
    根据模型自定义名字读取对应的配置。如果不存在，返回空字典。
    """
    return model_configs.get(model_custom_name, {})


def update_model_config(
    model_custom_name: str,
    api_custom_name: str = None,
    model_name: str = None,
    params: Dict[str, Any] = None
) -> bool:
    """
    更新模型配置。如果配置不存在，返回False。
    """
    if model_custom_name not in model_configs:
        return False
    if api_custom_name is not None:
        model_configs[model_custom_name]["api_custom_name"] = api_custom_name
    if model_name is not None:
        model_configs[model_custom_name]["model_name"] = model_name
    if params is not None:
        model_configs[model_custom_name]["params"] = params
    save_db()
    return True


def delete_model_config(model_custom_name: str) -> bool:
    """
    删除对应的模型配置。如果不存在，返回False。
    """
    if model_custom_name not in model_configs:
        return False
    del model_configs[model_custom_name]
    save_db()
    return True


def list_model_configs() -> List[Dict[str, Any]]:
    """
    列出所有已配置的模型信息。
    """
    result = []
    for name, data in model_configs.items():
        result.append({
            "model_custom_name": name,
            "api_custom_name": data["api_custom_name"],
            "model_name": data["model_name"],
            "params": data["params"],
        })
    return result