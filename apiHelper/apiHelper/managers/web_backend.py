# managers/web_backend.py
import socket
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

# 引入管理函数
from apiHelper.managers.api_config_manager import (
    create_api_config,
    read_api_config,
    update_api_config,
    delete_api_config,
    list_api_configs
)
from apiHelper.managers.model_config_manager import (
    create_model_config,
    read_model_config,
    update_model_config,
    delete_model_config,
    list_model_configs
)


app = FastAPI(title="LLM API & Model Config Manager", version="0.1.0")

# ------------------------------
# API 配置相关的请求模型
# ------------------------------
class ApiConfigCreate(BaseModel):
    api_custom_name: str
    url: str
    api_key_value: str

class ApiConfigUpdate(BaseModel):
    url: Optional[str] = None
    api_key_value: Optional[str] = None


# ------------------------------
# Model 配置相关的请求模型
# ------------------------------
class ModelConfigCreate(BaseModel):
    model_custom_name: str
    api_custom_name: str
    model_name: str
    params: Dict[str, Any]

class ModelConfigUpdate(BaseModel):
    api_custom_name: Optional[str] = None
    model_name: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


# ------------------------------
# API 配置路由
# ------------------------------
@app.post("/api-configs", summary="创建新的 API 配置")
def api_config_create(payload: ApiConfigCreate):
    success = create_api_config(
        api_custom_name=payload.api_custom_name,
        url=payload.url,
        api_key_value=payload.api_key_value
    )
    if not success:
        raise HTTPException(status_code=400, detail="API custom name already exists.")
    return {"message": "API config created successfully."}


@app.get("/api-configs", summary="列出所有 API 配置")
def api_config_list():
    return list_api_configs()


@app.get("/api-configs/{api_custom_name}", summary="读取指定 API 配置")
def api_config_read(api_custom_name: str):
    config = read_api_config(api_custom_name)
    if not config:
        raise HTTPException(status_code=404, detail="API config not found.")
    return config


@app.put("/api-configs/{api_custom_name}", summary="更新指定 API 配置")
def api_config_update(api_custom_name: str, payload: ApiConfigUpdate):
    success = update_api_config(
        api_custom_name=api_custom_name,
        url=payload.url,
        api_key_value=payload.api_key_value
    )
    if not success:
        raise HTTPException(status_code=404, detail="API config not found.")
    return {"message": "API config updated successfully."}


@app.delete("/api-configs/{api_custom_name}", summary="删除指定 API 配置")
def api_config_delete(api_custom_name: str):
    success = delete_api_config(api_custom_name)
    if not success:
        raise HTTPException(status_code=404, detail="API config not found.")
    return {"message": "API config deleted successfully."}


# ------------------------------
# Model 配置路由
# ------------------------------
@app.post("/model-configs", summary="创建新的模型配置")
def model_config_create(payload: ModelConfigCreate):
    success = create_model_config(
        model_custom_name=payload.model_custom_name,
        api_custom_name=payload.api_custom_name,
        model_name=payload.model_name,
        params=payload.params
    )
    if not success:
        raise HTTPException(status_code=400, detail="Model custom name already exists.")
    return {"message": "Model config created successfully."}


@app.get("/model-configs", summary="列出所有模型配置")
def model_config_list():
    return list_model_configs()


@app.get("/model-configs/{model_custom_name}", summary="读取指定模型配置")
def model_config_read(model_custom_name: str):
    config = read_model_config(model_custom_name)
    if not config:
        raise HTTPException(status_code=404, detail="Model config not found.")
    return config


@app.put("/model-configs/{model_custom_name}", summary="更新指定模型配置")
def model_config_update(model_custom_name: str, payload: ModelConfigUpdate):
    success = update_model_config(
        model_custom_name=model_custom_name,
        api_custom_name=payload.api_custom_name,
        model_name=payload.model_name,
        params=payload.params
    )
    if not success:
        raise HTTPException(status_code=404, detail="Model config not found.")
    return {"message": "Model config updated successfully."}


@app.delete("/model-configs/{model_custom_name}", summary="删除指定模型配置")
def model_config_delete(model_custom_name: str):
    success = delete_model_config(model_custom_name)
    if not success:
        raise HTTPException(status_code=404, detail="Model config not found.")
    return {"message": "Model config deleted successfully."}

def run():
    """
    获取本机默认路由对应IP并启动 Uvicorn。
    """
    # 1) 自动获取本机IP
    #    通过连接到公共DNS (8.8.8.8) 来确定当前默认出口 IP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        local_ip = sock.getsockname()[0]
    except Exception:
        # 如果自动检测失败，可以使用 127.0.0.1 或其他兜底IP
        local_ip = "127.0.0.1"
    finally:
        sock.close()

    # 2) 打印提示信息，方便复制粘贴到浏览器
    port = 8000
    print(f"Server is running, you can visit:")
    print(f"  • http://{local_ip}:{port}/docs")
    print("Press Ctrl+C to quit.")

    # 3) 监听 0.0.0.0 以对外暴露服务
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    run()