import os
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


def get_llm(temperature=None):
    model_type = os.getenv("MODEL_TYPE", "qwen").strip()
    model_name = os.getenv("MODEL_NAME", "qwen-plus")
    temperature_str = os.getenv("MODEL_TEMPERATURE", "0.75")
    model_api_key = os.getenv("MODEL_API_KEY")
    model_base_url = os.getenv("MODEL_BASE_URL")

    # 校验必要参数
    if not model_type:
        raise ValueError("Environment variable MODEL_TYPE is not set or is empty.")

    # if not model_api_key:
    #     raise ValueError("Environment variable MODEL_API_KEY is required.")

    try:
        if temperature is not None:
            temperature = float(temperature)
        else:
            temperature = float(temperature_str)
    except ValueError:
        temperature = 0  # 或者设置默认值

    model_map = {
        "openai": lambda: ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=model_base_url,
            api_key=model_api_key,
        ),
        "qwen": lambda: ChatTongyi(
            model=model_name,
            api_key=model_api_key,
            streaming=True,
            model_kwargs={"temperature": temperature},
        ),
        "ollama": lambda: ChatOllama(model=model_name, temperature=temperature, base_url=model_base_url),
    }

    if model_type in model_map:
        return model_map[model_type]()
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {model_type}. Supported types: {', '.join(model_map.keys())}")
