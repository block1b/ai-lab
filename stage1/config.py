"""
公共配置模块：加载 .env，构建千问 LLM 实例，解析 token 用量。

设计要点：
- 用 langchain-openai 的 ChatOpenAI 走百炼的「OpenAI 兼容模式」，
  这样不依赖 langchain-community 里可能滞后的 ChatTongyi 封装，
  且以后换任何 OpenAI 兼容的模型只需改 .env。
- 密钥与模型名全部来自 .env，代码里不硬编码，方便版本管理。
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 无论从哪个目录运行脚本，都从本文件所在目录读取 .env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

import os  # noqa: E402  （load_dotenv 之后才读环境变量）

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
BASE_URL = os.getenv(
    "QWEN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def build_llm(temperature: float = 0.7, streaming: bool = False) -> ChatOpenAI:
    """构建一个 ChatOpenAI 实例。

    temperature: 0 最确定（适合抽取/问答），2 最随机（适合创意）；千问范围通常为 [0, 2)。
    streaming:   是否流式输出，配合回调可逐字打印。
    """
    if not API_KEY or API_KEY.startswith("sk-这里"):
        raise RuntimeError(
            "未检测到有效的 DASHSCOPE_API_KEY。\n"
            "请把 .env.example 复制为 .env，并填入你的真实 API Key。"
        )
    return ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
        temperature=temperature,
        streaming=streaming,
    )


def usage_of(response) -> dict:
    """从 LangChain 响应里提取本次调用的 token 用量。

    百炼 OpenAI 兼容模式会在响应中返回 usage 字段，
    LangChain 把它放进 response_metadata['token_usage']。
    """
    meta = response.response_metadata or {}
    usage = meta.get("token_usage") or {}
    return {
        "prompt": usage.get("prompt_tokens", 0),
        "completion": usage.get("completion_tokens", 0),
        "total": usage.get("total_tokens", 0),
    }
