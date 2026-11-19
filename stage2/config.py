"""公共配置：加载 .env、构建 LLM、统计 token。

复用 stage1 的 .env（如果 stage2 自己没有 .env，会去上一级 stage1 找）。
这样你不用重复填 API Key。
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# 优先用 stage2/.env，没有则用 stage1/.env
for candidate in (BASE_DIR / ".env", BASE_DIR.parent / "stage1" / ".env"):
    if candidate.exists():
        load_dotenv(candidate, override=False)

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
BASE_URL = os.getenv(
    "QWEN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def build_llm(temperature: float = 0.0) -> ChatOpenAI:
    """构建聊天模型。抽取任务用 temperature=0 保证稳定。"""
    if not API_KEY or API_KEY.startswith("sk-这里"):
        raise RuntimeError(
            "未检测到有效的 DASHSCOPE_API_KEY。\n"
            "请在 stage1/.env 或 stage2/.env 中填入真实 API Key。"
        )
    return ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
        temperature=temperature,
    )


def usage_of(response) -> dict:
    meta = response.response_metadata or {}
    usage = meta.get("token_usage") or {}
    return {
        "prompt": usage.get("prompt_tokens", 0),
        "completion": usage.get("completion_tokens", 0),
        "total": usage.get("total_tokens", 0),
    }
