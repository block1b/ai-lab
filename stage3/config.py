"""公共配置：复用 stage1 的 .env，构建 LLM 与 Embeddings。

 Embedding 走百炼兼容端点（text-embedding-v3/v2），与聊天模型同一套 Key。
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# 复用 stage1/.env（本目录没有 .env 时向上找）
for candidate in (BASE_DIR / ".env", BASE_DIR.parent / "stage1" / ".env"):
    if candidate.exists():
        load_dotenv(candidate, override=False)

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = os.getenv(
    "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
CHAT_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
EMBED_MODEL = os.getenv("QWEN_EMBED_MODEL", "text-embedding-v3")


def _check_key():
    if not API_KEY or API_KEY.startswith("sk-这里"):
        raise RuntimeError("请在 stage1/.env 中配置有效的 DASHSCOPE_API_KEY")


def build_llm(temperature: float = 0.0) -> ChatOpenAI:
    _check_key()
    return ChatOpenAI(
        api_key=API_KEY, base_url=BASE_URL, model=CHAT_MODEL,
        temperature=temperature,
    )


class BatchingEmbeddings(OpenAIEmbeddings):
    """百炼兼容端点限制单次批量 ≤10 条，这里自动切分批次。"""

    def embed_documents(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), batch_size):
            out.extend(super().embed_documents(texts[i:i + batch_size]))
        return out


def build_embeddings() -> OpenAIEmbeddings:
    _check_key()
    return BatchingEmbeddings(
        api_key=API_KEY, base_url=BASE_URL, model=EMBED_MODEL,
        check_embedding_ctx_length=False,
    )


def usage_of(response) -> dict:
    meta = response.response_metadata or {}
    usage = meta.get("token_usage") or {}
    return {
        "prompt": usage.get("prompt_tokens", 0),
        "completion": usage.get("completion_tokens", 0),
        "total": usage.get("total_tokens", 0),
    }
