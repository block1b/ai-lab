"""公共配置：复用 stage1/.env，构建 LLM 与 Embeddings。

Key 与模型全部走环境变量，代码零硬编码。
若 401 报 Incorrect API key，请到百炼控制台检查 Key 状态/权限，更新 stage1/.env。
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# 复用 stage1/.env
for candidate in (BASE_DIR / ".env", BASE_DIR.parent / "stage1" / ".env"):
    if candidate.exists():
        load_dotenv(candidate, override=False)

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
CHAT_MODEL = os.getenv("QWEN_MODEL", "qwen-flash")
EMBED_MODEL = os.getenv("QWEN_EMBED_MODEL", "text-embedding-v3")


def _check_key():
    if not API_KEY or API_KEY.startswith("sk-这里"):
        raise RuntimeError("请在 stage1/.env 中配置有效的 DASHSCOPE_API_KEY")


def build_llm(temperature: float = 0.0) -> ChatOpenAI:
    _check_key()
    return ChatOpenAI(
        api_key=API_KEY, base_url=BASE_URL, model=CHAT_MODEL,
        temperature=temperature, max_retries=2,
    )


class BatchingEmbeddings(OpenAIEmbeddings):
    """百炼兼容端点限制单次批量 <=10 条。"""

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


class TokenCounter(BaseCallbackHandler):
    """从 on_llm_end 截获 token 用量（兼容 usage_metadata / token_usage）。"""

    def __init__(self):
        self.prompt = 0
        self.completion = 0
        self.calls = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        self.calls += 1
        for batch in response.generations:
            for gen in batch:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                um = getattr(msg, "usage_metadata", None) or {}
                tu = (getattr(msg, "response_metadata", None) or {}).get("token_usage") or {}
                if um:
                    self.prompt += int(um.get("input_tokens", 0))
                    self.completion += int(um.get("output_tokens", 0))
                elif tu:
                    self.prompt += int(tu.get("prompt_tokens", 0))
                    self.completion += int(tu.get("completion_tokens", 0))


class StepLogger:
    """Agent 每步日志：带耗时与步号，是调试 Agent 的唯一抓手。"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.steps: list[dict] = []

    def log(self, step: int, action: str, detail: str = "", started: float | None = None):
        elapsed = round(time.time() - started, 3) if started is not None else None
        entry = {"step": step, "action": action, "detail": detail, "elapsed": elapsed}
        self.steps.append(entry)
        if self.verbose:
            head = f"[step {step:02d}] {action}"
            if elapsed is not None:
                head += f" ({elapsed}s)"
            print(head + (f" | {detail}" if detail else ""))

    def summary(self) -> str:
        return f"共 {len(self.steps)} 步，总耗时 {round(sum(s['elapsed'] or 0 for s in self.steps), 2)}s"
