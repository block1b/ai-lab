"""核心：结构化抽取器 + 失败重试（含指数退避 + 错误反馈）。

提供两种实现，便于对比学习：
- StructuredExtractor：用 LangChain 的 with_structured_output（推荐，底层自动用 Pydantic 校验）
- ManualExtractor：手动 JSON 解析 + Pydantic 校验（透明，能看到每一步）

两者都实现同样的 run(text) -> (ReviewExtract, meta) 接口。
"""

import json
import random
import re
import time

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from config import usage_of
from prompts import build_system_prompt, build_user_prompt
from schemas import ReviewExtract, schema_json_example

SYSTEM_PROMPT = build_system_prompt(schema_json_example())


class ExtractionError(Exception):
    """重试耗尽后仍无法得到合法结果。"""

    def __init__(self, message: str, last_raw: str = "", attempts: int = 0):
        super().__init__(message)
        self.last_raw = last_raw
        self.attempts = attempts


class TokenCounter(BaseCallbackHandler):
    """ callback：在 LLM 调用结束时截获 token 用量。

    with_structured_output 直接返回 Pydantic 对象、拿不到 response，
    所以只能在 callback 里从底层 LLMResult 取用量。
    同时兼容两种字段：
      - LangChain 1.x 标准 usage_metadata (input_tokens/output_tokens)
      - OpenAI 兼容端点 response_metadata.token_usage (prompt_tokens/completion_tokens)
    """

    def __init__(self):
        self.prompt = 0
        self.completion = 0
        self.total = 0

    def on_llm_end(self, response, **kwargs) -> None:
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                # 优先 LangChain 1.x 标准字段
                um = getattr(msg, "usage_metadata", None) or {}
                p = um.get("input_tokens", 0)
                c = um.get("output_tokens", 0)
                t = um.get("total_tokens", 0)
                # 回退 OpenAI 风格字段
                if t == 0:
                    meta = getattr(msg, "response_metadata", None) or {}
                    usage = meta.get("token_usage") or {}
                    p = usage.get("prompt_tokens", 0)
                    c = usage.get("completion_tokens", 0)
                    t = usage.get("total_tokens", 0)
                self.prompt += p
                self.completion += c
                self.total += t


def _backoff(attempt: int, base: float = 0.5, cap: float = 8.0) -> float:
    """指数退避 + 随机抖动：0.5, 1, 2, 4 ... 封顶 8 秒。"""
    return min(cap, base * (2**attempt)) + random.uniform(0, 0.25)


def _clean_json(raw: str) -> str:
    """剥掉模型常见的 Markdown 代码块包裹和首尾杂字。"""
    raw = raw.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if match:
        raw = match.group(1).strip()
    # 截取第一个 { 到最后一个 }，去掉前后解释文字
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]
    return raw


class StructuredExtractor:
    """用 with_structured_output：LangChain 把 Schema 注入并在返回时自动 Pydantic 校验。"""

    def __init__(self, llm, max_retries: int = 3, with_feedback: bool = True,
                 method: str = "function_calling"):
        self.max_retries = max_retries
        self.with_feedback = with_feedback
        # 不同兼容端点支持的结构化输出方式不同：
        # 默认 function_calling（千问兼容模式普遍支持），不支持则试 json_schema，再退回默认。
        self.structured = None
        self._init_error = None
        for candidate in (method, "json_schema", None):
            try:
                self.structured = llm.with_structured_output(
                    ReviewExtract, method=candidate
                )
                self.method_used = candidate or "auto"
                break
            except (ValueError, NotImplementedError, TypeError) as exc:
                self._init_error = exc
                continue
        if self.structured is None:
            raise ExtractionError(
                f"with_structured_output 初始化失败，请改用 --mode manual：{self._init_error}",
                attempts=0,
            )

    def run(self, text: str):
        last_error = None
        total_tokens = 0
        for attempt in range(self.max_retries + 1):
            # 必须带 SystemMessage：with_structured_output 只把 Schema 结构注入工具定义，
            # 但角色、few-shot、"evidence 逐字引用/禁止编造"等约束全在 SYSTEM_PROMPT 里，
            # 不加则抽取质量下降（丢掉阶段2的 prompt 工程成果）。
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=build_user_prompt(text)),
            ]
            # 重试时把失败原因反馈给模型，帮它修正（对应"错误反馈"）
            if attempt > 0 and self.with_feedback and last_error is not None:
                messages.append(
                    HumanMessage(
                        content=(
                            f"上一次输出未通过校验：{last_error}。"
                            "请严格按 Schema 重新输出，只给 JSON，不要额外文字。"
                        )
                    )
                )
            counter = TokenCounter()
            try:
                result = self.structured.invoke(messages, config={"callbacks": [counter]})
                if not isinstance(result, ReviewExtract):
                    result = ReviewExtract.model_validate(result)
                return result, {
                    "attempts": attempt + 1,
                    "tokens": total_tokens + counter.total,
                    "prompt_tokens": counter.prompt,
                    "completion_tokens": counter.completion,
                }
            except (ValidationError, ValueError, TypeError) as exc:
                last_error = exc
            finally:
                # 无论成功或失败，模型这次调用都已真实产生并计费，全部累加
                total_tokens += counter.total
            if attempt < self.max_retries:
                time.sleep(_backoff(attempt))
        raise ExtractionError(
            f"结构化抽取重试 {self.max_retries} 次后仍失败：{last_error}",
            attempts=self.max_retries + 1,
        )


class ManualExtractor:
    """手动模式：完全透明——让模型吐 JSON 字符串，自己解析 + Pydantic 校验。"""

    def __init__(self, llm, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries

    def run(self, text: str):
        last_raw, last_error = "", None
        total_tokens = 0
        for attempt in range(self.max_retries + 1):
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=build_user_prompt(text)),
            ]
            if attempt > 0 and last_error is not None:
                messages.append(
                    HumanMessage(
                        content=(
                            f"上一次输出校验失败：{last_error}。\n"
                            f"你的输出是：{last_raw[:300]}\n"
                            "请修正后只输出合法 JSON。"
                        )
                    )
                )
            response = self.llm.invoke(messages)
            total_tokens += usage_of(response)["total"]
            raw = response.content if isinstance(response.content, str) else str(response.content)
            last_raw = raw
            try:
                data = json.loads(_clean_json(raw))
                result = ReviewExtract.model_validate(data)
                return result, {"attempts": attempt + 1, "tokens": total_tokens}
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(_backoff(attempt))
        raise ExtractionError(
            f"手动抽取重试 {self.max_retries} 次后仍失败：{last_error}",
            last_raw=last_raw,
            attempts=self.max_retries + 1,
        )


def build_extractor(llm, mode: str = "structured", max_retries: int = 3):
    if mode == "structured":
        return StructuredExtractor(llm, max_retries=max_retries)
    if mode == "manual":
        return ManualExtractor(llm, max_retries=max_retries)
    raise ValueError(f"未知模式: {mode}（可选 structured / manual）")
