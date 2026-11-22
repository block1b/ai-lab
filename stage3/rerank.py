"""环节 4：重排（对应打卡表「rerank 精排 Top-K」）。

这里用 LLM-as-judge 做重排：让模型给每个候选块与问题的相关性打 0-10 分。
生产环境可换专用 rerank 模型（如 bge-reranker），接口保持一致即可。
"""

import json
import re

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

_RERANK_SYSTEM = """你是检索相关性评分员。判断候选片段对回答用户问题有多大帮助。
只输出一个 JSON：{"score": 整数0-10}
评分标准：10=直接给出答案；5=部分相关；0=完全无关。不要输出其它文字。"""


def llm_rerank(llm: BaseChatModel, query: str,
               candidates: list[tuple[Document, float]], top_k: int = 3):
    """对候选块逐个打分并重排，返回 top_k 个 (Document, score)。"""
    scored = []
    for doc, _ in candidates:
        prompt = (
            f"问题：{query}\n\n候选片段：\n{doc.page_content}\n\n"
            "请只输出 {\"score\": 数字}。"
        )
        try:
            resp = llm.invoke(
                [SystemMessage(content=_RERANK_SYSTEM), HumanMessage(content=prompt)]
            )
            m = re.search(r"\{\s*\"score\"\s*:\s*(\d+)", resp.content)
            score = int(m.group(1)) if m else 0
        except Exception:
            score = 0
        scored.append((doc, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
