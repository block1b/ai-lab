"""环节 5：生成（带引用编号 + 无依据拒答）。

把检索到的块编号为 [1][2]... 拼进 prompt，强制模型：
- 每个事实后标注来源编号
- 依据不足时回答「文档中未提及」，禁止编造
"""

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from config import usage_of

ANSWER_SYSTEM = """你是基于给定资料回答问题的助手。规则：
1. 只依据【参考资料】回答，不要使用你的内部知识。
2. 每个关键事实后必须标注来源编号，格式如 [1]、[2]。
3. 若参考资料不足以回答，明确回答「文档中未提及」，不要猜测或编造。
4. 回答简洁，不超过 150 字。"""


def build_context(docs: list[Document]) -> str:
    lines = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "未知来源")
        lines.append(f"[{i}] （来源：{src}）{doc.page_content}")
    return "\n\n".join(lines)


def answer(llm: BaseChatModel, query: str, docs: list[Document]) -> dict:
    """生成带引用的答案。返回答案文本、引用来源、token 用量。"""
    if not docs:
        return {"answer": "文档中未提及", "sources": [], "tokens": 0}
    ctx = build_context(docs)
    resp = llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"【参考资料】\n{ctx}\n\n【问题】{query}\n\n请回答："),
    ])
    sources = list({d.metadata.get("source", "未知来源") for d in docs})
    return {
        "answer": resp.content.strip(),
        "sources": sources,
        "tokens": usage_of(resp)["total"],
    }


def is_refusal(text: str) -> bool:
    """判断是否为拒答（用于评估幻觉率）。"""
    markers = ["未提及", "没有提到", "文档中没有", "资料中未", "无法回答", "不知道"]
    return any(m in text for m in markers)
