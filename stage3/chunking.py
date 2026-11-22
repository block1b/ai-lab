"""环节 1：分块（对应打卡表「固定分块 vs 语义分块」实验）。

提供两种策略：
- fixed：按字符数固定切分，带 overlap（最简单，基线）
- semantic：按句子/段落边界 + 相似度聚合（尝试保住语义完整）
"""

import re

from langchain_core.documents import Document

_SENT_SPLIT = re.compile(r"(?<=[。！？；\n])")


def fixed_chunk(docs: list[Document], size: int = 120, overlap: int = 20) -> list[Document]:
    """固定字符数切块。size/overlap 以字符为单位（中文场景比 token 直观）。"""
    out = []
    for doc in docs:
        text = doc.page_content
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            out.append(Document(page_content=text[start:end], metadata=doc.metadata.copy()))
            if end == len(text):
                break
            start = end - overlap
    return out


def semantic_chunk(docs: list[Document], size: int = 160) -> list[Document]:
    """按句子边界聚合，尽量不切断句子；块超过 size 才切（启发式语义分块）。"""
    out = []
    for doc in docs:
        sents = [s.strip() for s in _SENT_SPLIT.split(doc.page_content) if s.strip()]
        buf = ""
        for sent in sents:
            if buf and len(buf) + len(sent) > size:
                out.append(Document(page_content=buf.strip(), metadata=doc.metadata.copy()))
                buf = ""
            buf += sent
        if buf.strip():
            out.append(Document(page_content=buf.strip(), metadata=doc.metadata.copy()))
    return out


def chunk(docs: list[Document], strategy: str = "fixed", **kwargs) -> list[Document]:
    if strategy == "fixed":
        return fixed_chunk(docs, **kwargs)
    if strategy == "semantic":
        return semantic_chunk(docs, **kwargs)
    raise ValueError(f"未知分块策略: {strategy}")
