"""环节 3：检索（对应打卡表「纯向量 vs 混合检索」实验）。

三种 retriever：
- vector：纯向量语义检索（擅长语义相近，怕专有名词）
- bm25：关键词检索（纯 Python 实现，擅长专有名词/数字精确匹配）
- hybrid：RRF（Reciprocal Rank Fusion）融合两者，兼顾语义与关键词

为什么中文 BM25 按 2-gram 切：中文没有空格分词，用 2-gram 能在不引入
jieba 等分词依赖的前提下，对「心率」「防水」这类词有基本的匹配能力。
"""

from collections import Counter

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore


def _bigrams(text: str) -> list[str]:
    """中文按 2-gram 切分，英文/数字按词切分。"""
    tokens = []
    buf = ""
    for ch in text.lower():
        if "\u4e00" <= ch <= "\u9fff":  # 中文字符
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append(ch)
        elif ch.isalnum():
            buf += ch
        else:
            if buf:
                tokens.append(buf)
                buf = ""
    if buf:
        tokens.append(buf)
    # 中文单字 + 相邻两字组合
    result = []
    for i, t in enumerate(tokens):
        result.append(t)
        if "\u4e00" <= t <= "\u9fff" and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if "\u4e00" <= nxt <= "\u9fff":
                result.append(t + nxt)
    return result


class BM25Retriever:
    """轻量 BM25 实现（无第三方依赖）。"""

    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tokens = [_bigrams(d.page_content) for d in docs]
        self.tfs = [Counter(t) for t in self.tokens]
        self.avgdl = sum(len(t) for t in self.tokens) / max(len(docs), 1)
        df = Counter()
        for t in self.tokens:
            df.update(set(t))
        import math
        n = len(docs)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def _score(self, tf: Counter, dl: int, term: str) -> float:
        f = tf.get(term, 0)
        if f == 0:
            return 0.0
        denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        return self.idf.get(term, 0) * f * (self.k1 + 1) / denom

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        qterms = _bigrams(query)
        scores = []
        for i, doc in enumerate(self.docs):
            s = sum(self._score(self.tfs[i], len(self.tokens[i]), t) for t in qterms)
            scores.append((doc, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


def vector_retrieve(store: InMemoryVectorStore, query: str, k: int = 5):
    """纯向量检索，返回 [(Document, score)]（这里 score 是相似度，越大越好）。"""
    return store.similarity_search_with_score(query, k=k)


def hybrid_retrieve(store: InMemoryVectorStore, bm25: BM25Retriever,
                    query: str, k: int = 5, rrf_k: int = 60):
    """RRF 融合向量与 BM25 的排序结果。

    RRF 公式：score(d) = Σ 1/(rrf_k + rank_i(d))
    不依赖两路分数的绝对量纲，只融合名次，工程上很稳。
    """
    vec = vector_retrieve(store, query, k=k * 2)
    kw = bm25.search(query, k=k * 2)

    # 用 page_content + source 作为文档唯一标识
    def key(doc):
        return (doc.page_content, doc.metadata.get("source", ""))

    scores, docs = {}, {}
    for rank, (doc, _) in enumerate(vec, start=1):
        kk = key(doc)
        scores[kk] = scores.get(kk, 0) + 1 / (rrf_k + rank)
        docs[kk] = doc
    for rank, (doc, _) in enumerate(kw, start=1):
        kk = key(doc)
        scores[kk] = scores.get(kk, 0) + 1 / (rrf_k + rank)
        docs.setdefault(kk, doc)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(docs[kk], s) for kk, s in ranked]
