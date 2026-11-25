"""阶段 3 端到端样例：看清一个问题如何流过 RAG 五环节。

输入：一个问题（命令行参数）
输出：分块 → 向量化 → 检索 → 重排 → 生成的每一步中间结果

两种模式：
    python examples/00_end_to_end.py "5ATM 防水等级是多少？"
        离线模式，零成本。用本地哈希向量和规则重排演示数据流，不调用 API。
    python examples/00_end_to_end.py "5ATM 防水等级是多少？" --live
        真实模式，调用千问 embedding + LLM，得到带引用的真实答案（消耗额度）。

先跑离线模式搞懂每一步在做什么，再加 --live 看真实效果。
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

from chunking import chunk  # noqa: E402
from corpus import load_documents  # noqa: E402
from rag import ANSWER_SYSTEM, answer, build_context, is_refusal  # noqa: E402
from rerank import llm_rerank  # noqa: E402
from retrieval import BM25Retriever, _bigrams, hybrid_retrieve, vector_retrieve  # noqa: E402
from vectorstore import build_store  # noqa: E402


class HashEmbedding(Embeddings):
    """纯本地字符哈希向量：只用于演示流程，没有任何语义能力。

    真实场景务必用 build_embeddings()（千问 text-embedding-v3），
    否则"语义相近"这一步无从谈起。
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed_query(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for ch in text.lower():
            vec[ord(ch) % self.dim] += 1.0
        if not any(vec):  # 避免出现零向量导致余弦相似度不可算
            vec[0] = 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


def rule_rerank(query: str, candidates, top_k: int = 3):
    """离线模式的简易重排：按查询词与候选的字面子串重合数打分。

    真实模式用 llm_rerank()（LLM-as-judge）或专用 rerank 模型。
    """
    qset = set(_bigrams(query))
    scored = [(doc, len(qset & set(_bigrams(doc.page_content)))) for doc, _ in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def show(title: str, content: str):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}\n{content}")


def main():
    args = sys.argv[1:]
    live = "--live" in args
    args = [a for a in args if a != "--live"]
    query = " ".join(args) if args else "5ATM 防水等级是多少？"

    print(f"模式：{'真实（调用千问，消耗额度）' if live else '离线（零成本，演示流程）'}")
    print(f"问题：{query}")

    # ---- 环节 1：分块 ----
    docs = load_documents()
    chunks = chunk(docs, strategy="semantic", size=160)
    show("① 分块：原始文档 → 可检索的块",
         f"输入：{len(docs)} 篇文档\n输出：{len(chunks)} 个块\n"
         f"示例块（来源 {chunks[0].metadata['source']}）：\n{chunks[0].page_content}")

    # ---- 环节 2：向量化 ----
    if live:
        from config import build_embeddings, build_llm
        emb = build_embeddings()
        llm = build_llm()
    else:
        emb = HashEmbedding()
        llm = None
    t0 = time.time()
    store = build_store(chunks, emb)
    bm25 = BM25Retriever(chunks)
    qvec = emb.embed_query(query)
    vec_dim = len(qvec)
    top_dims = sorted(enumerate(qvec), key=lambda x: x[1], reverse=True)[:5]
    vec_note = ("字符哈希向量仅演示流程，无语义能力；真实模式用千问 embedding"
                if not live else "千问 text-embedding-v3，语义相近的文本向量也相近")
    show("② 向量化：每个块 → 固定长度向量",
         f"向量维度：{vec_dim}\n建库耗时：{time.time() - t0:.2f}s\n"
         f"问题向量非零维数：{sum(1 for x in qvec if x > 0)}/{vec_dim}\n"
         f"最大 5 个分量（维度:值）：{[(i, round(v, 1)) for i, v in top_dims]}\n"
         f"说明：{vec_note}")

    # ---- 环节 3：检索（三路对比） ----
    k = 3
    vec_hits = vector_retrieve(store, query, k=k * 2)
    bm25_hits = bm25.search(query, k=k * 2)
    mix_hits = hybrid_retrieve(store, bm25, query, k=k * 2)

    def fmt(hits):
        return "\n".join(
            f"  [{i}] {d.metadata['source']}  (分数 {s:.4f})"
            for i, (d, s) in enumerate(hits, 1)
        )

    show("③ 检索：三种方式各取前 6 块",
         f"【纯向量】语义相近优先：\n{fmt(vec_hits)}\n\n"
         f"【BM25】字面命中优先：\n{fmt(bm25_hits)}\n\n"
         f"【混合 RRF】融合两路名次：\n{fmt(mix_hits)}")

    # ---- 环节 4：重排 ----
    candidates = mix_hits
    if live:
        finalists = llm_rerank(llm, query, candidates, top_k=k)
        rank_note = "LLM-as-judge 逐块打 0-10 分"
    else:
        finalists = rule_rerank(query, candidates, top_k=k)
        rank_note = "规则打分（字面子串重合数），仅演示机制"
    show("④ 重排：从 6 个候选精选出 3 个",
         f"方式：{rank_note}\n" +
         "\n".join(f"  [{i}] {d.metadata['source']}  (分数 {s})"
                   for i, (d, s) in enumerate(finalists, 1)))

    # ---- 环节 5：生成 ----
    final_docs = [d for d, _ in finalists]
    ctx = build_context(final_docs)
    if live:
        res = answer(llm, query, final_docs)
        show("⑤ 生成：带引用编号的答案",
             f"回答：{res['answer']}\n\n引用来源：{', '.join(res['sources'])}\n"
             f"token 用量：{res['tokens']}\n是否拒答：{is_refusal(res['answer'])}")
    else:
        show("⑤ 生成：拼装给模型的 Prompt（离线不调用）",
             f"System 指令：\n{ANSWER_SYSTEM}\n\n"
             f"拼好的参考资料（前 300 字）：\n{ctx[:300]}...\n\n"
             f"提示：加 --live 参数即可调用千问得到真实答案。")

    print(f"\n{'=' * 60}\n完成。数据链路：{len(docs)} 篇文档 → {len(chunks)} 块 → "
          f"6 个候选 → {k} 个精排块 → 答案\n{'=' * 60}")


if __name__ == "__main__":
    main()
