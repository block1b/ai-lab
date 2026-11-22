"""阶段 3 命令行问答：带引用 + 无依据拒答。

用法：
    python cli.py "手环的防水等级是多少？"
    python cli.py --mode hybrid --rerank "续航多久"
    python cli.py                          # 交互模式
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import build_embeddings, build_llm
from corpus import load_documents
from chunking import chunk
from vectorstore import build_store
from retrieval import BM25Retriever, vector_retrieve, hybrid_retrieve
from rerank import llm_rerank
from rag import answer


def main():
    parser = argparse.ArgumentParser(description="阶段3：RAG 问答")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--mode", default="hybrid", choices=["vector", "bm25", "hybrid"])
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    llm = build_llm()
    store = build_store(chunk(load_documents(), strategy="semantic"), build_embeddings())
    bm25 = BM25Retriever(chunk(load_documents(), strategy="semantic"))

    def run(q: str):
        if args.mode == "vector":
            cands = vector_retrieve(store, q, k=args.topk * 2)
        elif args.mode == "bm25":
            cands = bm25.search(q, k=args.topk * 2)
        else:
            cands = hybrid_retrieve(store, bm25, q, k=args.topk * 2)
        if args.rerank:
            cands = llm_rerank(llm, q, cands, top_k=args.topk)
        else:
            cands = cands[:args.topk]
        docs = [d for d, _ in cands]
        res = answer(llm, q, docs)
        print(f"\n问题：{q}")
        print(f"回答：{res['answer']}")
        print(f"引用：{', '.join(res['sources'])}  (token {res['tokens']})")
        for i, d in enumerate(docs, 1):
            print(f"  [{i}] {d.metadata['source']}: {d.page_content[:50]}...")

    if args.question:
        run(" ".join(args.question))
        return
    print("交互模式，输入 exit 退出。")
    while True:
        try:
            q = input("\n问> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit"):
            break
        if q:
            run(q)


if __name__ == "__main__":
    main()
