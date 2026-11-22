"""实验2：检索策略对比（打卡表「纯向量 vs 混合检索」）。

运行：python examples/02_retrieval_compare.py
关键：专有名词/型号/数字，看 BM25 与混合是否比纯向量召回更好。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunking import chunk  # noqa: E402
from config import build_embeddings  # noqa: E402
from corpus import load_documents  # noqa: E402
from retrieval import BM25Retriever, hybrid_retrieve, vector_retrieve  # noqa: E402
from vectorstore import build_store  # noqa: E402

chunks = chunk(load_documents(), strategy="semantic")
store = build_store(chunks, build_embeddings())
bm25 = BM25Retriever(chunks)

QUERIES = ["5ATM 防水等级", "蓝牙 5.3", "400-800-1234 客服电话", "续航 14 天"]

for q in QUERIES:
    print(f"\n=== 查询: {q} ===")
    for name, fn in [
        ("向量", lambda: vector_retrieve(store, q, k=3)),
        ("BM25", lambda: bm25.search(q, k=3)),
        ("混合", lambda: hybrid_retrieve(store, bm25, q, k=3)),
    ]:
        hits = [d.metadata["source"] for d, _ in fn()]
        print(f"  {name:<5}: {hits}")

print("\n观察点：专有名词/数字在纯向量下常召回不准，混合检索通常更稳。")
