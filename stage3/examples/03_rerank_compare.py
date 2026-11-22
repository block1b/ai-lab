"""实验3：重排对比（打卡表「加/不加 rerank」）。

运行：python examples/03_rerank_compare.py
观察：向量/混合检索的 Top-5，经 LLM 重排后 Top-3 是否更精准。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunking import chunk  # noqa: E402
from config import build_embeddings, build_llm  # noqa: E402
from corpus import load_documents  # noqa: E402
from retrieval import BM25Retriever, hybrid_retrieve  # noqa: E402
from rerank import llm_rerank  # noqa: E402
from vectorstore import build_store  # noqa: E402

llm = build_llm()
chunks = chunk(load_documents(), strategy="semantic")
store = build_store(chunks, build_embeddings())
bm25 = BM25Retriever(chunks)

Q = "手环的防水等级是多少，能游泳吗"

before = hybrid_retrieve(store, bm25, Q, k=5)
print("重排前 Top-5:")
for i, (d, s) in enumerate(before, 1):
    print(f"  [{i}] {d.metadata['source']} (RRF {s:.4f})")

after = llm_rerank(llm, Q, before, top_k=3)
print("\n重排后 Top-3:")
for i, (d, s) in enumerate(after, 1):
    print(f"  [{i}] {d.metadata['source']} (score {s})")

print("\n观察点：重排能把真正相关的块顶到前面，代价是多 5 次 LLM 调用。")
