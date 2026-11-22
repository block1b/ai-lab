"""实验1：分块策略对比（打卡表「固定512 vs 语义分块」）。

运行：python examples/01_chunking_compare.py
纯本地，不调用 API。观察不同参数下块数量与块长度分布。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunking import chunk  # noqa: E402
from corpus import load_documents  # noqa: E402

docs = load_documents()
for name, strategy, kw in [
    ("固定 120/overlap20", "fixed", {"size": 120, "overlap": 20}),
    ("固定 60/overlap15", "fixed", {"size": 60, "overlap": 15}),
    ("语义分块 80", "semantic", {"size": 80}),
]:
    chunks = chunk(docs, strategy=strategy, **kw)
    lens = [len(c.page_content) for c in chunks]
    print(f"\n{name}: {len(chunks)} 块, 平均 {sum(lens)/len(lens):.0f} 字, "
          f"最短 {min(lens)}, 最长 {max(lens)}")
    print("  首块:", chunks[0].page_content[:200].replace("\n", " "))

print("\n观察点：块太碎会丢上下文，块太大引入噪声。后续用 Recall@5 选出最优参数。")
