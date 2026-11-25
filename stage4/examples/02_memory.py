"""示例2：短期 + 长期记忆（离线，无 API）。

短期记忆：窗口裁剪 + 可选摘要压缩
长期记忆：向量库存取事实，按语义召回
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from memory import LongTermMemory, ShortTermMemory  # noqa: E402

mem = ShortTermMemory(max_turns=2, summarizer=lambda text: f"摘要({len(text)}字)")
for i in range(5):
    mem.add(HumanMessage(content=f"问{i}"))
    mem.add(AIMessage(content=f"答{i}"))
dropped = mem.trim()
print(f"短期记忆：裁剪 {dropped} 轮")
print("送入模型的消息:", [m.content for m in mem.get()])

lt = LongTermMemory(DeterministicFakeEmbedding(size=32))
lt.remember("星云智能2024年营收12.8亿元，同比增长28%")
lt.remember("海外收入占比35%，计划2025年提升至45%")
lt.remember("风险：前五大客户收入占比45%，集中度较高")
print("\n长期记忆召回（问「营收」）:", lt.recall("营收", k=2))
print("长期记忆召回（问「风险」）:", lt.recall("风险", k=1))
print("\n要点：短期记忆管「这一轮对话」，长期记忆管「跨会话的事实」。")
