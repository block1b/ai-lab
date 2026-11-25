"""Agent 记忆：短期（对话历史）+ 长期（向量库持久化事实）。

短期记忆的本质是「窗口管理」：LLM 上下文有限，不能无限堆历史。
这里实现两种策略：保留最近 N 轮 + 超出时把早期对话压缩成摘要。

长期记忆的本质是「把事实写进向量库，下次按语义取回」——
和 stage3 的 RAG 是同一套机制，区别在于写入的是对话中产生的结论。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

BASE_DIR = Path(__file__).resolve().parent


class ShortTermMemory:
    """对话历史窗口：保留最近 max_turns 轮，更早的丢弃（可接入摘要）。"""

    def __init__(self, max_turns: int = 6, summarizer=None):
        self.max_turns = max_turns
        self.summarizer = summarizer  # 可选：llm.invoke 生成摘要的函数
        self.messages: list[BaseMessage] = []
        self.archive_summary: str = ""

    def add(self, message: BaseMessage):
        self.messages.append(message)

    def get(self) -> list[BaseMessage]:
        head = ([SystemMessage(content=f"此前对话摘要：{self.archive_summary}")]
                if self.archive_summary else [])
        return head + self.messages

    def trim(self) -> int:
        """轮数 = 消息数//2（一问一答）。超出时丢弃最早的一轮。"""
        dropped = 0
        while len(self.messages) // 2 > self.max_turns:
            removed = self.messages[:2]
            self.messages = self.messages[2:]
            dropped += 1
            if self.summarizer:
                text = "\n".join(f"{type(m).__name__}: {m.content}" for m in removed)
                new = self.summarizer(text)
                self.archive_summary = (self.archive_summary + "\n" + new).strip()
        return dropped

    def __len__(self):
        return len(self.messages)


class LongTermMemory:
    """基于 stage3 同款向量库的长期事实存储。

    默认用内存向量库；生产可换 Chroma/Qdrant 做持久化。
    """

    def __init__(self, embeddings):
        from langchain_core.vectorstores import InMemoryVectorStore
        self.store = InMemoryVectorStore(embeddings)
        self.facts: list[str] = []

    def remember(self, fact: str, source: str = "对话"):
        self.facts.append(fact)
        self.store.add_texts([fact], metadatas=[{"source": source}])

    def recall(self, query: str, k: int = 3) -> list[str]:
        if not self.facts:
            return []
        docs = self.store.similarity_search(query, k=k)
        return [d.page_content for d in docs]


if __name__ == "__main__":
    from langchain_core.messages import AIMessage

    mem = ShortTermMemory(max_turns=2)
    for i in range(6):
        mem.add(HumanMessage(content=f"问题{i}"))
        mem.add(AIMessage(content=f"回答{i}"))
    dropped = mem.trim()
    print(f"短期记忆：丢弃 {dropped} 轮，剩余 {len(mem)//2} 轮")
    print("保留的消息:", [m.content for m in mem.get()])

    # 用假 embedding 演示长期记忆（无 API）
    from langchain_core.embeddings import DeterministicFakeEmbedding
    try:
        import numpy  # noqa: F401
        lt = LongTermMemory(DeterministicFakeEmbedding(size=16))
        lt.remember("星云智能2024年营收12.8亿元")
        lt.remember("星云智能2024年净利润1.92亿元")
        print("长期记忆召回:", lt.recall("营收"))
    except ImportError:
        print("长期记忆演示跳过（需 numpy）")
