"""环节 2：向量化 + 建库。

默认用 LangChain 内置 InMemoryVectorStore（零额外依赖，开箱即跑）。
文档带 source 元数据，检索时可溯源。
"""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore


def build_store(chunks: list[Document], embeddings: Embeddings) -> InMemoryVectorStore:
    """把切好的块向量化并存入向量库。"""
    return InMemoryVectorStore.from_documents(chunks, embeddings)


def vector_search(store: InMemoryVectorStore, query: str, k: int = 5):
    """纯向量检索，返回 (Document, 距离) 列表（距离越小越相关）。"""
    return store.similarity_search_with_score(query, k=k)
