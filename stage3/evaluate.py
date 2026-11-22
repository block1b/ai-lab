"""评估闭环：阶段 3 三项硬指标。

- Recall@5：检索前5中是否命中人工标注的相关文档（纯计算，不需 LLM）
- Faithfulness：答案中的每个事实是否都能从检索片段找到依据（LLM 裁判）
- 幻觉率：应该拒答的问题上是否编造了答案（规则 + LLM 辅助）

用法：
    python evaluate.py --mode hybrid --rerank   # 完整评估
    python evaluate.py --mode vector            # 只看向量检索
"""

import argparse

from config import build_embeddings, build_llm
from corpus import EVAL_QUESTIONS, load_documents
from chunking import chunk
from vectorstore import build_store
from retrieval import BM25Retriever, vector_retrieve, hybrid_retrieve
from rerank import llm_rerank
from rag import answer, is_refusal

REFUSAL_QUESTIONS = {"低功耗模式下续航多久？", "支持无线充电吗？", "能监测血压吗？"}


def recall_at_k(retrieved_docs, relevant_sources: list[str], k: int = 5) -> float:
    """命中率：前 k 个检索结果中是否包含任一相关来源。"""
    top_sources = {d.metadata.get("source") for d in retrieved_docs[:k]}
    return 1.0 if top_sources & set(relevant_sources) else 0.0


FAITH_SYSTEM = """你是事实核查员。判断【答案】中的每个事实是否都能从【参考资料】找到依据。
只输出 JSON：{"faithful": true 或 false, "reason": "简短理由"}。
若答案包含参考资料没有的信息，判 false。不要输出其它文字。"""


def faithfulness(llm, ans: str, docs) -> tuple[bool, str]:
    import json
    import re
    from langchain_core.messages import HumanMessage, SystemMessage

    ctx = "\n".join(d.page_content for d in docs)
    try:
        resp = llm.invoke([
            SystemMessage(content=FAITH_SYSTEM),
            HumanMessage(content=f"【参考资料】\n{ctx}\n\n【答案】\n{ans}"),
        ])
        m = re.search(r"\{.*\}", resp.content, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        return bool(data.get("faithful", False)), data.get("reason", "")
    except Exception as e:
        return False, f"裁判调用失败: {e}"


def main():
    parser = argparse.ArgumentParser(description="阶段3 RAG 评估")
    parser.add_argument("--mode", default="hybrid", choices=["vector", "bm25", "hybrid"])
    parser.add_argument("--rerank", action="store_true", help="是否启用 LLM 重排")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--faith", action="store_true", help="是否跑 LLM 忠实度评估（耗时/花钱）")
    args = parser.parse_args()

    llm = build_llm()
    emb = build_embeddings()
    docs = load_documents()
    chunks = chunk(docs, strategy="semantic")
    store = build_store(chunks, emb)
    bm25 = BM25Retriever(chunks)
    print(f"语料 {len(docs)} 篇 -> {len(chunks)} 块 | 检索模式: {args.mode}\n")

    recalls = []
    hallucinated = 0
    refusal_total = 0
    faith_results = []

    for item in EVAL_QUESTIONS:
        q = item["question"]
        if args.mode == "vector":
            retrieved = vector_retrieve(store, q, k=args.k * 2)
        elif args.mode == "bm25":
            retrieved = bm25.search(q, k=args.k * 2)
        else:
            retrieved = hybrid_retrieve(store, bm25, q, k=args.k * 2)

        if args.rerank:
            retrieved = llm_rerank(llm, q, retrieved, top_k=args.k)
        else:
            retrieved = retrieved[:args.k]

        top_docs = [d for d, _ in retrieved]
        r = recall_at_k(top_docs, item["relevant"], k=args.k)
        recalls.append(r)

        # 拒答类问题：检查是否编造
        if q in REFUSAL_QUESTIONS:
            refusal_total += 1
            res = answer(llm, q, top_docs)
            if not is_refusal(res["answer"]):
                hallucinated += 1

        # 忠实度：抽样或全量
        if args.faith and item["relevant"]:
            res = answer(llm, q, top_docs)
            ok, reason = faithfulness(llm, res["answer"], top_docs)
            faith_results.append(ok)

        print(f"[{'HIT' if r else 'MISS'}] {q[:30]:<32} {[d.metadata['source'] for d in top_docs[:2]]}")

    n = len(EVAL_QUESTIONS)
    print("\n" + "=" * 50)
    print(f"Recall@{args.k}   : {sum(recalls)/n*100:.1f}%  ({int(sum(recalls))}/{n})   目标 ≥ 90%")
    if faith_results:
        print(f"Faithfulness : {sum(faith_results)/len(faith_results)*100:.1f}%   目标 ≥ 90%")
    if refusal_total:
        rate = hallucinated / refusal_total * 100
        print(f"幻觉率      : {rate:.1f}%  ({hallucinated}/{refusal_total})   目标 < 5%")
    print("=" * 50)


if __name__ == "__main__":
    main()
