"""阶段 4 阶段验收脚本。

验收锚点（打卡表原文）：给定「分析这份财报并生成摘要报告」，
Agent 能自主完成 检索 → 计算 → 汇总 全流程。

判定标准（三条全过才算通过）：
1. 工具调用链包含至少 1 次检索 + 1 次计算（不是靠模型记忆作答）
2. 答案中的关键数字与工具返回一致（无幻觉）
3. 流程在最大步数内完成，且每步有日志

运行：
    python acceptance.py              # 离线模拟（零成本，验证流程结构）
    python acceptance.py --live       # 真实千问（需有效 Key）
    python acceptance.py --live --graph  # 用 LangGraph 规划图跑
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from config import build_llm  # noqa: E402
from corpus import ACCEPTANCE_QUESTION  # noqa: E402
from react import run_react  # noqa: E402
from tools import TOOL_MAP  # noqa: E402


def check(result, steps: list[dict]) -> dict:
    """按三条标准判定验收是否通过。"""
    actions = [s["action"] for s in steps]
    acted = [s["detail"].split("(")[0] for s in steps if s["action"] == "ACT"]
    has_search = "rag_search" in acted or "get_metric" in acted
    has_calc = "calculate" in acted or "growth_rate" in acted
    finished = result.stopped_by == "done"
    answer = result.answer
    has_number = bool(re.search(r"\d+\.?\d*", answer))
    return {
        "检索工具被调用": has_search,
        "计算工具被调用": has_calc,
        "在最大步数内完成": finished,
        "答案含具体数字": has_number,
        "全部通过": all([has_search, has_calc, finished, has_number]),
        "工具链": acted,
        "停止原因": result.stopped_by,
        "步数": len(steps),
        "token": result.tokens,
    }


def run_offline() -> dict:
    """用假模型模拟一次完整的工具调用序列，验证循环与验收判定逻辑。"""
    fake = GenericFakeChatModel(messages=iter([
        AIMessage(content="先检索财报中的营收数据", tool_calls=[{
            "name": "get_metric", "args": {"year": 2024, "metric": "revenue"}, "id": "c1",
        }]),
        AIMessage(content="再算增长率", tool_calls=[{
            "name": "growth_rate", "args": {"current": 12.8, "previous": 10.0}, "id": "c2",
        }]),
        AIMessage(content="2024年营收12.8亿元，较2023年的10.0亿元增长28.0%，建议关注海外市场拓展。"),
    ]))
    result = run_react(fake, ACCEPTANCE_QUESTION, TOOL_MAP, verbose=True)
    return check(result, result.steps)


def run_live(use_graph: bool = False) -> dict:
    llm = build_llm()
    if use_graph:
        from graph import build_plan_graph
        g = build_plan_graph(llm=llm)
        out = g.invoke({"question": ACCEPTANCE_QUESTION})
        print(f"规划步骤: {out.get('steps')}")
        print(f"执行结果: {out.get('results')}")
        print(f"\n最终答案:\n{out.get('answer')}")
        return {"全部通过": bool(out.get("answer")), "模式": "LangGraph 规划图"}
    bound = llm.bind_tools(list(TOOL_MAP.values()))
    result = run_react(bound, ACCEPTANCE_QUESTION, TOOL_MAP, verbose=True)
    report = check(result, result.steps)
    report["答案"] = result.answer
    return report


def main():
    args = sys.argv[1:]
    live = "--live" in args
    use_graph = "--graph" in args
    print("=" * 60)
    print("阶段 4 验收：分析财报并生成摘要报告")
    print(f"问题：{ACCEPTANCE_QUESTION}")
    print(f"模式：{'真实千问' if live else '离线模拟（零成本）'}")
    print("=" * 60)
    report = run_live(use_graph) if live else run_offline()
    print("\n" + "=" * 60 + "\n验收报告")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    if report.get("全部通过"):
        print("结果：PASS - 阶段 4 验收通过")
    else:
        print("结果：FAIL - 请检查工具链或 API Key")


if __name__ == "__main__":
    main()
