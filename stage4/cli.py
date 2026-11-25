"""阶段 4 Agent 命令行：三种运行模式。

    python cli.py --mode react "分析财报增长情况"     # 手写 ReAct
    python cli.py --mode graph "分析财报增长情况"     # LangGraph 有状态
    python cli.py --mode plan  "分析财报增长情况"     # 规划-执行图
    python cli.py --mode supervisor "分析财报增长情况" # 多 Agent 协作
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage  # noqa: E402

from config import build_llm  # noqa: E402
from react import run_react  # noqa: E402
from tools import TOOL_MAP  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="阶段4：Agent 问答")
    p.add_argument("question", nargs="*")
    p.add_argument("--mode", default="react",
                   choices=["react", "graph", "plan", "supervisor"])
    p.add_argument("--max-steps", type=int, default=8)
    args = p.parse_args()
    q = " ".join(args.question) or "分析星云智能2024年的营收增长情况"
    llm = build_llm()

    if args.mode == "react":
        result = run_react(llm.bind_tools(list(TOOL_MAP.values())), q, TOOL_MAP,
                           max_steps=args.max_steps)
        print("\n回答:", result.answer)
        print("统计:", result.tokens, "| 停止原因:", result.stopped_by)
    elif args.mode == "graph":
        from graph import build_stateful_agent, run_with_thread
        r = run_with_thread(build_stateful_agent(llm), q, thread_id="cli")
        print("\n回答:", r["answer"])
        print(f"会话 {r['thread_id']} 共 {r['turns']} 条消息（同 thread 再问会记住上下文）")
    elif args.mode == "plan":
        from graph import build_plan_graph
        out = build_plan_graph(llm).invoke({"question": q})
        print("\n规划步骤:", out.get("steps"))
        print("执行结果:", out.get("results"))
        print("\n回答:", out.get("answer"))
    else:
        from graph import build_supervisor_agent
        r = build_supervisor_agent(llm).invoke(
            {"messages": [HumanMessage(content=q)]}
        )
        print("\n回答:", r["messages"][-1].content)


if __name__ == "__main__":
    main()
