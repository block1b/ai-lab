"""LangGraph 有状态工作流：checkpoint 记忆 + 规划 + 多 Agent 协作。

三层递进（对应打卡表第 11-12 周任务）：
1. create_react_agent + InMemorySaver：有状态 Agent，同一 thread_id 跨轮记住上下文
2. plan_and_execute：先让 LLM 拆解任务成步骤，再逐步执行（任务规划与分解）
3. supervisor：一个主管 Agent 把任务分派给 specializing 子 Agent（多 Agent 协作）

LangGraph 1.x API：create_react_agent 在 langgraph.prebuilt；
checkpointer 用 langgraph.checkpoint.memory.InMemorySaver。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from config import StepLogger, build_llm
from tools import ALL_TOOLS

PLAN_SYSTEM = """你是任务规划器。把用户任务拆成 2-4 个可执行的子步骤。
只输出 JSON：{"steps": ["步骤1", "步骤2", ...]}
子步骤必须是工具能完成的具体动作，例如"检索2023和2024年营收数据"。"""


def build_stateful_agent(llm=None, tools=None, checkpointer=None):
    """① 有状态 Agent：同一 thread_id 下对话历史自动持久化。"""
    llm = llm or build_llm()
    tools = tools if tools is not None else ALL_TOOLS
    checkpointer = checkpointer or InMemorySaver()
    return create_react_agent(llm, tools, checkpointer=checkpointer)


def run_with_thread(agent, question: str, thread_id: str = "default") -> dict:
    """通过 config.configurable.thread_id 隔离不同会话的记忆。"""
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return {
        "answer": result["messages"][-1].content,
        "turns": len(result["messages"]),
        "thread_id": thread_id,
    }


class PlanState(BaseModel):
    question: str
    steps: list[str] = []
    results: list[str] = []
    answer: str = ""


def build_plan_graph(llm=None, tools=None):
    """② 规划-执行图：规划节点拆步骤，执行节点逐步跑，最后汇总。"""
    llm = llm or build_llm()
    tools = tools if tools is not None else ALL_TOOLS
    tool_map = {t.name: t for t in tools}
    import json
    import re

    def plan(state: dict) -> dict:
        resp = llm.invoke([
            SystemMessage(content=PLAN_SYSTEM),
            HumanMessage(content=state["question"]),
        ])
        m = re.search(r"\{.*\}", resp.content, re.DOTALL)
        steps = json.loads(m.group()).get("steps", []) if m else [state["question"]]
        StepLogger(verbose=True).log(1, "PLAN", f"拆解为 {len(steps)} 步: {steps}")
        return {"steps": steps}

    def execute(state: dict) -> dict:
        results = []
        logger = StepLogger(verbose=True)
        for i, step in enumerate(state["steps"], 2):
            # 每步用一个轻量 ReAct：让 LLM 选工具执行这一步
            from langchain_core.messages import ToolMessage
            bound = llm.bind_tools(tools)
            msgs = [SystemMessage(content="用工具完成这一步，只调用工具，不要解释。"),
                    HumanMessage(content=step)]
            resp = bound.invoke(msgs)
            for tc in resp.tool_calls:
                t = tool_map.get(tc["name"])
                obs = t.invoke(tc["args"]) if t else f"未知工具 {tc['name']}"
                results.append(f"[{step}] {obs}")
                logger.log(i, "EXEC", f"{tc['name']} -> {str(obs)[:80]}")
        return {"results": results}

    def summarize(state: dict) -> dict:
        ctx = "\n".join(state["results"])
        resp = llm.invoke([
            SystemMessage(content="基于以下执行结果，用不超过150字回答问题。不要编造未提供的数字。"),
            HumanMessage(content=f"问题：{state['question']}\n\n执行结果：\n{ctx}"),
        ])
        return {"answer": resp.content.strip()}

    graph = StateGraph(dict)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("summarize", summarize)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


def build_supervisor_agent(llm=None):
    """③ 多 Agent 协作：主管用路由工具把任务分派给「检索员」和「计算员」。"""
    llm = llm or build_llm()

    from langchain_core.tools import tool

    @tool
    def ask_researcher(question: str) -> str:
        """把文本检索类问题交给检索员 Agent。"""
        messages = [SystemMessage(content="你是财报检索员。用 rag_search 工具检索，直接返回原文要点。"),
                    HumanMessage(content=question)]
        r = llm.bind_tools([t for t in ALL_TOOLS if t.name == "rag_search"]).invoke(messages)
        for tc in r.tool_calls:
            if tc["name"] == "rag_search":
                return next(t for t in ALL_TOOLS if t.name == "rag_search").invoke(tc["args"])
        return r.content

    @tool
    def ask_calculator(question: str) -> str:
        """把数值计算类问题交给计算员 Agent。"""
        calc_tools = [t for t in ALL_TOOLS if t.name in ("calculate", "growth_rate", "get_metric")]
        messages = [SystemMessage(content="你是财务计算员。必须用工具计算，返回数值结果。"),
                    HumanMessage(content=question)]
        r = llm.bind_tools(calc_tools).invoke(messages)
        out = []
        tool_map = {t.name: t for t in calc_tools}
        for tc in r.tool_calls:
            t = tool_map.get(tc["name"])
            if t:
                out.append(str(t.invoke(tc["args"])))
        return "\n".join(out) if out else r.content

    supervisor_tools = [ask_researcher, ask_calculator]
    return create_react_agent(
        llm, supervisor_tools,
        prompt="你是主管。遇到文本事实问题派给 ask_researcher，遇到数值计算派给 ask_calculator。"
               "综合两位专家的结果给出最终答案。",
    )


if __name__ == "__main__":
    print("LangGraph 工作流模块加载成功")
    print("  - build_stateful_agent: 有状态 Agent + checkpoint")
    print("  - build_plan_graph: 规划-执行-汇总图")
    print("  - build_supervisor_agent: 多 Agent 协作")
