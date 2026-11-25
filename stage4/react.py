"""手写 ReAct 循环：思考 → 行动 → 观察（不依赖 LangGraph 封装）。

为什么要手写：LangGraph 的 create_react_agent 是黑盒，学阶段 4 必须先看清楚
一次 tool call 到底发生了什么——模型输出 tool_calls → 程序执行工具 →
把工具结果作为 ToolMessage 塞回对话 → 模型继续推理，直到不再调用工具。

护栏（打卡表「防死循环」要求）：
- max_steps：最大循环轮数
- timeout：单次工具调用超时
- StepLogger：每步日志
- 工具失败反馈：异常转成 ToolMessage 让模型自我修正
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402

from config import StepLogger, TokenCounter  # noqa: E402

REACT_SYSTEM = """你是一个能使用工具的分析助手。遵循 ReAct 流程：
1. 思考：需要什么信息才能回答问题？
2. 行动：调用合适的工具获取信息（一次可调多个）。
3. 观察：查看工具返回结果。
4. 重复直到信息足够，然后给出最终答案。

规则：
- 回答财报问题时，数值必须从工具获取，不要凭记忆编造。
- 增长率/利润率等计算必须调用 calculate 或 growth_rate 工具。
- 信息足够时直接给最终答案，不要无意义地反复调用工具。"""


class AgentResult:
    def __init__(self, answer: str, steps: list[dict], tokens: dict, stopped_by: str):
        self.answer = answer
        self.steps = steps
        self.tokens = tokens
        self.stopped_by = stopped_by  # "done" / "max_steps" / "timeout" / "error"

    def __repr__(self):
        return (f"AgentResult(answer={self.answer[:60]!r}, steps={len(self.steps)}, "
                f"tokens={self.tokens}, stopped_by={self.stopped_by!r})")


def run_react(llm_with_tools, question: str, tool_map: dict,
              max_steps: int = 8, timeout: float = 30.0,
              verbose: bool = True) -> AgentResult:
    """执行 ReAct 循环。

    llm_with_tools: llm.bind_tools(tools) 的返回值
    tool_map: {工具名: BaseTool}，用 .invoke() 执行
    """
    logger = StepLogger(verbose=verbose)
    counter = TokenCounter()
    messages = [
        SystemMessage(content=REACT_SYSTEM),
        HumanMessage(content=question),
    ]

    for step in range(1, max_steps + 1):
        started = time.time()
        try:
            # 用 callback 统计这轮 token（结构化工具调用拿不到 response，靠 callback 截获）
            response = llm_with_tools.invoke(messages, config={"callbacks": [counter]})
        except Exception as exc:
            logger.log(step, "LLM_ERROR", str(exc)[:120], started)
            return AgentResult(f"LLM 调用失败: {exc}", logger.steps,
                               _token_dict(counter), "error")

        logger.log(step, "THINK", _brief(response.content), started)

        if not response.tool_calls:
            logger.log(step, "FINAL", _brief(response.content))
            return AgentResult(response.content.strip(), logger.steps,
                               _token_dict(counter), "done")

        # 执行工具（一次可能有多个 tool_call）
        for tc in response.tool_calls:
            name, args = tc["name"], tc["args"]
            tool = tool_map.get(name)
            t0 = time.time()
            if tool is None:
                observation = f"错误：未知工具 {name}"
                logger.log(step, "ACT", f"{name} 不存在", t0)
            else:
                try:
                    # 超时护栏：用线程跑工具，超时则放弃
                    observation = _call_with_timeout(tool.invoke, args, timeout)
                    logger.log(step, "ACT", f"{name}({_brief(str(args), 60)})", t0)
                except Exception as exc:
                    observation = f"工具执行失败: {type(exc).__name__}: {exc}"
                    logger.log(step, "ACT_ERROR", f"{name}: {str(exc)[:80]}", t0)
            logger.log(step, "OBSERVE", _brief(observation, 120))
            messages.append(ToolMessage(content=str(observation), tool_call_id=tc["id"]))

        # 把模型这轮的 tool_calls 回复也放进历史（必须，否则消息序列不合法）
        messages.append(response)

    logger.log(max_steps, "STOP", "达到最大步数，强制终止")
    return AgentResult("达到最大步数，未能完成分析。请检查工具或问题复杂度。",
                       logger.steps, _token_dict(counter), "max_steps")


def _call_with_timeout(fn, arg, timeout: float):
    """用守护线程实现工具超时；超时返回错误字符串而不是让 Agent 卡死。"""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, arg)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return f"工具调用超时（>{timeout}s）"


def _token_dict(counter: TokenCounter) -> dict:
    return {"prompt": counter.prompt, "completion": counter.completion,
            "total": counter.total, "calls": counter.calls}


def _brief(text: str, limit: int = 80) -> str:
    if not text:
        return ""
    s = " ".join(text.split())
    return s if len(s) <= limit else s[:limit] + "..."


if __name__ == "__main__":
    # 无 API 的演示：用一个假的 bind_tools 模型验证循环结构
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from tools import ALL_TOOLS, TOOL_MAP

    fake = GenericFakeChatModel(messages=iter([
        AIMessage(content="需要计算", tool_calls=[{
            "name": "growth_rate", "args": {"current": 12.8, "previous": 10.0},
            "id": "call_1",
        }]),
        AIMessage(content="营收增长 28%，可以回答了。"),
    ]))
    result = run_react(fake, "计算营收增长率", TOOL_MAP, verbose=True)
    print("\n结果:", result)
