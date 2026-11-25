"""示例1：Function Calling 全流程（离线，用假模型演示 tool_call 结构）。

真实模式把 fake 换成 build_llm().bind_tools(ALL_TOOLS) 即可。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from tools import ALL_TOOLS, TOOL_MAP  # noqa: E402

print("可用的工具（由 @tool + Pydantic Schema 定义）:")
for t in ALL_TOOLS:
    print(f"  {t.name}: {list(t.args)}")

# 模拟模型决定调用两个工具
fake = GenericFakeChatModel(messages=iter([
    AIMessage(content="", tool_calls=[
        {"name": "get_metric", "args": {"year": 2024, "metric": "revenue"}, "id": "1"},
        {"name": "growth_rate", "args": {"current": 12.8, "previous": 10.0}, "id": "2"},
    ]),
    AIMessage(content="营收12.8亿，增长28%。"),
]))
messages = [HumanMessage(content="2024年营收和增长率？")]
resp = fake.invoke(messages)
print("\n模型输出的 tool_calls:", resp.tool_calls)

messages.append(resp)
for tc in resp.tool_calls:
    t = TOOL_MAP[tc["name"]]
    obs = t.invoke(tc["args"])
    print(f"执行 {tc['name']} -> {obs}")
    messages.append(ToolMessage(content=str(obs), tool_call_id=tc["id"]))

final = fake.invoke(messages)
print("\n最终回答:", final.content)
print("\n要点：模型负责「决定调什么」，程序负责「执行并把结果喂回去」——这就是 Function Calling。")
