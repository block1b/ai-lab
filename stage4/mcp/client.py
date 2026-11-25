"""MCP Client：在 Agent 中接入 MCP Server，把 MCP 工具转成 LangChain 工具。

教学要点：MCP 工具与本地 @tool 的区别只在「从哪来」——
本地工具是进程内函数调用；MCP 工具是通过 stdio 跨进程调用。
转成 LangChain 工具后，对 Agent 而言接口一致（都是 BaseTool）。

关键工程点：stdio session 的生命周期必须覆盖整个 Agent 运行期。
所以 load_and_run() 把「连接 → 发现工具 → 跑 Agent → 关闭」放在同一个 async 上下文，
不能先返回一个工具列表再在外部用——那样 session 已关闭，调用会失败。

运行（需先 pip install mcp）：python mcp/client.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
SERVER_PATH = BASE_DIR / "mcp" / "server.py"


class MCPBridgeTool(BaseTool):
    """把一个 MCP 工具包装成 LangChain BaseTool。"""

    name: str
    description: str
    args_schema: type[BaseModel]
    session: object
    tool_name: str

    def _run(self, **kwargs) -> str:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        result = asyncio.run(self.session.call_tool(self.tool_name, kwargs))
        parts = [getattr(block, "text", "") for block in result.content]
        return "\n".join(p for p in parts if p) or "(工具返回空内容)"


def _schema_from_mcp(tool) -> type[BaseModel]:
    """把 MCP 工具的 inputSchema 转成 Pydantic 模型，供 Agent 做参数校验。"""
    props = tool.inputSchema.get("properties", {})
    required = set(tool.inputSchema.get("required", []))
    annotations, defaults = {}, {}
    for name, spec in props.items():
        typ = {"string": str, "number": float, "integer": int, "boolean": bool}.get(
            spec.get("type", "string"), str
        )
        annotations[name] = typ
        desc = spec.get("description", "")
        defaults[name] = (Field(..., description=desc) if name in required
                          else Field(default=None, description=desc))
    return type("MCPArgs", (BaseModel,), {"__annotations__": annotations, **defaults})


async def load_and_run(question: str | None = None) -> str:
    """连接 MCP Server → 发现工具 → 跑 ReAct Agent → 自动关闭 session。"""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
        env=os.environ.copy(),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            discovered = await session.list_tools()
            print("MCP Server 暴露的工具:", [t.name for t in discovered.tools])

            bridged = [
                MCPBridgeTool(
                    name=t.name,
                    description=t.description or "",
                    args_schema=_schema_from_mcp(t),
                    session=session,
                    tool_name=t.name,
                )
                for t in discovered.tools
            ]

            # 本地跑不了 LLM（Key 可能失效）时，演示一次直接工具调用即可
            if question is None:
                tool = next(t for t in bridged if t.name == "growth_rate_tool")
                result = tool._run(current=12.8, previous=10.0)
                print("直接调用 MCP 工具结果:", result)
                return result

            # 真实 Agent 模式：在 session 存活期内跑完整 ReAct
            from config import build_llm
            from react import run_react
            llm = build_llm().bind_tools(bridged)
            result = run_react(llm, question, {t.name: t for t in bridged})
            return result.answer


if __name__ == "__main__":
    q = " ".join(a for a in sys.argv[1:] if not a.startswith("--"))
    if "--agent" in sys.argv and q:
        print(asyncio.run(load_and_run(q)))
    else:
        asyncio.run(load_and_run())
