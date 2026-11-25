"""MCP Client：在 Agent 中接入 MCP Server，把 MCP 工具转成 LangChain 工具。

教学要点：MCP 工具与本地 @tool 的区别只在「从哪来」——
本地工具是进程内函数调用；MCP 工具是通过 stdio 跨进程调用。
转成 LangChain 工具后，对 Agent 而言接口一致（都是 BaseTool）。

关键工程点（本文件最值得学的地方）：
1. stdio session 的生命周期必须覆盖整个 Agent 运行期——不能返回一个工具列表后
   就关闭 session。这里用一个后台线程常驻 event loop 持有 session，Agent 全程可用。
2. MCP 的 call_tool 是 async，而手写的 run_react 是同步循环。把 async 调用提交到
   后台 loop 并用 run_coroutine_threadsafe(...).result() 同步等待，桥接两种世界，
   这样 react.py 完全不用改。

运行（需先 pip install mcp）：python mcp/client.py
"""

import asyncio
import os
import sys
import threading
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
SERVER_PATH = BASE_DIR / "mcp" / "server.py"


class MCPBridgeTool(BaseTool):
    """把一个 MCP 工具包装成 LangChain BaseTool（保持同步 _run 接口）。"""

    name: str
    description: str
    args_schema: type[BaseModel]
    session: Any
    loop: Any
    tool_name: str

    def _run(self, **kwargs) -> str:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # 跨线程把 coroutine 提交给 MCP 专用 loop，同步等结果（不嵌套 asyncio.run）
        future = asyncio.run_coroutine_threadsafe(
            self.session.call_tool(self.tool_name, kwargs), self.loop
        )
        result = future.result(timeout=30)
        parts = [getattr(block, "text", "") for block in result.content]
        return "\n".join(p for p in parts if p) or "(工具返回空内容)"


def _schema_from_mcp(tool) -> type[BaseModel]:
    """把 MCP 工具的 inputSchema 转成 Pydantic 模型，供 Agent 做参数校验。"""
    # mcp 2.x 把 inputSchema 改为 input_schema；两种版本都兼容
    schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", {})
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
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


class MCPConnection:
    """在独立后台线程常驻 event loop，持有 MCP stdio session 全程存活。"""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._loop_worker, daemon=True)
        self.session = None
        self.tools_meta = []
        self._stdio_ctx = None
        self._session_ctx = None

    def _loop_worker(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _connect(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        # mcp 2.x 的 stdio_client 仍在 mcp.client.stdio，API 未变
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(SERVER_PATH)],
            env=os.environ.copy(),
        )
        # 手动进入 async 上下文并保持，不在 __aexit__ 前关闭
        self._stdio_ctx = stdio_client(params)
        read, write = await self._stdio_ctx.__aenter__()
        self._session_ctx = ClientSession(read, write)
        self.session = await self._session_ctx.__aenter__()
        await self.session.initialize()
        discovered = await self.session.list_tools()
        self.tools_meta = discovered.tools

    async def _close(self):
        if self._session_ctx is not None:
            await self._session_ctx.__aexit__(None, None, None)
        if self._stdio_ctx is not None:
            await self._stdio_ctx.__aexit__(None, None, None)

    def connect(self, timeout: float = 60.0):
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self._connect(), self.loop).result(timeout)
        return self.tools_meta

    def bridged_tools(self) -> list[BaseTool]:
        return [
            MCPBridgeTool(
                name=t.name,
                description=t.description or "",
                args_schema=_schema_from_mcp(t),
                session=self.session,
                loop=self.loop,
                tool_name=t.name,
            )
            for t in self.tools_meta
        ]

    def close(self):
        if self.session is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._close(), self.loop).result(10)
            except Exception:
                pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)


def load_and_run(question: str | None = None) -> str:
    """连接 MCP Server → 发现工具 → 跑 ReAct Agent → 关闭 session。"""
    conn = MCPConnection()
    try:
        tools_meta = conn.connect()
        print("MCP Server 暴露的工具:", [t.name for t in tools_meta])
        bridged = conn.bridged_tools()

        # 无 LLM / Key 失效时，直接演示一次跨进程工具调用
        if question is None:
            tool = next(t for t in bridged if t.name == "growth_rate_tool")
            result = tool._run(current=12.8, previous=10.0)
            print("直接调用 MCP 工具结果:", result)
            return result

        # 真实 Agent 模式：session 存活期内跑完整 ReAct
        from config import build_llm
        from react import run_react

        llm = build_llm().bind_tools(bridged)
        result = run_react(llm, question, {t.name: t for t in bridged})
        return result.answer
    finally:
        conn.close()


if __name__ == "__main__":
    q = " ".join(a for a in sys.argv[1:] if not a.startswith("--"))
    if "--agent" in sys.argv and q:
        print(load_and_run(q))
    else:
        print(load_and_run())
