# 阶段 4｜Agent 核心脚手架（第 9-12 周，约 80h）

> 验收锚点：**给定「分析这份财报并生成摘要报告」，Agent 自主完成 检索 → 计算 → 汇总**。
> 判定：工具链含检索+计算、答案数字与工具返回一致、最大步数内完成。

## 一、这个阶段在学什么

阶段 3 的 RAG 是"给模型喂资料"，阶段 4 的 Agent 是"让模型自己决定调什么工具、分几步走"。
核心心智模型只有一句话：

> **模型负责「决定调什么」，程序负责「执行并把结果喂回去」。**

所有复杂概念（ReAct / LangGraph / MCP / 多 Agent）都是这句话的不同封装层次。

## 二、文件结构（对应打卡表任务）

```
stage4/
├── config.py              # LLM / Embeddings / TokenCounter / StepLogger
├── corpus.py              # ★ 验收语料：虚构公司财报（文本 + 结构化数据）
├── tools/
│   └── finance_tools.py   # ① Function Calling：4 个工具（含 stage3 RAG 封装）
├── react.py               # ② 手写 ReAct 循环 + 护栏（最大步数/超时/日志）
├── memory.py              # ③ 短期记忆（窗口+摘要）+ 长期记忆（向量库）
├── graph.py               # ④ LangGraph：有状态 / 规划-执行 / 多 Agent
├── mcp/
│   ├── server.py          # ⑤ MCP Server（FastMCP/MCPServer 兼容写法封装工具）
│   └── client.py          # MCP Client（转成 LangChain 工具接入 Agent）
├── acceptance.py          # ★ 阶段验收脚本
├── cli.py                 # 四种模式命令行入口
└── examples/
    ├── 01_tool_calling.py # Function Calling 结构演示（离线）
    └── 02_memory.py       # 记忆演示（离线）
```

## 三、快速开始

```bash
cd stage4
# 0. 无需 API 的演示（全部离线，零成本）
C:/Python312/python.exe examples/01_tool_calling.py     # 看 tool_calls 结构
C:/Python312/python.exe examples/02_memory.py           # 看两种记忆
C:/Python312/python.exe acceptance.py                   # 离线模拟完整验收流程

# 1. 真实千问（需有效 Key，消耗额度）
C:/Python312/python.exe acceptance.py --live
C:/Python312/python.exe cli.py --mode react "分析2024年营收增长情况"
C:/Python312/python.exe cli.py --mode graph  "分析2024年营收增长情况"
C:/Python312/python.exe cli.py --mode plan   "分析2024年营收增长情况"
C:/Python312/python.exe cli.py --mode supervisor "分析2024年营收增长情况"

# 2. MCP（需 pip install mcp）
C:/Python312/python.exe mcp/client.py                   # 发现工具 + 直接调用演示
C:/Python312/python.exe mcp/client.py --agent "营收增长多少"  # 经 MCP 工具跑 Agent
```

## 四、五环节怎么映射到 Agent

| 打卡表任务 | 对应文件 | 关键点 |
|---|---|---|
| Function Calling（3+ 工具） | `tools/finance_tools.py` | `@tool` + Pydantic Schema；恶意表达式被 AST 白名单拦截 |
| ReAct 循环 | `react.py` | 手写 messages 序列：System → Human → AI(tool_calls) → Tool → AI... |
| MCP Server / 接入 | `mcp/` | MCPServer（v1 FastMCP / v2 MCPServer 兼容）封装；Client 把 MCP 工具桥接成 BaseTool；**session 生命周期必须覆盖整个 Agent 运行期** |
| 短期 / 长期记忆 | `memory.py` | 短期=窗口裁剪+摘要压缩；长期=向量库存取 |
| 规划与分解 | `graph.py::build_plan_graph` | plan → execute → summarize 三节点 StateGraph |
| LangGraph 有状态 | `graph.py::build_stateful_agent` | `InMemorySaver` + `thread_id` 隔离会话 |
| 多 Agent 协作 | `graph.py::build_supervisor_agent` | 主管分派给检索员/计算员子 Agent |
| 防死循环 | `react.py` | max_steps + 工具超时（线程池）+ 每步日志 + 失败反馈 |
| RAG 封装成工具 | `tools/finance_tools.py::rag_search` | 阶段 3 检索能力变成 Agent 可调用的工具 |

## 五、验收怎么判定（`acceptance.py`）

四条硬指标，全过才算 PASS：

1. **检索工具被调用**（get_metric 或 rag_search）
2. **计算工具被调用**（calculate 或 growth_rate）
3. **最大步数内完成**（stopped_by == "done"）
4. **答案含具体数字**（防"我分析了一下"这种空话）

离线模式用 `GenericFakeChatModel` 模拟一次完整 tool_call 序列，验证的是**循环与判定逻辑**；
真实模式才验证模型是否真会自主选工具。

## 六、关键坑点（实测/设计经验）

- **百炼 embedding 批量 ≤10 条** → `config.py::BatchingEmbeddings` 已处理（阶段 3 踩过的坑）
- **stdlib 路径冲突** → `tools/` 与 `mcp/` 都显式 `sys.path.insert`；工具层不要 import stage3 的 `corpus`（同名模块会抢先被导入）
- **eval 安全** → 计算器用 AST 白名单，只允许算术节点，拒绝 `__import__` 等调用
- **stdio session 生命周期** → MCP Client 用后台线程常驻 event loop 持有 session（`MCPConnection`），覆盖整个 Agent 运行期，不能先返回工具列表再关闭
- **async/sync 桥接** → MCP 的 `call_tool` 是 async，手写 `run_react` 是同步循环；用 `run_coroutine_threadsafe(...).result()` 桥接，**不能在 async 上下文里 `asyncio.run()` 嵌套**（会报 "cannot be called from a running event loop"）
- **mcp 2.x 破坏性变更** → `FastMCP` → `MCPServer`（`mcp.server.mcpserver`）、`inputSchema` → `input_schema`；代码已做双版本兼容，`requirements.txt` 用 `mcp>=1.0.0` 即可
- **本地 `mcp/` 目录与 pip 包同名** → 直接运行 `python mcp/client.py` 没问题；若要从代码导入用文件路径加载（`importlib.util.spec_from_file_location`），写 `from mcp.client import ...` 会导入 pip 包而非本地文件
- **工具失败要反馈给模型** → 异常转成 `ToolMessage` 塞回对话，让模型自我修正，而不是直接崩
- **API Key 401** → `sk-ws-` 开头的工作空间 Key 若失效，到百炼控制台检查状态；脚手架不硬编码 Key

## 七、当前环境状态（2026-11-25）

- Python 3.12（`C:\Python312\python.exe`），langgraph 1.2.22 / langchain 1.5.0 / pydantic 2.14.3 已装
- `mcp` 2.3.0 已装；MCP Server + Client 全链路**离线实测通过**：4 个工具跨进程发现并调用成功（calculator / growth_rate / get_metric / rag_search）
- 本地 `mcp/` 目录与 pip 包同名，从代码导入本地 client 要用文件路径加载（见坑点）
- 百炼 API Key 返回 401，**无法实时验证真实 function calling**；离线模拟全部通过
