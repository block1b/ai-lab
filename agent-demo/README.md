# agent-demo — MCP Server + LLM Function Calling 最小可运行样例

一个 Go 实现的「设备运维 Agent」：MCP Server 暴露 3 个工具，Agent 客户端把工具清单自动转成大模型的 function calling 定义，跑「决策 → 调用 → 结果回填 → 再决策」的完整循环。

定位：**代码量小、链路完整、可讲透**，适合作为 MCP 协议与 LLM 工具调用的学习参考，也可以作为自建 Agent 网关的起步骨架。

## 目录结构

```
agent-demo/
├── cmd/server/main.go       MCP Server（stdio 传输），3 个工具 + 危险操作护栏
├── cmd/agent/main.go        Agent 客户端：工具发现 → function calling 循环 / mock 演示
├── internal/device/store.go 内存数据源（设备、告警、重启）
└── bin/                     构建产物（不入库）
```

## 运行

```bash
# 1. 编译 MCP Server
go build -o bin/mcp-server ./cmd/server        # Windows: bin/mcp-server.exe

# 2. 不依赖大模型，脚本化演示完整链路（推荐先跑这个）
go run ./cmd/agent -mock

# 3. 接真实大模型（OpenAI 兼容接口）
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://api.deepseek.com/v1   # 可选，默认就是它
export OPENAI_MODEL=deepseek-chat                    # 可选
go run ./cmd/agent -q "查询有哪些设备离线，分析原因，未经确认不要重启"
```

## 三个工具

| 工具 | 参数 | 说明 |
|---|---|---|
| `list_devices` | tenant(必填) / status(可选) | 查设备清单 |
| `query_alarms` | tenant(必填) / since_minutes(默认 60) / level(可选) | 查告警，按时间倒序 |
| `restart_device` | device_id / tenant / confirm(均必填) | 危险操作，confirm=false 时只校验不执行 |

## 四个实现细节

**1. stdout 是协议流，日志只能走 stderr**
stdio 模式下客户端与 Server 通过 stdout 传 JSON-RPC，任何 `fmt.Println` 打到 stdout 都会污染协议、导致握手失败，日志必须重定向到 stderr。

**2. 工具 schema 自动转 function calling，不手写两遍**
MCP 的 `tool.InputSchema` 本身就是 JSON Schema，直接 `json.Marshal` 填进 function calling 的 `parameters`。Server 新增工具，客户端代码零改动——协议层解耦的实际收益。

**3. 危险操作三重护栏**
- schema 层将 `confirm` 设为必填布尔值，模型无法"忘记"传；
- `confirm=false` 时返回 `executed:false` + 提示语，**而不是报错**——报错会让模型以为工具坏了并反复重试，反而更危险；
- `restart` 前校验 `tenant` 归属，防跨租户越权。

**4. 循环必须有刹车**
`maxRounds=6` 防死循环、单次模型请求 60s 超时、上游 429/5xx 退避重试一次、每次工具调用打 traceId 与耗时。没有这些约束，一次模型异常就能把账单和下游接口一起打爆。

## 接到 Claude Desktop / Cursor

```json
{
  "mcpServers": {
    "iot-ops": {
      "command": "/path/to/agent-demo/bin/mcp-server",
      "args": []
    }
  }
}
```

把 `command` 替换为本机编译出的可执行文件路径即可（stdio 传输，本地运行，无需端口）。

## 扩展方向

- **加 RAG**：设备手册切分入 pgvector，暴露 `search_docs(query)` 工具，让模型先查文档再动手
- **加 SSE**：把对话包成 HTTP 接口，前端流式渲染，工具调用状态实时推送
- **加鉴权**：MCP Server 侧校验调用方身份（JWT/OAuth），工具粒度授权
- **加观测**：工具调用耗时、失败率、token 消耗打点，接入 Prometheus

## License

[MIT](LICENSE)
