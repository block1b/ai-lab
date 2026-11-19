# ai-lab — AI 应用工程实践

从「会用大模型 API」到「工程化落地 Agent 与工具调用」的三段实践记录，每一段都有可运行的代码与量化验收标准。

| 子项目 | 内容 | 技术栈 |
|---|---|---|
| [`stage1/`](stage1/) | **LLM API 基础**：聊天接口、token / 上下文 / 温度实验、流式输出 | Python + LangChain + 通义千问（OpenAI 兼容模式） |
| [`stage2/`](stage2/) | **结构化输出工程**：Pydantic Schema 校验、few-shot、错误反馈重试，验收标准为连续 100 次抽取格式正确率 ≥ 98% | Python + Pydantic + 指数退避重试 |
| [`agent-demo/`](agent-demo/) | **MCP 工具调用 Agent**：MCP Server（stdio）暴露 3 个工具，客户端自动转 function calling 并跑完整推理循环，含危险操作护栏与多租户越权校验 | Go + mark3labs/mcp-go |

## 为什么要按这个顺序

三条线对应 AI 应用开发的三个层次：

1. **把模型当 API 用熟**（stage1）——理解 token、上下文窗口、温度这些基本参数的实际影响，而不是背概念；
2. **把模型输出约束成可靠数据**（stage2）——LLM 应用进入生产的第一道门槛是「输出可解析、可校验、可重试」，这一层用 Schema + 反馈重试做到 98%+ 的格式正确率；
3. **让模型调用真实世界**（agent-demo）——工具调用不是加个 if 判断，而是协议（MCP）、安全边界（二次确认、租户校验、轮次上限）和可观测性（traceId、耗时）的完整工程问题。

## 各项目快速入口

```bash
# stage1：命令行问答 + token 统计（需在 stage1/.env 配置 DASHSCOPE_API_KEY）
cd stage1 && pip install -r requirements.txt && python cli.py

# stage2：信息抽取器（先跑本地演示，不花 API 额度）
cd stage2 && python examples/01_schema_demo.py && python cli.py "屏幕很好但电池太差"

# agent-demo：MCP Server + Agent 循环（mock 模式无需任何 API Key）
cd agent-demo && go build -o bin/mcp-server ./cmd/server && go run ./cmd/agent -mock
```

## 设计原则

- **密钥隔离**：所有 Key 走 `.env`（已被 `.gitignore` 排除），仓库只保留 `.env.example` 模板
- **可验收**：每个阶段有明确验收指标（token 统计、格式正确率、mock 链路走通），不是「跑起来就行」
- **安全边界优先**：危险操作二次确认、跨租户校验、轮次上限、超时重试——把不确定的模型输出约束在确定的执行边界内

## License

[MIT](LICENSE)
