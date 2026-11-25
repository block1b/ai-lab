# go-lab：用 Go（Eino）重写的学习前 4 阶段

与仓库里的 Python 版 `stage1/` ~ `stage4/` **并存对照**——同一套概念，两种语言实现。
适合你对照着看"LangChain 的概念在 Go 里叫什么"。

## 一、技术栈

| 项 | 选型 | 说明 |
|---|---|---|
| 框架 | **Eino** v0.9.23（字节 CloudWeGo） | Go 原生泛型设计，组件/编排对应 LangChain 概念 |
| 模型接入 | `eino-ext/components/model/openai` | 走千问 OpenAI 兼容端点，复用 `stage1/.env` |
| Embedding | `eino-ext/components/embedding/openai` | text-embedding-v3，自动分批 ≤10 条 |
| 工具 Schema | `utils.InferTool` + struct tag | Go 版的 Pydantic：tag 自动生成 JSON Schema |
| Agent | `flow/agent/react.NewAgent` | Eino 内置 ReAct，MaxStep 是防死循环护栏 |
| MCP | `mark3labs/mcp-go` v1.5.0 | 与你 `agent-demo/` 一致；注意它要求 Go ≥1.25.5 |
| 向量库 | 自写内存实现 | Eino 无内置内存库；纯 Go 余弦相似度 40 行，零额外依赖 |

Go 版本：go.mod 声明 1.25.5（mcp-go 要求），本机 go1.23.4 会自动切到已装的 1.26.11 工具链。

## 二、目录结构

```
go-lab/
├── go.mod / go.sum
├── pkg/config/           # 共享配置：读 stage1/.env、构造 ChatModel/Embedder
├── stage1/               # LLM 调用基础
│   ├── main.go           # CLI：单次/交互/温度/流式 + token 统计
│   └── experiments/      # temperature 对比、流式实验
├── stage2/               # 结构化输出
│   ├── schema.go         # ReviewExtract + Validate（Go 版 Pydantic）
│   ├── prompts.go        # 结构化 Prompt + few-shot
│   ├── extractor.go      # tool calling 承载结构化输出 + 错误反馈重试
│   └── cmd/              # --demo 本地校验 / --eval 批量验收
├── stage3/               # RAG 五环节
│   ├── corpus.go         # 15 篇文档 + 41 个标注问题
│   ├── chunking.go       # 固定 / 语义分块
│   ├── vectorstore.go    # 内存向量库 + 余弦相似度
│   ├── retrieval.go      # 向量 / BM25 / 混合 RRF
│   ├── rag.go            # LLM 重排 + 带引用生成
│   ├── retrieval_test.go # 离线 Recall 测试（哈希假向量，零成本）
│   └── cmd/              # --chunk / --recall / 问答
└── stage4/               # Agent 核心
    ├── tools/finance.go  # 4 个工具（计算器/增长率/指标/检索）
    ├── agent.go          # Eino ReAct Agent
    ├── acceptance.go     # 财报验收判定
    ├── mcp/server.go     # MCP Server（stdio）
    └── cmd/              # --tools 自测 / 真实 Agent
```

## 三、Go vs Python 概念对照（重点）

| Python（LangChain） | Go（Eino） | 备注 |
|---|---|---|
| `ChatOpenAI` | `openai.NewChatModel(ctx, cfg)` | 都走 OpenAI 兼容端点 |
| `SystemMessage / HumanMessage` | `schema.SystemMessage / UserMessage` | 消息抽象一致 |
| `llm.invoke()` | `cm.Generate(ctx, msgs)` | 同步调用 |
| `llm.stream()` | `cm.Stream(ctx, msgs)` → `StreamReader.Recv()` | 流式 |
| `response_metadata["token_usage"]` | `resp.ResponseMeta.Usage` | token 统计 |
| `Pydantic BaseModel` | `struct + json tag + Validate()` | Go 靠显式校验函数 |
| `with_structured_output()` | `WithTools()` + 强制调用提交工具 | Eino 无直接等价 API，用 function calling 承载 |
| `@tool` 装饰器 | `utils.InferTool(fn, struct)` | tag 自动生成 JSON Schema |
| `InMemoryVectorStore` | 自写 `InMemoryStore` | Eino 无内置内存库 |
| `BM25Retriever` | 自写 `BM25Retriever` | 都按中文单字+2-gram |
| `create_react_agent` | `react.NewAgent(ctx, cfg)` | MaxStep=护栏 |
| `LangGraph StateGraph` | `compose.NewGraph` | 本次未展开，留作进阶 |
| `with_structured_output` 的重试 | 手写 for 循环 + 错误反馈 | Go 版完全透明 |

## 四、怎么跑

所有命令在 `go-lab/` 目录下执行。**零成本命令不需要 API Key**。

```bash
cd src/ai-lab/go-lab

# 全量编译（逐包，避免 ./... 通配符卡顿）
for p in ./pkg/config ./stage1 ./stage1/experiments ./stage2 ./stage2/cmd \
         ./stage3 ./stage3/cmd ./stage4 ./stage4/cmd ./stage4/tools ./stage4/mcp; do
  go build "$p" || break
done

# 离线验证（零成本）
go test ./stage3 -v                        # BM25/混合检索 Recall + 分块
go run ./stage2/cmd --demo                 # Pydantic 式校验演示
go run ./stage3/cmd --chunk                # 分块对比
go run ./stage4/cmd --tools                # 工具自测 + 安全拦截

# 真实调用（需 stage1/.env 的 Key 有效）
go run ./stage1 "什么是 RAG？"
go run ./stage1 --stream "解释 RAG"
go run ./stage1/experiments --exp temperature
go run ./stage3/cmd --recall               # Recall@5（真实 embedding）
go run ./stage3/cmd --mode hybrid --rerank "防水等级是多少"
go run ./stage2/cmd --eval 20              # 格式正确率验收
go run ./stage4/cmd "分析 2024 年营收增长情况"
```

## 五、当前环境与已知状态

- **离线验证结果**：BM25 Recall@5 = 41/41（100%）；混合检索（哈希假向量）= 40/41（97.6%）；4 个工具调用与安全拦截全部通过。
- **API Key**：仓库 `stage1/.env` 的 `sk-ws-` Key 目前返回 401，所以真实模型调用（stage1/2/3/4 的在线功能）尚未实测——换好 Key 即可跑。
- **Go 工具链**：本机 go1.23.4 因 mcp-go 要求会自动切到 1.26.11（已安装）。若不想升级 mcp-go，可降到 support go1.23 的旧版本。
- **卡点经验**：`go build ./...` 在本机会偶发 SIGTERM，改用逐包 `go build <具体路径>` 稳定通过；`go run` 首次编译也可能超时，先 build 成 exe 再跑更稳。

## 六、与 Python 版的关系

- Python 版是**主线学习材料**（资料、教程最丰富），Go 版是**对照实现**（Eino 是 Go 生态生产首选）。
- 两份语料、验收指标、设计思路一致，方便逐段比对"同一个 RAG 在两种语言里怎么写"。
- 后续阶段 5-6（工程化上线）两个版本可以各自独立推进。
