// Agent 客户端：连接 MCP Server，把工具清单转成大模型的 function calling 定义，
// 然后跑「模型决策 → 调用工具 → 结果回填 → 再决策」的循环。
//
// 三种运行方式：
//
//	go run ./cmd/agent -mock          # 不依赖大模型，脚本化演示工具调用与护栏
//	go run ./cmd/agent -q "问题"       # 真实调用大模型（需配置 OPENAI_* 环境变量）
//
// 环境变量：
//
//	OPENAI_BASE_URL  默认 https://api.deepseek.com/v1（任何 OpenAI 兼容网关都行）
//	OPENAI_API_KEY   必填（真实模式）
//	OPENAI_MODEL     默认 deepseek-chat
//	MCP_SERVER_CMD   默认 bin/mcp-server（Windows 下自动补 .exe）
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"runtime"
	"strings"
	"sync/atomic"
	"time"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
)

const (
	maxRounds  = 6                // 单次会话最多几轮工具调用，防止死循环
	llmTimeout = 60 * time.Second // 单次模型请求超时
	connectTO  = 20 * time.Second
)

var traceCounter int64

type chatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type toolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type functionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

type toolDef struct {
	Type     string      `json:"type"`
	Function functionDef `json:"function"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Tools       []toolDef     `json:"tools,omitempty"`
	ToolChoice  string        `json:"tool_choice,omitempty"`
	Temperature float64       `json:"temperature"`
}

type chatResponse struct {
	Choices []struct {
		Message      chatMessage `json:"message"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func main() {
	mock := flag.Bool("mock", false, "不调用大模型，用脚本化流程演示工具调用")
	question := flag.String("q", "郑州大学有哪些设备离线？帮我查清原因。注意：不要擅自重启设备。", "用户问题")
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	c := mustConnect(ctx)
	defer c.Close()

	tools := mustListTools(ctx, c)
	toolsJSON := toOpenAITools(tools)
	fmt.Printf("已连接 MCP Server，发现 %d 个工具：\n", len(tools))
	for _, t := range tools {
		fmt.Printf("  - %s：%s\n", t.Name, t.Description)
	}
	fmt.Println()

	if *mock {
		runMock(ctx, c, *question)
		return
	}
	runLLM(ctx, c, toolsJSON, *question)
}

// ---------- MCP 连接 ----------

func mustConnect(ctx context.Context) *client.Client {
	cmd := os.Getenv("MCP_SERVER_CMD")
	if cmd == "" {
		cmd = "bin/mcp-server"
		if runtime.GOOS == "windows" {
			cmd += ".exe"
		}
	}
	parts := strings.Fields(cmd)

	t := transport.NewStdio(parts[0], nil, parts[1:]...)
	c := client.NewClient(t)

	// 注意：START/握手用的 ctx 不能 cancel，transport 会跟随 ctx 生命周期关闭
	if err := c.Start(ctx); err != nil {
		log.Fatalf("启动 MCP Server 失败（先执行 go build -o bin/mcp-server ./cmd/server）: %v", err)
	}

	initReq := mcp.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcp.Implementation{Name: "iot-ops-agent", Version: "1.0.0"}
	initReq.Params.Capabilities = mcp.ClientCapabilities{}
	if _, err := c.Initialize(ctx, initReq); err != nil {
		log.Fatalf("MCP 握手失败: %v", err)
	}
	return c
}

func mustListTools(ctx context.Context, c *client.Client) []mcp.Tool {
	res, err := c.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		log.Fatalf("获取工具列表失败: %v", err)
	}
	return res.Tools
}

// toOpenAITools 把 MCP 的 inputSchema 原样搬成 function calling 的 parameters，
// 不用手写一遍 schema，工具改了客户端无需改动。
func toOpenAITools(tools []mcp.Tool) []toolDef {
	out := make([]toolDef, 0, len(tools))
	for _, t := range tools {
		raw, err := json.Marshal(t.InputSchema)
		if err != nil || len(raw) == 0 || string(raw) == "null" {
			raw = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		out = append(out, toolDef{Type: "function", Function: functionDef{
			Name: t.Name, Description: t.Description, Parameters: raw,
		}})
	}
	return out
}

// ---------- 工具调用 ----------

func callTool(ctx context.Context, c *client.Client, name string, args map[string]any) string {
	trace := fmt.Sprintf("t-%d-%d", time.Now().Unix()%100000, atomic.AddInt64(&traceCounter, 1))
	start := time.Now()

	rawArgs, _ := json.Marshal(args)
	fmt.Printf("  → [%s] 调用 %s%s\n", trace, name, string(rawArgs))

	cctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	res, err := c.CallTool(cctx, mcp.CallToolRequest{Params: mcp.CallToolParams{Name: name, Arguments: args}})
	if err != nil {
		fmt.Printf("  ← [%s] 调用失败 %v（耗时 %dms）\n", trace, err, time.Since(start).Milliseconds())
		return "ERROR: " + err.Error()
	}
	text := resultText(res)
	fmt.Printf("  ← [%s] 返回 %d 字节，耗时 %dms\n", trace, len(text), time.Since(start).Milliseconds())
	return text
}

func resultText(r *mcp.CallToolResult) string {
	var b strings.Builder
	for _, item := range r.Content {
		if t, ok := item.(mcp.TextContent); ok {
			b.WriteString(t.Text)
		}
	}
	if r.IsError {
		return "ERROR: " + b.String()
	}
	return b.String()
}

// ---------- 真实大模型循环 ----------

func runLLM(ctx context.Context, c *client.Client, tools []toolDef, question string) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("未配置 OPENAI_API_KEY，改为运行 mock 流程（加 -mock 可直接指定）")
		runMock(ctx, c, question)
		return
	}

	msgs := []chatMessage{
		{Role: "system", Content: "你是物联网运维助手。回答前先用工具查证，不要臆测数据。" +
			"涉及重启等危险操作时，必须先向用户说明影响并取得确认，confirm=false 时不得重复尝试。" +
			"回答要给出依据（设备 ID、告警内容、时间）。"},
		{Role: "user", Content: question},
	}

	for round := 1; round <= maxRounds; round++ {
		resp, err := chat(ctx, msgs, tools)
		if err != nil {
			log.Printf("第 %d 轮模型调用失败: %v", round, err)
			return
		}
		choice := resp.Choices[0]
		msgs = append(msgs, choice.Message)

		if len(choice.Message.ToolCalls) == 0 {
			fmt.Printf("\n=== 最终回答（第 %d 轮，tokens=%d）===\n%s\n", round, resp.Usage.TotalTokens, choice.Message.Content)
			return
		}

		for _, tc := range choice.Message.ToolCalls {
			args := map[string]any{}
			if tc.Function.Arguments != "" {
				if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
					msgs = append(msgs, chatMessage{Role: "tool", ToolCallID: tc.ID, Content: "ERROR: 参数不是合法 JSON"})
					continue
				}
			}
			out := callTool(ctx, c, tc.Function.Name, args)
			msgs = append(msgs, chatMessage{Role: "tool", ToolCallID: tc.ID, Content: out})
		}
	}
	fmt.Printf("已达到最大轮次 %d，停止循环（防止无限调用）\n", maxRounds)
}

func chat(ctx context.Context, msgs []chatMessage, tools []toolDef) (*chatResponse, error) {
	base := os.Getenv("OPENAI_BASE_URL")
	if base == "" {
		base = "https://api.deepseek.com/v1"
	}
	model := os.Getenv("OPENAI_MODEL")
	if model == "" {
		model = "deepseek-chat"
	}

	body, _ := json.Marshal(chatRequest{
		Model: model, Messages: msgs, Tools: tools, ToolChoice: "auto", Temperature: 0.2,
	})

	var lastErr error
	for attempt := 1; attempt <= 2; attempt++ {
		cctx, cancel := context.WithTimeout(ctx, llmTimeout)
		req, _ := http.NewRequestWithContext(cctx, http.MethodPost,
			strings.TrimRight(base, "/")+"/chat/completions", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+os.Getenv("OPENAI_API_KEY"))

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			cancel()
			lastErr = err
			time.Sleep(time.Duration(attempt) * time.Second)
			continue
		}
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		cancel()

		if resp.StatusCode >= 500 || resp.StatusCode == 429 {
			lastErr = fmt.Errorf("上游返回 %d: %s", resp.StatusCode, string(raw))
			time.Sleep(time.Duration(attempt) * time.Second)
			continue
		}
		var out chatResponse
		if err := json.Unmarshal(raw, &out); err != nil {
			return nil, fmt.Errorf("解析响应失败: %w", err)
		}
		if out.Error != nil {
			return nil, fmt.Errorf("模型返回错误: %s", out.Error.Message)
		}
		if len(out.Choices) == 0 {
			return nil, fmt.Errorf("响应为空: %s", string(raw))
		}
		return &out, nil
	}
	return nil, lastErr
}

// ---------- mock 流程（无 API Key 也能演示）----------

func runMock(ctx context.Context, c *client.Client, question string) {
	fmt.Printf("用户问题：%s\n\n", question)
	fmt.Println("模拟模型决策过程：")

	steps := []struct {
		name string
		args map[string]any
		note string
	}{
		{"list_devices", map[string]any{"tenant": "zzu", "status": "offline"}, "先确认离线设备清单"},
		{"query_alarms", map[string]any{"tenant": "zzu", "since_minutes": 120, "level": "critical"}, "再查告警定位原因"},
		{"restart_device", map[string]any{"device_id": "dev-1002", "tenant": "zzu", "confirm": false}, "未确认时工具拒绝执行（护栏）"},
		{"restart_device", map[string]any{"device_id": "dev-1002", "tenant": "zzu", "confirm": true}, "获得确认后真正执行"},
	}

	var lastList, lastAlarms, lastRestart string
	for i, s := range steps {
		fmt.Printf("\n第 %d 步 · %s\n", i+1, s.note)
		out := callTool(ctx, c, s.name, s.args)
		switch s.name {
		case "list_devices":
			lastList = out
		case "query_alarms":
			lastAlarms = out
		default:
			lastRestart = out
		}
		fmt.Printf("     结果：%s\n", truncate(out, 260))
	}

	fmt.Println("\n=== 汇总（mock 模式下由固定模板拼装）===")
	fmt.Printf("离线设备：%s\n", truncate(lastList, 200))
	fmt.Printf("关联告警：%s\n", truncate(lastAlarms, 200))
	fmt.Printf("重启结果：%s\n", truncate(lastRestart, 200))
	fmt.Println("\n说明：以上四步即真实的「模型决策 → 工具执行 → 结果回填」链路，")
	fmt.Println("      配置 OPENAI_API_KEY 后模型会自行决定调用顺序与参数。")
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "..."
}
