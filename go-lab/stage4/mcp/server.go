// MCP Server：把 stage4 的财报工具暴露成标准 MCP 服务（stdio 传输）。
// 与 Python 版 mcp/server.py 等价，但用 mark3labs/mcp-go（你 agent-demo 已用的库）。
//
// 运行：go run ./stage4/mcp/server
// （stdio 模式下 stdout 是 JSON-RPC 协议流，日志必须走 stderr）
package main

import (
	"context"
	"encoding/json"
	"log"
	"os"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"golab/stage4/tools"
)

func main() {
	s := server.NewMCPServer("finance-report-tools", "1.0.0",
		server.WithToolCapabilities(true))

	s.AddTool(mcp.NewTool("calculate",
		mcp.WithDescription("安全算术计算器，支持 + - * / %，拒绝非算术字符"),
		mcp.WithString("expression", mcp.Description("算术表达式"), mcp.Required()),
	), calcHandler)

	s.AddTool(mcp.NewTool("growth_rate",
		mcp.WithDescription("计算增长率百分比"),
		mcp.WithNumber("current", mcp.Description("本期数值"), mcp.Required()),
		mcp.WithNumber("previous", mcp.Description("上期数值"), mcp.Required()),
	), growthHandler)

	s.AddTool(mcp.NewTool("get_metric",
		mcp.WithDescription("查询财报结构化指标"),
		mcp.WithNumber("year", mcp.Description("年份"), mcp.Required()),
		mcp.WithString("metric", mcp.Description("指标名"), mcp.Required()),
	), metricHandler)

	s.AddTool(mcp.NewTool("rag_search",
		mcp.WithDescription("检索财报文本段落"),
		mcp.WithString("query", mcp.Description("检索关键词"), mcp.Required()),
	), searchHandler)

	log.SetOutput(os.Stderr)
	log.Printf("finance-report-tools MCP 启动，工具数 4，transport=stdio")
	if err := server.ServeStdio(s); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

// 用 Eino 工具实现 handler：把 mcp 参数转成 JSON，调 InvokableRun。
func callTool(ctx context.Context, name string, args map[string]any) (*mcp.CallToolResult, error) {
	all, err := tools.BuildTools()
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	for _, t := range all {
		info, _ := t.Info(ctx)
		if info.Name != name {
			continue
		}
		b, _ := json.Marshal(args)
		out, err := t.InvokableRun(ctx, string(b))
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}
		return mcp.NewToolResultText(out), nil
	}
	return mcp.NewToolResultError("未知工具: " + name), nil
}

func calcHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	a := req.GetArguments()
	return callTool(ctx, "calculate", map[string]any{"expression": a["expression"]})
}

func growthHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	a := req.GetArguments()
	return callTool(ctx, "growth_rate", map[string]any{"current": a["current"], "previous": a["previous"]})
}

func metricHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	a := req.GetArguments()
	return callTool(ctx, "get_metric", map[string]any{"year": a["year"], "metric": a["metric"]})
}

func searchHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	a := req.GetArguments()
	return callTool(ctx, "rag_search", map[string]any{"query": a["query"]})
}
