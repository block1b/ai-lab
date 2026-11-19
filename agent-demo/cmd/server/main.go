// MCP Server：把设备运维能力暴露成标准工具，供任意支持 MCP 的客户端（Claude Desktop、
// 自研 Agent、IDE 插件等）调用。
//
// 注意：stdio 模式下 stdout 是 JSON-RPC 协议流，日志必须走 stderr，否则会污染协议。
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"agentdemo/internal/device"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

var store = device.NewStore()

func main() {
	s := server.NewMCPServer(
		"iot-ops-mcp",
		"1.0.0",
		server.WithToolCapabilities(true),
	)

	s.AddTool(mcp.NewTool("list_devices",
		mcp.WithDescription("按租户和状态查询物联网设备列表，返回设备 ID、名称、型号、IP、在线状态与最后心跳时间"),
		mcp.WithString("tenant",
			mcp.Description("租户标识，例如 zzu（郑州大学）、haut（河南工业大学）"),
			mcp.Required(),
		),
		mcp.WithString("status",
			mcp.Description("设备状态过滤：online / offline / rebooting，留空表示全部"),
		),
	), handleListDevices)

	s.AddTool(mcp.NewTool("query_alarms",
		mcp.WithDescription("查询指定时间范围内的设备告警，按时间倒序返回；用于定位设备异常原因"),
		mcp.WithString("tenant",
			mcp.Description("租户标识"),
			mcp.Required(),
		),
		mcp.WithNumber("since_minutes",
			mcp.Description("查询最近多少分钟内的告警，默认 60"),
			mcp.DefaultNumber(60),
		),
		mcp.WithString("level",
			mcp.Description("告警级别过滤：info / warn / critical，留空表示全部"),
		),
	), handleQueryAlarms)

	s.AddTool(mcp.NewTool("restart_device",
		mcp.WithDescription("重启指定设备。这是危险操作：必须先把设备 ID 和影响告知用户并获得明确确认，confirm 才能置为 true"),
		mcp.WithString("device_id",
			mcp.Description("设备 ID，例如 dev-1002"),
			mcp.Required(),
		),
		mcp.WithString("tenant",
			mcp.Description("租户标识，用于校验设备归属，防止越权操作"),
			mcp.Required(),
		),
		mcp.WithBoolean("confirm",
			mcp.Description("是否已获得用户明确确认；false 时工具只做校验并返回待确认提示，不会真正执行"),
			mcp.Required(),
		),
	), handleRestartDevice)

	log.SetOutput(os.Stderr)
	log.Printf("iot-ops-mcp 启动，工具数 3，transport=stdio")
	if err := server.ServeStdio(s); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

func handleListDevices(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := req.GetArguments()
	tenant, _ := args["tenant"].(string)
	if tenant == "" {
		return mcp.NewToolResultError("tenant 不能为空"), nil
	}
	status, _ := args["status"].(string)
	if status != "" && status != "online" && status != "offline" && status != "rebooting" {
		return mcp.NewToolResultError("status 只能是 online / offline / rebooting"), nil
	}

	list := store.ListDevices(tenant, status)
	payload := map[string]any{"total": len(list), "devices": list}
	return jsonResult(payload)
}

func handleQueryAlarms(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := req.GetArguments()
	tenant, _ := args["tenant"].(string)
	if tenant == "" {
		return mcp.NewToolResultError("tenant 不能为空"), nil
	}
	minutes := 60.0
	if v, ok := args["since_minutes"].(float64); ok && v > 0 {
		minutes = v
	}
	level, _ := args["level"].(string)

	since := time.Now().Add(-time.Duration(minutes) * time.Minute)
	alarms := store.QueryAlarms(tenant, since, level)
	payload := map[string]any{
		"total":         len(alarms),
		"since_minutes": int(minutes),
		"alarms":        alarms,
	}
	return jsonResult(payload)
}

func handleRestartDevice(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args := req.GetArguments()
	deviceID, _ := args["device_id"].(string)
	tenant, _ := args["tenant"].(string)
	confirm, _ := args["confirm"].(bool)

	if deviceID == "" || tenant == "" {
		return mcp.NewToolResultError("device_id 与 tenant 均为必填"), nil
	}

	// 护栏一：未确认不执行，只返回影响面，让模型回去问用户
	if !confirm {
		return jsonResult(map[string]any{
			"executed": false,
			"reason":   "缺少用户确认",
			"hint":     "请向用户说明将重启设备 " + deviceID + " 并请求确认，确认后再以 confirm=true 调用",
		})
	}

	// 护栏二：校验设备归属，防止跨租户越权
	d, err := store.Restart(deviceID, tenant)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("设备 %s 不存在或不属于租户 %s", deviceID, tenant)), nil
	}

	log.Printf("tool=restart_device device=%s tenant=%s status=%s", d.ID, d.Tenant, d.Status)
	return jsonResult(map[string]any{
		"executed": true,
		"device":   d,
		"message":  "重启指令已下发，预计 30 秒内恢复在线",
	})
}

// jsonResult 把结构化结果同时写入 content 与 structuredContent，
// 模型读文本、程序读结构，两边都不丢信息。
func jsonResult(v any) (*mcp.CallToolResult, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return mcp.NewToolResultError("结果序列化失败: " + err.Error()), nil
	}
	return &mcp.CallToolResult{
		Content:           []mcp.Content{mcp.TextContent{Type: "text", Text: string(b)}},
		StructuredContent: v,
	}, nil
}
