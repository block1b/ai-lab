// stage1 实验：temperature 对比 + 流式输出，对标 Python 版 examples/02、03。
//
//   go run ./stage1/experiments --exp temperature
//   go run ./stage1/experiments --exp stream
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/cloudwego/eino/schema"
	"golab/pkg/config"
)

var exp = flag.String("exp", "temperature", "实验名：temperature / stream")

func main() {
	flag.Parse()
	ctx := context.Background()
	switch *exp {
	case "temperature":
		runTemperature(ctx)
	case "stream":
		runStream(ctx)
	default:
		fmt.Fprintln(os.Stderr, "未知实验:", *exp)
		os.Exit(1)
	}
}

const q = "请给「续航很久」想一句 15 字以内的手机广告词。"

func runTemperature(ctx context.Context) {
	fmt.Println("=== 实验：temperature 0 vs 1.5（各 3 次）===")
	for _, temp := range []float32{0, 1.5} {
		cm, err := config.NewChatModel(ctx, &temp)
		if err != nil {
			fmt.Fprintln(os.Stderr, "初始化失败:", err)
			return
		}
		fmt.Printf("\n[temperature=%.1f]\n", temp)
		for i := 1; i <= 3; i++ {
			resp, err := cm.Generate(ctx, []*schema.Message{
				schema.SystemMessage("你是广告文案助手。"),
				schema.UserMessage(q),
			})
			if err != nil {
				fmt.Printf("  第%d次失败: %v\n", i, err)
				continue
			}
			fmt.Printf("  第%d次: %s\n", i, resp.Content)
		}
	}
	fmt.Println("\n观察：temperature=0 三次输出高度一致；1.5 明显发散。")
}

func runStream(ctx context.Context) {
	fmt.Println("=== 实验：流式逐字输出 ===")
	temp := float32(0.7)
	cm, err := config.NewChatModel(ctx, &temp)
	if err != nil {
		fmt.Fprintln(os.Stderr, "初始化失败:", err)
		return
	}
	start := time.Now()
	sr, err := cm.Stream(ctx, []*schema.Message{
		schema.SystemMessage("你是技术助手，用中文回答。"),
		schema.UserMessage("用三句话解释什么是 RAG。"),
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "流式调用失败:", err)
		return
	}
	defer sr.Close()
	var first *time.Duration
	var last *schema.Message
	for {
		chunk, err := sr.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, "\n读取失败:", err)
			return
		}
		if first == nil {
			d := time.Since(start)
			first = &d
			fmt.Printf("\n[首字延迟] %.2fs\n", d.Seconds())
		}
		fmt.Print(chunk.Content)
		last = chunk
	}
	fmt.Println()
	if last != nil && last.ResponseMeta != nil && last.ResponseMeta.Usage != nil {
		u := last.ResponseMeta.Usage
		fmt.Printf("[token] 输入 %d / 输出 %d / 合计 %d\n",
			u.PromptTokens, u.CompletionTokens, u.TotalTokens)
	}
	fmt.Printf("[总耗时] %.2fs\n", time.Since(start).Seconds())
}
