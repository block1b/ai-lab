// stage1：用 Eino 调千问（OpenAI 兼容端点），对标 Python 版 stage1。
//
// 运行（在项目根 go-lab 下）：
//   go run ./stage1 "什么是 RAG？"        # 单次问答，打印 token 用量
//   go run ./stage1                      # 交互模式，输入 exit 退出
//   go run ./stage1 --temp 0 "..."       # 指定温度
//   go run ./stage1 --stream "..."       # 流式输出
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"golab/pkg/config"
)

var (
	flagTemp   = flag.Float64("temp", 0, "采样温度，0 最确定")
	flagStream = flag.Bool("stream", false, "流式输出")
	flagSystem = flag.String("system", "你是一个简洁的技术助手，用中文回答，控制在 100 字以内。", "系统提示词")
)

func main() {
	flag.Parse()

	ctx := context.Background()
	info := config.Info()
	fmt.Printf("[配置] model=%s | base=%s | key=%s\n", info.Model, info.BaseURL, info.KeyPrefix)

	temp := float32(*flagTemp)
	cm, err := config.NewChatModel(ctx, &temp)
	if err != nil {
		fmt.Fprintln(os.Stderr, "初始化模型失败:", err)
		os.Exit(1)
	}

	args := flag.Args()
	if len(args) == 0 {
		interactive(ctx, cm, *flagSystem, *flagStream)
		return
	}
	ask(ctx, cm, *flagSystem, strings.Join(args, " "), *flagStream)
}

func interactive(ctx context.Context, cm model.BaseChatModel, system string, stream bool) {
	fmt.Println("进入交互模式（输入 exit / quit 退出）:")
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			return
		}
		q := strings.TrimSpace(scanner.Text())
		if q == "exit" || q == "quit" {
			fmt.Println("再见")
			return
		}
		if q != "" {
			ask(ctx, cm, system, q, stream)
		}
	}
}

func ask(ctx context.Context, cm model.BaseChatModel, system, question string, stream bool) {
	messages := []*schema.Message{
		schema.SystemMessage(system),
		schema.UserMessage(question),
	}
	started := time.Now()

	if stream {
		sr, err := cm.Stream(ctx, messages)
		if err != nil {
			fmt.Fprintln(os.Stderr, "流式调用失败:", err)
			return
		}
		defer sr.Close()
		fmt.Print("Assistant: ")
		var last *schema.Message
		for {
			chunk, err := sr.Recv()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				fmt.Fprintln(os.Stderr, "\n流式读取失败:", err)
				return
			}
			fmt.Print(chunk.Content)
			last = chunk
		}
		fmt.Println()
		printUsage(last, started)
		return
	}

	resp, err := cm.Generate(ctx, messages)
	if err != nil {
		fmt.Fprintln(os.Stderr, "调用失败:", err)
		return
	}
	fmt.Println("Assistant:", resp.Content)
	printUsage(resp, started)
}

func printUsage(msg *schema.Message, started time.Time) {
	if msg != nil && msg.ResponseMeta != nil && msg.ResponseMeta.Usage != nil {
		u := msg.ResponseMeta.Usage
		fmt.Printf("[token] 输入 %d / 输出 %d / 合计 %d\n",
			u.PromptTokens, u.CompletionTokens, u.TotalTokens)
	}
	fmt.Printf("[耗时] %.2fs\n", time.Since(started).Seconds())
}
