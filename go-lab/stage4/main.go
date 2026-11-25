// stage4 CLI + 工具自测。
//
//   go run ./stage4/cmd --tools                    # 工具层自测（零成本）
//   go run ./stage4/cmd --check "答案文本..."       # 验收判定演示（零成本）
//   go run ./stage4/cmd "分析2024年营收增长情况"    # 真实 Agent（需 API）
package stage4

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"golab/stage4/tools"
)

var (
	flagTools = flag.Bool("tools", false, "工具层自测，零成本")
	flagCheck = flag.String("check", "", "验收判定演示，传入答案文本")
)

// Main 是 stage4 CLI 入口。
func Main() {
	flag.Parse()
	if *flagTools {
		toolsSelfTest()
		return
	}
	if *flagCheck != "" {
		Check(*flagCheck).Print()
		return
	}
	args := flag.Args()
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "用法: go run ./stage4/cmd \"问题\" | --tools | --check \"答案\"")
		os.Exit(1)
	}
	ctx := context.Background()
	ag, err := NewAgent(ctx, 8)
	if err != nil {
		fmt.Fprintln(os.Stderr, "创建 Agent 失败:", err)
		os.Exit(1)
	}
	ans, err := Run(ctx, ag, strings.Join(args, " "))
	if err != nil {
		fmt.Fprintln(os.Stderr, "Agent 运行失败:", err)
		os.Exit(1)
	}
	fmt.Println("Agent 答案:\n" + ans)
	Check(ans).Print()
}

func toolsSelfTest() {
	fmt.Println("=== 工具层自测（零成本）===")
	ctx := context.Background()
	all, err := tools.BuildTools()
	if err != nil {
		fmt.Fprintln(os.Stderr, "构建工具失败:", err)
		return
	}
	fmt.Printf("工具数: %d\n", len(all))
	for _, t := range all {
		info, _ := t.Info(ctx)
		fmt.Printf("  - %s: %s\n", info.Name, truncRunes(info.Desc, 40))
	}

	// 通过 Invoke 直接验证各工具（对标 Python 工具自测）
	cases := []struct {
		name string
		args string
	}{
		{"calculate", `{"expression":"100*1.28"}`},
		{"growth_rate", `{"current":12.8,"previous":10.0}`},
		{"get_metric", `{"year":2024,"metric":"revenue"}`},
		{"rag_search", `{"query":"风险提示 客户集中度"}`},
	}
	for _, c := range cases {
		for _, t := range all {
			info, _ := t.Info(ctx)
			if info.Name != c.name {
				continue
			}
			out, err := t.InvokableRun(ctx, c.args)
			if err != nil {
				fmt.Printf("[%s] 失败: %v\n", c.name, err)
			} else {
				fmt.Printf("[%s] %s\n", c.name, out)
			}
		}
	}

	// 安全测试：恶意表达式应被拦截
	for _, t := range all {
		info, _ := t.Info(ctx)
		if info.Name == "calculate" {
			out, err := t.InvokableRun(ctx, `{"expression":"__import__(\"os\")"}`)
			fmt.Printf("[安全] 恶意表达式: err=%v out=%s\n", err, out)
		}
	}
}

func truncRunes(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "..."
}
