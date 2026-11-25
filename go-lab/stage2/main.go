// stage2 CLI：评论抽取工具，对标 Python 版 cli.py。
//
//   go run ./stage2/cmd "屏幕很好但电池太差"   # 单次抽取（需 API）
//   go run ./stage2/cmd --demo                 # 本地 Schema 校验演示（零成本）
//   go run ./stage2/cmd --eval 20              # 批量验收（需 API）
package stage2

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"
)

var (
	flagDemo = flag.Bool("demo", false, "本地 Schema 校验演示，不调 API")
	flagEval = flag.Int("eval", 0, "批量验收运行次数，0 表示不评估")
)

func Main() {
	flag.Parse()
	if *flagDemo {
		demoValidate()
		return
	}
	if *flagEval > 0 {
		evaluate(*flagEval)
		return
	}
	args := flag.Args()
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "用法: go run ./stage2 \"评论文本\" | --demo | --eval N")
		os.Exit(1)
	}
	text := strings.Join(args, " ")
	ctx := context.Background()
	ex, err := NewExtractor(ctx, 3)
	if err != nil {
		fmt.Fprintln(os.Stderr, "初始化失败:", err)
		os.Exit(1)
	}
	started := time.Now()
	res, err := ex.Run(ctx, text)
	if err != nil {
		fmt.Fprintln(os.Stderr, "抽取失败:", err)
		os.Exit(1)
	}
	fmt.Println("抽取结果:")
	fmt.Println(Marshal(res.Data))
	fmt.Printf("尝试 %d 次 | token 输入 %d / 输出 %d | 耗时 %.2fs\n",
		res.Attempts, res.PromptTok, res.ComplTok, time.Since(started).Seconds())
}

// demoValidate 展示四类错误如何被本地校验拦截（对标 Python examples/01）。
func demoValidate() {
	fmt.Println("=== 本地 Schema 校验演示（零成本）===")
	cases := []struct{ name, raw string }{
		{"合法", `{"sentiment":"mixed","aspects":["屏幕","电池"],"evidence":"电池太差"}`},
		{"自造情感值", `{"sentiment":"还行吧","aspects":["屏幕"],"evidence":"还行"}`},
		{"类型错误", `{"sentiment":"positive","aspects":"屏幕, 电池","evidence":"很好"}`},
		{"缺字段", `{"sentiment":"positive","aspects":["屏幕"]}`},
		{"空证据", `{"sentiment":"negative","aspects":["电池"],"evidence":"  "}`},
	}
	for _, c := range cases {
		data, err := ParseOnly(c.raw)
		if err != nil {
			fmt.Printf("[%s] 拦截: %v\n", c.name, err)
		} else {
			fmt.Printf("[%s] 通过: %s\n", c.name, Marshal(data))
		}
	}
}

// evaluate 批量验收：统计格式正确率、首过率、token、耗时。
func evaluate(runs int) {
	fmt.Printf("=== 批量验收（%d 次，真实 API）===\n", runs)
	ctx := context.Background()
	ex, err := NewExtractor(ctx, 3)
	if err != nil {
		fmt.Fprintln(os.Stderr, "初始化失败:", err)
		return
	}
	texts := evalTexts()
	passed, firstTry := 0, 0
	var totalTok, totalPrompt, totalCompl int
	started := time.Now()
	for i := 0; i < runs; i++ {
		text := texts[i%len(texts)]
		res, err := ex.Run(ctx, text)
		if err != nil {
			fmt.Printf("  #%d 失败: %v\n", i+1, err)
			continue
		}
		passed++
		if res.Attempts == 1 {
			firstTry++
		}
		totalTok += res.Tokens
		totalPrompt += res.PromptTok
		totalCompl += res.ComplTok
		fmt.Printf("  #%d OK (%s) 尝试 %d 次\n", i+1, res.Data.Sentiment, res.Attempts)
	}
	rate := float64(passed) / float64(runs) * 100
	fmt.Println("==============================")
	fmt.Printf("格式正确率: %.1f%% (%d/%d)  验收线 98%%\n", rate, passed, runs)
	fmt.Printf("首过率     : %.1f%% (%d/%d)\n", float64(firstTry)/float64(runs)*100, firstTry, runs)
	fmt.Printf("累计 token : %d (输入 %d / 输出 %d)\n", totalTok, totalPrompt, totalCompl)
	fmt.Printf("单次均值   : %d token | 总耗时 %.2fs\n", totalTok/runs, time.Since(started).Seconds())
	if rate >= 98 {
		fmt.Println("结论: PASS ✅")
	} else {
		fmt.Println("结论: FAIL ❌ 需要调 Prompt 或加重试")
	}
}

func evalTexts() []string {
	return []string{
		"屏幕很好但电池太差，充电一次撑不过半天。",
		"物流很快，包装仔细，客服回复也及时，很满意。",
		"音质一般，价格偏贵，不推荐购买。",
		"外观漂亮，手感不错，就是系统偶尔卡顿。",
		"用了三天就坏了，质量太差。",
	}
}
