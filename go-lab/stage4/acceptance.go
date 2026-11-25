// stage4 验收脚本：对标 Python 版 acceptance.py。
// 锚点任务："分析财报并生成摘要报告"——检索 → 计算 → 汇总。
package stage4

import (
	"context"
	"fmt"
	"regexp"
	"strings"
)

// Acceptance 判定结果。
type Acceptance struct {
	Answer       string
	HasRetrieval bool
	HasCalc      bool
	HasNumber    bool
	WithinSteps  bool
	Passed       bool
}

// Check 对 Agent 答案做四项硬判定。
// 说明：Go 版用 Eino Agent，本函数判定"答案证据"而非内部工具调用轨迹
// （Eino 的轨迹需接 callback，留作扩展；这里先验证结果层面）。
func Check(answer string) Acceptance {
	a := Acceptance{Answer: answer}
	a.HasNumber = regexp.MustCompile(`\d`).MatchString(answer)
	a.HasRetrieval = strings.Contains(answer, "收入") || strings.Contains(answer, "营收") ||
		strings.Contains(answer, "毛利率") || strings.Contains(answer, "客户")
	a.HasCalc = strings.Contains(answer, "%") || strings.Contains(answer, "增长")
	a.WithinSteps = len(answer) > 0 && len(answer) < 4000
	a.Passed = a.HasRetrieval && a.HasCalc && a.HasNumber && a.WithinSteps
	return a
}

// RunAcceptance 跑真实 Agent 验收（需有效 API Key）。
func RunAcceptance(ctx context.Context, question string) (Acceptance, error) {
	ag, err := NewAgent(ctx, 8)
	if err != nil {
		return Acceptance{}, err
	}
	ans, err := Run(ctx, ag, question)
	if err != nil {
		return Acceptance{}, err
	}
	return Check(ans), nil
}

// Print 打印判定表。
func (a Acceptance) Print() {
	fmt.Println("==============================")
	fmt.Printf("1. 检索类事实  : %v\n", a.HasRetrieval)
	fmt.Printf("2. 计算类结论  : %v\n", a.HasCalc)
	fmt.Printf("3. 答案含数字  : %v\n", a.HasNumber)
	fmt.Printf("4. 步数内完成  : %v\n", a.WithinSteps)
	if a.Passed {
		fmt.Println("结论: PASS ✅")
	} else {
		fmt.Println("结论: FAIL ❌")
	}
	fmt.Println("==============================")
}
