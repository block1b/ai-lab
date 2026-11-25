// stage4 工具层：对标 Python 版 tools/finance_tools.py。
// 用 Go struct + jsonschema tag 定义参数（Eino utils.InferTool 自动转 JSON Schema），
// 这是 Go 生态里 Pydantic Schema 的对应物。
package tools

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
)

// CalcInput 计算器参数；tag 约束值域与描述。
type CalcInput struct {
	Expression string `json:"expression" jsonschema:"required,description=算术表达式，支持 + - * / %，用于算增长率和利润率"`
}

// GrowthInput 增长率参数。
type GrowthInput struct {
	Current  float64 `json:"current" jsonschema:"required,description=本期数值"`
	Previous float64 `json:"previous" jsonschema:"required,description=上期数值"`
}

// MetricInput 财务指标查询参数。
type MetricInput struct {
	Year   int    `json:"year" jsonschema:"required,description=年份，如 2024"`
	Metric string `json:"metric" jsonschema:"required,description=指标名：revenue/net_profit/gross_margin/rd_spend/overseas_ratio"`
}

// SearchInput 财报文本检索参数。
type SearchInput struct {
	Query string `json:"query" jsonschema:"required,description=检索关键词，如 风险提示 客户集中度"`
}

// 虚构公司财报数据（与 Python 版一致）。
var financials = map[int]map[string]float64{
	2024: {"revenue": 12.8, "net_profit": 1.9, "gross_margin": 45.2, "rd_spend": 1.54, "overseas_ratio": 32.0},
	2023: {"revenue": 10.0, "net_profit": 1.4, "gross_margin": 42.0, "rd_spend": 1.2, "overseas_ratio": 28.0},
}

var corpus = []string{
	"公司 2024 年营业收入 12.8 亿元，同比增长 28%。净利润 1.9 亿元。",
	"2024 年毛利率为 45.2%，较上年提升 3.2 个百分点。研发投入占营业收入的 12%。",
	"公司面临供应链成本波动风险、海外市场汇率风险以及行业竞争加剧风险。前五大客户收入占比为 45%，客户集中度较高。原材料价格上涨可能影响毛利率。",
	"海外收入占比为 32%，汇率波动对利润有一定影响。未来目标将海外收入占比提升至 45%。",
	"管理层预计未来三年营业收入复合增长率可达 20%，但该预测基于市场环境稳定的假设，不构成业绩承诺。",
}

// BuildTools 用 Eino InferTool 从 Go struct tag 推断 JSON Schema，创建全部 4 个工具。
// 这是 Go 生态里 Pydantic Schema 的对应物：tag 描述参数，框架自动生成 Schema。
func BuildTools() ([]tool.InvokableTool, error) {
	calcTool, err := utils.InferTool("calculate",
		"安全算术计算器。支持 + - * / %，用于算增长率、利润率。拒绝任何非算术字符。", calc)
	if err != nil {
		return nil, err
	}
	growthTool, err := utils.InferTool("growth_rate",
		"计算增长率百分比：(当前值-上期值)/上期值×100%。", growth)
	if err != nil {
		return nil, err
	}
	metricTool, err := utils.InferTool("get_metric",
		"查询虚构公司财报的结构化指标。metric: revenue/net_profit/gross_margin/rd_spend/overseas_ratio。", metric)
	if err != nil {
		return nil, err
	}
	searchTool, err := utils.InferTool("rag_search",
		"检索财报文本段落。适合查风险提示、战略展望、客户集中度等文本事实。", search)
	if err != nil {
		return nil, err
	}
	return []tool.InvokableTool{calcTool, growthTool, metricTool, searchTool}, nil
}

// --- 工具函数实现（签名供 InferTool 使用）---

func calc(_ context.Context, in *CalcInput) (string, error) {
	expr := strings.ReplaceAll(in.Expression, " ", "")
	if expr == "" {
		return "", fmt.Errorf("表达式不能为空")
	}
	// 极简安全计算：仅允许数字与 + - * / % ( ) .，拒绝字母（挡住 __import__ 等）
	for _, r := range expr {
		if !((r >= '0' && r <= '9') || strings.ContainsRune("+-*/%().", r)) {
			return "", fmt.Errorf("表达式含非法字符 %q，仅支持算术", r)
		}
	}
	v, err := evalSimple(expr)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%s = %g", in.Expression, v), nil
}

func growth(_ context.Context, in *GrowthInput) (string, error) {
	if in.Previous == 0 {
		return "", fmt.Errorf("上期数值不能为 0")
	}
	r := (in.Current - in.Previous) / in.Previous * 100
	return fmt.Sprintf("增长率 = (%g - %g) / %g × 100%% = %.1f%%", in.Current, in.Previous, in.Previous, r), nil
}

func metric(_ context.Context, in *MetricInput) (string, error) {
	y, ok := financials[in.Year]
	if !ok {
		return fmt.Sprintf("%d 年财务数据未收录", in.Year), nil
	}
	v, ok := y[in.Metric]
	if !ok {
		return fmt.Sprintf("指标 %s 未收录，可用：revenue/net_profit/gross_margin/rd_spend/overseas_ratio", in.Metric), nil
	}
	return fmt.Sprintf("%d 年 %s = %g", in.Year, in.Metric, v), nil
}

func search(_ context.Context, in *SearchInput) (string, error) {
	// 复用 stage3 的轻量关键词检索思路：2-gram 重叠计数
	q := bigramSet(in.Query)
	type hit struct {
		score int
		text  string
	}
	var hits []hit
	for _, text := range corpus {
		t := bigramSet(text)
		n := 0
		for g := range q {
			if t[g] {
				n++
			}
		}
		if n > 0 {
			hits = append(hits, hit{n, text})
		}
	}
	if len(hits) == 0 {
		return "财报中未检索到相关内容", nil
	}
	for i := 0; i < len(hits); i++ {
		for j := i + 1; j < len(hits); j++ {
			if hits[j].score > hits[i].score {
				hits[i], hits[j] = hits[j], hits[i]
			}
		}
	}
	out := make([]string, 0, 3)
	for i := 0; i < 3 && i < len(hits); i++ {
		out = append(out, fmt.Sprintf("[%d] %s", i+1, hits[i].text))
	}
	return strings.Join(out, "\n\n"), nil
}

func bigramSet(s string) map[string]bool {
	out := map[string]bool{}
	var runes []rune
	for _, r := range strings.ToLower(s) {
		if r >= '\u4e00' && r <= '\u9fff' || r >= '0' && r <= '9' {
			runes = append(runes, r)
		}
	}
	for i := 0; i+1 < len(runes); i++ {
		out[string(runes[i:i+2])] = true
	}
	return out
}

// evalSimple 支持 + - * / %，用简单递归下降求值（避免 eval 的安全问题）。
func evalSimple(expr string) (float64, error) {
	p := &exprParser{s: expr}
	v, err := p.expr()
	if err != nil {
		return 0, err
	}
	if p.pos != len(p.s) {
		return 0, fmt.Errorf("表达式解析错误，剩余: %s", p.s[p.pos:])
	}
	return v, nil
}

type exprParser struct {
	s   string
	pos int
}

func (p *exprParser) expr() (float64, error) {
	v, err := p.term()
	if err != nil {
		return 0, err
	}
	for p.pos < len(p.s) && (p.s[p.pos] == '+' || p.s[p.pos] == '-') {
		op := p.s[p.pos]
		p.pos++
		r, err := p.term()
		if err != nil {
			return 0, err
		}
		if op == '+' {
			v += r
		} else {
			v -= r
		}
	}
	return v, nil
}

func (p *exprParser) term() (float64, error) {
	v, err := p.factor()
	if err != nil {
		return 0, err
	}
	for p.pos < len(p.s) && (p.s[p.pos] == '*' || p.s[p.pos] == '/' || p.s[p.pos] == '%') {
		op := p.s[p.pos]
		p.pos++
		r, err := p.factor()
		if err != nil {
			return 0, err
		}
		switch op {
		case '*':
			v *= r
		case '/':
			if r == 0 {
				return 0, fmt.Errorf("除数不能为 0")
			}
			v /= r
		case '%':
			if r == 0 {
				return 0, fmt.Errorf("模数不能为 0")
			}
			v = float64(int(v) % int(r))
		}
	}
	return v, nil
}

func (p *exprParser) factor() (float64, error) {
	if p.pos >= len(p.s) {
		return 0, fmt.Errorf("表达式不完整")
	}
	if p.s[p.pos] == '(' {
		p.pos++
		v, err := p.expr()
		if err != nil {
			return 0, err
		}
		if p.pos >= len(p.s) || p.s[p.pos] != ')' {
			return 0, fmt.Errorf("缺少右括号")
		}
		p.pos++
		return v, nil
	}
	start := p.pos
	for p.pos < len(p.s) && ((p.s[p.pos] >= '0' && p.s[p.pos] <= '9') || p.s[p.pos] == '.') {
		p.pos++
	}
	if start == p.pos {
		return 0, fmt.Errorf("期望数字，实际: %c", p.s[p.pos])
	}
	v, err := strconv.ParseFloat(p.s[start:p.pos], 64)
	if err != nil {
		return 0, err
	}
	return v, nil
}
