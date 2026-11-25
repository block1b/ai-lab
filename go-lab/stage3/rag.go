// stage3 环节④⑤：LLM 重排 + 带引用生成，对标 Python 版 rag.py / rerank.py。
package stage3

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/cloudwego/eino/schema"
	"golab/pkg/config"
)

const rerankSystem = `你是检索结果评审员。对每条候选片段与问题的相关性打分（0-10 分）。
严格输出 JSON 数组，格式 [{"id":"块ID","score":8}]，不要任何额外文字。`

const answerSystem = `你是企业知识库问答助手。规则：
1. 只依据【参考资料】回答，禁止使用外部知识或编造。
2. 每个事实后标注引用编号，如 [1]、[2]。
3. 若参考资料不足以回答问题，直接回答"文档中未提及"，不要猜测。
4. 用中文、简洁、分点回答。`

// Rerank 用 LLM-as-judge 对 Top-K 逐条打分精排。
func Rerank(ctx context.Context, query string, candidates []Scored, top int) ([]Scored, error) {
	temp := float32(0)
	cm, err := config.NewChatModel(ctx, &temp)
	if err != nil {
		return nil, err
	}
	var b strings.Builder
	b.WriteString("【问题】\n" + query + "\n\n【候选片段】\n")
	for i, c := range candidates {
		fmt.Fprintf(&b, "[%d] id=%s\n%s\n\n", i+1, c.ID, c.Text)
	}
	resp, err := cm.Generate(ctx, []*schema.Message{
		schema.SystemMessage(rerankSystem),
		schema.UserMessage(b.String()),
	})
	if err != nil {
		return nil, err
	}
	// 解析 JSON 数组
	var rates []struct {
		ID    string  `json:"id"`
		Score float64 `json:"score"`
	}
	raw := strings.TrimSpace(resp.Content)
	raw = strings.TrimPrefix(raw, "```json")
	raw = strings.TrimSuffix(raw, "```")
	if err := json.Unmarshal([]byte(strings.TrimSpace(raw)), &rates); err != nil {
		// 解析失败时退回原顺序，不让整条链崩
		if top > 0 && top < len(candidates) {
			return candidates[:top], nil
		}
		return candidates, nil
	}
	scoreMap := map[string]float64{}
	for _, r := range rates {
		scoreMap[r.ID] = r.Score
	}
	out := make([]Scored, 0, len(candidates))
	for _, c := range candidates {
		c.Score = scoreMap[c.ID]
		out = append(out, c)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if top > 0 && top < len(out) {
		return out[:top], nil
	}
	return out, nil
}

// Answer 带引用生成，返回答案文本。
func Answer(ctx context.Context, query string, docs []Scored) (string, error) {
	temp := float32(0)
	cm, err := config.NewChatModel(ctx, &temp)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString("【参考资料】\n")
	for i, d := range docs {
		fmt.Fprintf(&b, "[%d] 来源：%s\n%s\n\n", i+1, d.Source, d.Text)
	}
	b.WriteString("【问题】\n" + query)
	resp, err := cm.Generate(ctx, []*schema.Message{
		schema.SystemMessage(answerSystem),
		schema.UserMessage(b.String()),
	})
	if err != nil {
		return "", err
	}
	return resp.Content, nil
}

// IsRefusal 判断是否拒答（评估幻觉率用）。
func IsRefusal(answer string) bool {
	a := strings.TrimSpace(answer)
	return strings.Contains(a, "未提及") || strings.Contains(a, "没有相关信息") ||
		strings.Contains(a, "无法回答") || strings.Contains(a, "文档中未")
}

// CitationCount 统计答案里的引用编号个数。
func CitationCount(answer string) int {
	n := 0
	for i := 0; i < len(answer); i++ {
		if answer[i] == '[' {
			if j := strings.IndexByte(answer[i:], ']'); j > 1 {
				if _, err := strconv.Atoi(answer[i+1 : i+j]); err == nil {
					n++
				}
			}
		}
	}
	return n
}
