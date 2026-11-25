// stage2：结构化输出 Schema，对标 Python 版 schemas.py。
//
// Go 没有 Pydantic，这里用 struct + 手写 validate() 实现同等契约：
// 类型由 encoding/json 保证，值域/非空等约束由 validate() 保证。
package stage2

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// Sentiment 情感值域——用自定义类型锁定，杜绝模型自造值。
type Sentiment string

const (
	Positive Sentiment = "positive"
	Negative Sentiment = "negative"
	Neutral  Sentiment = "neutral"
	Mixed    Sentiment = "mixed"
)

// ReviewExtract 商品评论抽取结果，对应 Python 版 ReviewExtract。
type ReviewExtract struct {
	Sentiment Sentiment `json:"sentiment"`
	Aspects   []string  `json:"aspects"`
	Evidence  string    `json:"evidence"`
}

// Validate 执行与 Pydantic 等效的硬校验，返回所有违规项。
func (r ReviewExtract) Validate() error {
	var errs []string
	switch r.Sentiment {
	case Positive, Negative, Neutral, Mixed:
	default:
		errs = append(errs, fmt.Sprintf("sentiment=%q 不在值域 [positive/negative/neutral/mixed]", r.Sentiment))
	}
	if len(r.Aspects) == 0 {
		errs = append(errs, "aspects 不能为空数组")
	}
	if strings.TrimSpace(r.Evidence) == "" {
		errs = append(errs, "evidence 不能为空")
	}
	if len(errs) > 0 {
		return errors.New(strings.Join(errs, "; "))
	}
	return nil
}

// ParseReview 把模型返回的 JSON 字符串解析并校验，失败时返回可反馈给模型的错误。
func ParseReview(raw string) (ReviewExtract, error) {
	var out ReviewExtract
	if err := json.Unmarshal([]byte(strings.TrimSpace(raw)), &out); err != nil {
		return out, fmt.Errorf("JSON 解析失败: %w", err)
	}
	if err := out.Validate(); err != nil {
		return out, fmt.Errorf("Schema 校验失败: %w", err)
	}
	return out, nil
}

// SchemaJSON 返回给 Prompt 用的 Schema 描述（few-shot / 手动模式用）。
func SchemaJSON() string {
	return `{
  "sentiment": "positive | negative | neutral | mixed",
  "aspects": ["属性1", "属性2"],
  "evidence": "原文逐字引用的证据句"
}`
}
