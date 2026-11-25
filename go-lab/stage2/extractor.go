// stage2：结构化抽取器，对标 Python 版 extractor.py。
//
// Eino 没有 with_structured_output，这里用 function calling 承载结构化输出：
// 定义一个 submit_review 工具，其参数就是输出 Schema，模型必须调用它。
// 这比 json_schema response_format 兼容性更好（千问兼容端点普遍支持 tool call）。
package stage2

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"time"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"golab/pkg/config"
)

// ExtractResult 抽取结果 + 元信息。
type ExtractResult struct {
	Data      ReviewExtract
	Attempts  int
	Tokens    int
	PromptTok int
	ComplTok  int
}

// ExtractError 携带尝试次数，便于评估脚本统计。
type ExtractError struct {
	Message  string
	Attempts int
}

func (e *ExtractError) Error() string { return e.Message }

// Extractor 用 tool calling 做结构化输出 + 校验失败反馈重试。
type Extractor struct {
	base    model.BaseChatModel
	withTool model.ToolCallingChatModel
	maxRetry int
}

// NewExtractor 创建抽取器；temperature 固定 0（抽取任务要确定性）。
func NewExtractor(ctx context.Context, maxRetry int) (*Extractor, error) {
	temp := float32(0)
	cm, err := config.NewChatModel(ctx, &temp)
	if err != nil {
		return nil, err
	}
	tc, ok := cm.(model.ToolCallingChatModel)
	if !ok {
		return nil, errors.New("当前模型不支持 ToolCallingChatModel，无法用工具承载结构化输出")
	}
	withTool, err := tc.WithTools([]*schema.ToolInfo{{
		Name: "submit_review",
		Desc: "提交商品评论的抽取结果。必须调用此工具，不要在正文输出 JSON。",
		ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
			"sentiment": {
				Type: schema.String, Required: true,
				Enum: []string{"positive", "negative", "neutral", "mixed"},
				Desc: "情感极性，只能取枚举值",
			},
			"aspects": {
				Type: schema.Array, Required: true,
				ElemInfo: &schema.ParameterInfo{Type: schema.String},
				Desc: "评论涉及的属性列表，至少一项",
			},
			"evidence": {
				Type: schema.String, Required: true,
				Desc: "从原文逐字引用的证据句",
			},
		}),
	}})
	if err != nil {
		return nil, fmt.Errorf("绑定工具失败: %w", err)
	}
	if maxRetry < 0 {
		maxRetry = 3
	}
	return &Extractor{base: cm, withTool: withTool, maxRetry: maxRetry}, nil
}

// Run 执行抽取：校验失败时把错误反馈给模型，指数退避重试。
func (e *Extractor) Run(ctx context.Context, text string) (*ExtractResult, error) {
	var lastFeedback string
	var totalPrompt, totalCompl int

	for attempt := 0; attempt <= e.maxRetry; attempt++ {
		if attempt > 0 {
			time.Sleep(backoff(attempt-1))
		}
		resp, err := e.withTool.Generate(ctx, []*schema.Message{
			schema.SystemMessage(SYSTEM_PROMPT),
			schema.UserMessage(BuildUserPrompt(text, lastFeedback)),
		})
		if err != nil {
			return nil, &ExtractError{Message: fmt.Sprintf("LLM 调用失败: %v", err), Attempts: attempt + 1}
		}
		// 统计本次 token（即便后面校验失败，这次也计费）
		if resp.ResponseMeta != nil && resp.ResponseMeta.Usage != nil {
			totalPrompt += resp.ResponseMeta.Usage.PromptTokens
			totalCompl += resp.ResponseMeta.Usage.CompletionTokens
		}
		if len(resp.ToolCalls) == 0 {
			lastFeedback = "模型没有调用 submit_review 工具。请必须调用该工具提交。"
			continue
		}
		raw := resp.ToolCalls[0].Function.Arguments
		data, err := ParseReview(raw)
		if err != nil {
			lastFeedback = err.Error() + " 原始提交: " + raw
			continue
		}
		return &ExtractResult{
			Data: data, Attempts: attempt + 1,
			Tokens: totalPrompt + totalCompl,
			PromptTok: totalPrompt, ComplTok: totalCompl,
		}, nil
	}
	return nil, &ExtractError{Message: "重试耗尽: " + lastFeedback, Attempts: e.maxRetry + 1}
}

// ParseOnly 只做本地解析校验（离线演示/测试用，不调 API）。
func ParseOnly(raw string) (ReviewExtract, error) { return ParseReview(raw) }

func backoff(attempt int) time.Duration {
	d := time.Duration(500<<attempt) * time.Millisecond
	if d > 8*time.Second {
		d = 8 * time.Second
	}
	return d + time.Duration(rand.Intn(250))*time.Millisecond
}

// Marshal 把结果转 JSON 字符串，供 CLI 打印。
func Marshal(v any) string {
	b, _ := json.MarshalIndent(v, "", "  ")
	return string(b)
}
