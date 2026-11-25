// stage4：Eino ReAct Agent，对标 Python 版 graph.py / react.py。
package stage4

import (
	"context"
	"fmt"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/flow/agent/react"
	"github.com/cloudwego/eino/schema"
	"golab/pkg/config"
	"golab/stage4/tools"
)

const systemPrompt = `你是一个财报分析助手。遵循 ReAct 流程：
1. 需要数值事实时，调用 get_metric 或 rag_search 工具获取，禁止凭记忆编造。
2. 增长率、利润率等计算必须调用 calculate 或 growth_rate 工具。
3. 信息足够后给出最终答案，不要无意义地反复调用工具。
4. 最终答案必须含具体数字，并说明数据来源。`

// NewAgent 创建带 4 个工具的 ReAct Agent。
// MaxStep 是对标 Python 版"最大步数"的护栏（Eino 内置）。
func NewAgent(ctx context.Context, maxStep int) (*react.Agent, error) {
	temp := float32(0)
	cm, err := config.NewChatModel(ctx, &temp)
	if err != nil {
		return nil, err
	}
	tc, ok := cm.(model.ToolCallingChatModel)
	if !ok {
		return nil, fmt.Errorf("模型不支持 ToolCallingChatModel")
	}
	allTools, err := tools.BuildTools()
	if err != nil {
		return nil, err
	}
	toolInfos := make([]*schema.ToolInfo, 0, len(allTools))
	for _, t := range allTools {
		info, err := t.Info(ctx)
		if err != nil {
			return nil, fmt.Errorf("读取工具信息失败: %w", err)
		}
		toolInfos = append(toolInfos, info)
	}
	bound, err := tc.WithTools(toolInfos)
	if err != nil {
		return nil, err
	}
	if maxStep <= 0 {
		maxStep = 8
	}
	base := make([]tool.BaseTool, 0, len(allTools))
	for _, t := range allTools {
		base = append(base, t)
	}
	return react.NewAgent(ctx, &react.AgentConfig{
		ToolCallingModel: bound,
		ToolsConfig:      compose.ToolsNodeConfig{Tools: base},
		MessageModifier:  react.NewPersonaModifier(systemPrompt),
		MaxStep:          maxStep,
	})
}

// Run 执行 Agent，返回最终答案。
func Run(ctx context.Context, ag *react.Agent, question string) (string, error) {
	resp, err := ag.Generate(ctx, []*schema.Message{schema.UserMessage(question)})
	if err != nil {
		return "", err
	}
	return resp.Content, nil
}
