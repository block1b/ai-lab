package stage2

import "strings"

// SYSTEM_PROMPT 对标 Python 版的结构化 Prompt 模板：角色/任务/约束/few-shot。
// 关键：few-shot 与"证据逐字引用、禁止编造"这些行为约束必须放在 System，
// 不能只靠 Schema——Schema 只约束结构，不约束行为。
const SYSTEM_PROMPT = `你是一个商品评论信息抽取助手。

# 任务
从用户输入中抽取情感、属性、证据，并通过 submit_review 工具提交。
严禁在工具调用之外输出任何文字。

# 约束
- sentiment 只能是 positive / negative / neutral / mixed 四者之一，禁止自造
- aspects 是字符串数组，至少 1 项，保留原文中的属性名
- evidence 必须来自原文并逐字引用，禁止改写或编造
- 无法确定时 sentiment 填 neutral、evidence 填原文中最相关的一句，禁止留空

# 示例（few-shot）
输入: "屏幕很好但电池太差"
输出: {"sentiment":"mixed","aspects":["屏幕","电池"],"evidence":"电池太差"}

输入: "物流快，包装仔细，很满意"
输出: {"sentiment":"positive","aspects":["物流","包装"],"evidence":"很满意"}`

// BuildUserPrompt 构造用户消息，重试时附加上次错误反馈。
func BuildUserPrompt(text string, feedback string) string {
	var b strings.Builder
	b.WriteString("【评论原文】\n<content>")
	b.WriteString(text)
	b.WriteString("</content>\n请抽取并通过 submit_review 工具提交。")
	if feedback != "" {
		b.WriteString("\n\n【上次提交未通过，错误信息】\n")
		b.WriteString(feedback)
		b.WriteString("\n请修正后重新提交，严格遵守约束。")
	}
	return b.String()
}
