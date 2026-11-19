"""结构化 Prompt 模板：对应打卡表「结构化 Prompt 模板（角色/任务/约束/示例/输入）」。
"""


def build_system_prompt(schema_description: str, few_shot: bool = True) -> str:
    parts = [
        "# 角色",
        "你是一个严格的商品评论信息抽取助手。",
        "",
        "# 任务",
        "从 <input> 中抽取字段，只输出一个 JSON 对象，不要任何额外文字、解释或 Markdown 代码块。",
        "",
        "# 约束",
        "- 只输出这三个字段：sentiment, aspects, evidence",
        "- sentiment 只能是 positive / negative / neutral 三者之一，禁止自造值",
        "- aspects 是字符串数组，没有提到属性就填 []",
        "- evidence 必须逐字引用原文片段，禁止编造；找不到就填一句原文中最相关的短语",
        "- 不要输出定义之外的字段",
        "",
        "# 输出格式（JSON Schema）",
        schema_description,
    ]
    if few_shot:
        parts += [
            "",
            "# 示例（few-shot）",
            '输入: "屏幕很好但电池太差，充电还发烫"',
            '输出: {"sentiment":"negative","aspects":["屏幕","电池","充电"],"evidence":"电池太差，充电还发烫"}',
            "",
            '输入: "物流很快，包装仔细，下次还买"',
            '输出: {"sentiment":"positive","aspects":["物流","包装"],"evidence":"物流很快，包装仔细，下次还买"}',
        ]
    return "\n".join(parts)


def build_user_prompt(text: str) -> str:
    return f"<input>\n{text}\n</input>"
