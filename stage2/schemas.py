"""Pydantic Schema：定义"评论抽取器"的输出契约。

这一个文件就是阶段 2 任务「用 Pydantic 定义输出 Schema 并做校验」的落地。
字段设计直接对应打卡表验收项：情感 / 属性 / 证据句。
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ReviewExtract(BaseModel):
    """从一条商品评论中抽取的结构化结果。

    Pydantic 会在模型实例化时自动校验：
    - sentiment 只能是三个枚举值之一（值域约束）
    - aspects 必须是字符串列表（类型约束）
    - 各字段必填（缺字段会抛 ValidationError）
    """

    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="整体情感：positive=正面, negative=负面, neutral=中性或混合"
    )
    aspects: list[str] = Field(
        description="评论中提到的产品属性，如 ['屏幕', '电池']；没有则空列表"
    )
    evidence: str = Field(
        min_length=1,
        description="支撑情感判断的原文证据片段，必须逐字来自原文，不得编造",
    )

    @field_validator("aspects")
    @classmethod
    def cap_aspects(cls, value: list[str]) -> list[str]:
        """自定义校验：属性数量不超过 8 个，且去掉空白项。"""
        cleaned = [item.strip() for item in value if item.strip()]
        if len(cleaned) > 8:
            raise ValueError("aspects 最多 8 项")
        return cleaned


# 把 Schema 转成 JSON Schema，供 prompt 注入 / 观察模型看到的约束
def schema_json_example() -> str:
    import json

    return json.dumps(ReviewExtract.model_json_schema(), ensure_ascii=False, indent=2)
