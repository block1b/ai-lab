"""示例 1：Pydantic Schema 校验演示（不调用 API，纯本地）。

运行：python examples/01_schema_demo.py
目的：直观看到 Pydantic 如何让"格式错误"变成明确的异常。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import ValidationError

from schemas import ReviewExtract  # noqa: E402

print("=== ① 合法数据：校验通过 ===")
good = {"sentiment": "negative", "aspects": ["屏幕", "电池"], "evidence": "电池太差"}
result = ReviewExtract.model_validate(good)
print("通过 ->", result.model_dump())

print("\n=== ② sentiment 自造值：值域校验失败 ===")
bad1 = {"sentiment": "还行吧", "aspects": [], "evidence": "还行"}
try:
    ReviewExtract.model_validate(bad1)
except ValidationError as exc:
    print("拦截 ->", exc.errors()[0]["msg"])

print("\n=== ③ aspects 类型错误：类型校验失败 ===")
bad2 = {"sentiment": "positive", "aspects": "屏幕, 电池", "evidence": "屏幕好"}
try:
    ReviewExtract.model_validate(bad2)
except ValidationError as exc:
    print("拦截 ->", exc.errors()[0]["msg"])

print("\n=== ④ 缺字段：必填校验失败 ===")
bad3 = {"sentiment": "positive", "aspects": []}
try:
    ReviewExtract.model_validate(bad3)
except ValidationError as exc:
    print("拦截 ->", exc.errors()[0]["msg"])

print("\n结论：这些异常就是触发重试的明确信号。")
