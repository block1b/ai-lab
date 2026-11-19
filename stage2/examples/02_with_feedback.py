"""示例 2：错误反馈重试演示（真实调用 API）。

运行：python examples/02_with_feedback.py
目的：观察"第一次输出不合规 -> 把错误反馈给模型 -> 第二次修正"的过程。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import build_llm  # noqa: E402
from extractor import build_extractor  # noqa: E402

llm = build_llm(temperature=0.0)
extractor = build_extractor(llm, mode="manual", max_retries=3)

# 故意用一句容易让模型"自造情感值"的评论，观察是否能一次通过
text = "这个东西吧，不好也不坏，说不上来什么感觉"

result, meta = extractor.run(text)
print("抽取结果：", result.model_dump())
print(f"尝试次数：{meta['attempts']} | 累计 token：{meta['tokens']}")
print("\n如果 attempts > 1，说明第一次输出没通过校验，错误反馈机制生效了。")
