"""示例 1：最简调用 —— 一句话跑通千问。

运行：python examples/01_basic.py
（在 stage1 目录下运行；脚本会自动把上级目录加入导入路径）
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL, build_llm, usage_of  # noqa: E402

llm = build_llm()
response = llm.invoke("用一句话解释什么是大语言模型。")

print("回答：", response.content)
usage = usage_of(response)
print(f"\n[token] 输入 {usage['prompt']} / 输出 {usage['completion']} / 合计 {usage['total']}")
print(f"[模型] {MODEL}")

# 观察点：response.response_metadata 里还有什么？试着 print(response.response_metadata)
