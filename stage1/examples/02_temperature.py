"""示例 2：温度对比实验 —— 直观理解 temperature 的作用。

运行：python examples/02_temperature.py

同一句话，分别用 temperature=0（确定）和 temperature=1.5（发散）各跑几次，
对比输出的稳定性与创造性。这是理解「温度」最快的方式。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import build_llm  # noqa: E402

PROMPT = "给一家咖啡店想一句广告语。"

for temp in (0.0, 1.5):
    llm = build_llm(temperature=temp)
    print(f"\n===== temperature = {temp} =====")
    # 同一温度跑 3 次，看输出是否一致
    for i in range(1, 4):
        try:
            answer = llm.invoke(PROMPT).content.strip().replace("\n", " ")
            print(f"  第{i}次: {answer}")
        except Exception as exc:
            print(f"  第{i}次失败: {exc}")

# 预期现象：
#   temperature=0   -> 多次结果高度相似甚至完全一致，适合事实问答/信息抽取
#   temperature=1.5 -> 每次都不一样，更有创意，但可能出现不稳定或跑偏
