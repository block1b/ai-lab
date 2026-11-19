"""示例 3：流式输出 —— 逐字打印回答，同时统计 token。

运行：python examples/03_stream.py

流式输出能显著降低「首 token 延迟」的体感等待，是后面做产品级应用的必备能力。
这里用回调 Handler 实现逐字打印，invoke 仍返回完整响应，方便拿 token 统计。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI

from config import API_KEY, BASE_URL, MODEL, usage_of  # noqa: E402

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,
    temperature=0.7,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],  # 每个新 token 到达就打印
)

print("模型正在逐字输出：\n")
response = llm.invoke("用三句话解释什么是 RAG（检索增强生成）。")

usage = usage_of(response)
print(f"\n\n[token] 输入 {usage['prompt']} / 输出 {usage['completion']} / 合计 {usage['total']}")
