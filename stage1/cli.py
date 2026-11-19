"""
阶段 1 验收工具：命令行 AI 问答 + Token 统计

验收标准（对应学习计划阶段 1）：
    输入任意问题 -> 输出模型回答 -> 打印本次消耗 token 数

用法：
    python cli.py "什么是 RAG？"                 # 单次提问
    python cli.py --system "你是Python老师" "解释装饰器"
    python cli.py --temperature 1.2 "写一句诗"   # 调温度
    python cli.py                               # 进入交互模式，输入 exit 退出
"""

import argparse
import os
import sys

# 把本文件所在目录加入导入路径：无论从哪个工作目录、用什么方式运行，
# 都能找到同目录的 config.py（例如从上级目录执行 python -m stage1.cli 时）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage, SystemMessage

from config import MODEL, build_llm, usage_of

DEFAULT_SYSTEM = "你是一个简洁、有帮助的 AI 助手，回答控制在 200 字以内。"


def ask(llm, question: str, system: str = DEFAULT_SYSTEM):
    """用「系统消息 + 用户消息」的标准结构发起一次聊天调用。"""
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=question),
    ]
    return llm.invoke(messages)


def print_result(response) -> None:
    """打印回答和 token 统计。"""
    print("\n===== 回答 =====")
    print(response.content)
    usage = usage_of(response)
    print("\n===== 本次 Token 消耗 =====")
    print(f"  输入 (prompt)     : {usage['prompt']}")
    print(f"  输出 (completion) : {usage['completion']}")
    print(f"  合计 (total)      : {usage['total']}")
    print(f"  模型              : {MODEL}")


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段1验收工具：命令行问答 + Token 统计")
    parser.add_argument("question", nargs="*", help="要问的问题（不传则进入交互模式）")
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="系统提示词")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度 0~2，默认 0.7")
    args = parser.parse_args()

    llm = build_llm(temperature=args.temperature)

    if args.question:
        question = " ".join(args.question)
        print_result(ask(llm, question, args.system))
        return

    print(f"进入交互模式（模型: {MODEL}）。输入 exit 或 quit 退出。")
    while True:
        try:
            question = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if question.lower() in ("exit", "quit"):
            print("再见")
            break
        if not question:
            continue
        try:
            print_result(ask(llm, question, args.system))
        except Exception as exc:  # 交互模式下不让一次失败中断整个会话
            print(f"[调用失败] {exc}")


if __name__ == "__main__":
    main()

# --temperature 0.7

# ===== 回答 =====
# RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索与生成模型的技术。它在生成回答前，先从外部知识库中检索相关文档或片段，再将这些信息作为上下文输入给生成模型（如大语言模型），从而提升回答的准确性、时效性和可解释性。

# 相比纯生成模型，RAG 能有效避免“幻觉”问题，尤其适用于需要精准知识的应用场景，如问答系统、智能客服和专业领域助手。其核心思想是：先检索，后生成。

# --temperature 1

# ===== 回答 =====
# RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索与生成模型的AI技术。它在生成回答前，先从外部知识库中检索相关文档或信息，再基于检索内容生成更准确、可靠的回答。相比纯生成模型，RAG能减少幻觉，提升回答的准确性与可解释性，广泛应用于问答系统、客服机器人和知识库查询等场景。

#  --temperature 1.5

# ===== 回答 =====
# RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索与生成模型的AI技术。它在生成回答前，先从外部知识库中检索相关文档或信息，再基于检索内容生成更准确、可靠的回答。

# 相比纯生成模型（如大语言模型），RAG能减少“幻觉”问题，提升答案的时效性和准确性。典型应用包括智能客服、问答系统、知识库查询等。

# 工作流程：用户提问 → 检索相关文档 → 用检索内容+问题共同生成答案。

# 例如：用户问“2023年诺贝尔奖得主是谁？”，RAG会先检索最新获奖信息，再生成准确答案。
