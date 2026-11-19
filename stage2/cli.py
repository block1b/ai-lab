"""阶段 2 命令行工具：输入评论，输出抽取结果（情感/属性/证据）。

用法：
    python cli.py "屏幕很好但电池太差"
    python cli.py --mode manual "物流很快，包装仔细"
    python cli.py                      # 交互模式
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import build_llm
from extractor import ExtractionError, build_extractor


def main():
    parser = argparse.ArgumentParser(description="阶段2：评论信息抽取器")
    parser.add_argument("text", nargs="*", help="要抽取的评论文本")
    parser.add_argument("--mode", default="structured", choices=["structured", "manual"])
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    llm = build_llm(temperature=0.0)
    extractor = build_extractor(llm, mode=args.mode, max_retries=args.retries)

    def run_one(text: str):
        try:
            result, meta = extractor.run(text)
            print("\n----- 抽取结果 -----")
            print(f"  sentiment : {result.sentiment}")
            print(f"  aspects   : {result.aspects}")
            print(f"  evidence  : {result.evidence}")
            print(f"  (尝试 {meta['attempts']} 次, token {meta['tokens']})")
        except ExtractionError as exc:
            print(f"\n[抽取失败] {exc}")
            if exc.last_raw:
                print(f"[最后输出] {exc.last_raw[:300]}")

    if args.text:
        run_one(" ".join(args.text))
        return

    print("进入交互模式，输入 exit 退出。")
    while True:
        try:
            text = input("\n评论> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if text.lower() in ("exit", "quit"):
            print("再见")
            break
        if text:
            run_one(text)


if __name__ == "__main__":
    main()
