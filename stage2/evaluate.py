"""阶段 2 验收脚本：连续 N 次测试，统计格式正确率（对应 ≥98% 验收标准）。

用法：
    python evaluate.py                 # 默认跑 100 次
    python evaluate.py --runs 20       # 先跑 20 次快速看看
    python evaluate.py --mode manual   # 用手动模式测
    python evaluate.py --delay 0.2     # 每次调用间隔秒数（控成本/限流）
"""

import argparse
import time

from config import build_llm
from dataset import SAMPLES
from extractor import ExtractionError, build_extractor


def main():
    parser = argparse.ArgumentParser(description="阶段2验收：格式正确率测试")
    parser.add_argument("--runs", type=int, default=100, help="总测试次数，默认100")
    parser.add_argument("--mode", default="structured", choices=["structured", "manual"])
    parser.add_argument("--delay", type=float, default=0.1, help="每次调用间隔秒")
    parser.add_argument("--retries", type=int, default=3, help="单次最大重试次数")
    args = parser.parse_args()

    llm = build_llm(temperature=0.0)
    extractor = build_extractor(llm, mode=args.mode, max_retries=args.retries)

    passed = 0
    failed = 0
    first_try = 0
    total_tokens = 0
    total_prompt = 0
    total_completion = 0
    start = time.time()

    print(f"开始测试：{args.runs} 次 | 模式={args.mode} | 数据集={len(SAMPLES)} 条\n")
    for i in range(1, args.runs + 1):
        sample = SAMPLES[(i - 1) % len(SAMPLES)]
        try:
            result, meta = extractor.run(sample["text"])
            passed += 1
            if meta["attempts"] == 1:
                first_try += 1
            total_tokens += meta["tokens"]
            total_prompt += meta.get("prompt_tokens", 0)
            total_completion += meta.get("completion_tokens", 0)
            tag = "OK" if meta["attempts"] == 1 else f"OK(重试{meta['attempts']-1}次)"
            print(f"[{i:>3}/{args.runs}] {tag:<12} {result.sentiment:<8} {sample['text'][:24]}")
        except ExtractionError as exc:
            failed += 1
            print(f"[{i:>3}/{args.runs}] FAIL         {exc}")

        if args.delay:
            time.sleep(args.delay)

    elapsed = time.time() - start
    rate = passed / args.runs * 100
    first_rate = first_try / args.runs * 100

    print("\n" + "=" * 46)
    print(f"格式正确率     : {rate:.1f}%  ({passed}/{args.runs})   目标 ≥ 98%")
    print(f"首次通过率     : {first_rate:.1f}%  ({first_try}/{args.runs})")
    print(f"失败次数       : {failed}")
    print(f"总耗时         : {elapsed:.1f}s")
    print(f"累计 token     : {total_tokens}  (输入 {total_prompt} / 输出 {total_completion})")
    if total_tokens:
        print(f"单次均值       : {total_tokens / args.runs:.0f} token")
    print("=" * 46)
    if rate >= 98:
        print("结论：达到阶段 2 验收标准 ✅")
    else:
        print("结论：未达标，建议检查 prompt 约束 / few-shot / 重试反馈 ❌")


if __name__ == "__main__":
    main()
