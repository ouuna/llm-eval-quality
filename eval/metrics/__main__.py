"""
统计层自检入口：python -m eval.metrics.__main__
"""

from eval.metrics import (
    describe, distribution, mean, median, p95, stability_stats,
)


def main():
    sample = [120, 350, 800, 1200, 1500, 1500, 3200, 2100]

    print("=" * 60)
    print("统计层自检")
    print("=" * 60)
    print(f"样本= {sample}")
    print()
    print("describe():")
    for k, v in describe(sample).items():
        print(f"  {k:8} = {v}")

    print()
    print("空输入的行为（应全部为 None，而非 0）：")
    print(f"  mean([])      = {mean([])}")
    print(f"  median([])    = {median([])}")
    print(f"  p95([])       = {p95([])}")
    print(f"  describe([])  = {describe([])}")

    print()
    print("分位数插值验证：")
    print(f"  p95([1,2,3,4,5]) = {p95([1, 2, 3, 4, 5])}"
          f"  （插值法 4.8；旧索引法会得 5）")

    print()
    print("稳定性汇总（含极差）：")
    for k, v in stability_stats([0.90, 0.95, 0.88]).items():
        print(f"  {k:8} = {v}")

    print()
    print("分布：")
    for b in distribution([1, 2, 2, 3, 3, 3, 4, 5]):
        print(f"  [{b['lower']}, {b['upper']}) -> {b['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
