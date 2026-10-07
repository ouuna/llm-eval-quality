"""
阈值校准工具
--------------------------------
解决一个具体问题：阈值不该是拍脑袋定的。

现状
----
`eval/thresholds.py` 里的 15 项阈值是**工程初始值**，
来源只有两类：
  · heuristic  领域内常见做法，无本地实验支撑
  · observed   在本项目 30 条 Gold Set 上观察后手工设定

它们**不是**从标注数据里拟合出来的。
README 里必须如实这样写，不能说成「经过实验验证」。

本脚本做什么
------------
在 Gold Set 上扫描阈值，产出**准确率-召回率权衡曲线**，
让人可以基于数据选点，而不是靠感觉。

它回答的问题
------------
    「如果我把'有支撑'的阈值从 0.6 调到 0.5，
      误报和漏报会怎么变？」

用法
----
    python -m eval.calibrate                    # 扫描幻觉检测阈值
    python -m eval.calibrate --metric refusal   # 扫描拒答相关阈值
    python -m eval.calibrate --explain 0.6      # 解释某个阈值的含义

局限（必须说清）
----------------
1. Gold Set 只有 30 条，且由项目作者本人标注 → **存在循环论证**：
   评测器的检测规则是为检出这些样本的缺陷而设计的，
   用同一批数据验证它，结论天然偏乐观。
2. 没有第二位标注者，无法计算标注者间一致性（IAA）。
3. 30 条样本上的扫描结果，统计不确定性很大，
   不应当作精确的最优值，只当作**参考区间**。

要真正校准需要：
  · 200+ 条独立标注样本
  · 至少两位标注者 + IAA 指标
  · 按业务可接受的误报/漏报代价选拐点
"""

import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple

from eval.datasets.gold_set import GoldSample
from eval.evaluators.faithfulness import detect_hallucination
from eval.evaluators.validation import ConfusionMatrix, resolve_gold_samples


# ============================================================
# 指标计算
# ============================================================
def evaluate_at_threshold(samples: List[GoldSample],
                          threshold: float,
                          partial_threshold: float = 0.35
                          ) -> Dict[str, Any]:
    """
    在给定阈值下测量幻觉检测的表现。

    判据与 detect_hallucination 保持一致：
    覆盖率 < partial_threshold → 无支撑 → 判幻觉

    所以这里扫描的是 supported_threshold，
    而 partial 固定——它是「无支撑」的分界，
    两者一起动会让曲线无法解读。
    """
    cm = ConfusionMatrix()
    errors = []

    for s in samples:
        if s.label_hallucination is None:
            continue

        r = detect_hallucination(
            s.answer, s.context,
            supported_threshold=threshold,
            partial_threshold=partial_threshold,
        )
        got = r["has_hallucination"]
        expect = s.label_hallucination

        if expect and got:
            cm.tp += 1
        elif expect and not got:
            cm.fn += 1
        elif not expect and got:
            cm.fp += 1
        else:
            cm.tn += 1

        if expect != got:
            errors.append({
                "id": s.id,
                "expect": expect,
                "got": got,
                "reason": r["reason"][:70],
            })

    m = cm.metrics()
    m["_errors"] = errors
    return m


def scan(samples: List[GoldSample],
         thresholds: List[float]) -> List[Dict[str, Any]]:
    """扫描一系列阈值，返回每个点的指标"""
    rows = []
    for t in thresholds:
        m = evaluate_at_threshold(samples, t)
        rows.append({
            "threshold": t,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "fpr": m["fpr"],
            "fnr": m["fnr"],
            "errors": m["_errors"],
        })
    return rows


# ============================================================
# 权衡曲线
# ============================================================
def format_curve(rows: List[Dict[str, Any]]) -> str:
    """把扫描结果渲染成可读表格"""
    lines = ["=" * 74,
             "阈值扫描：幻觉检测（声明级）",
             "=" * 74,
             "",
             f"{'阈值':>6}{'准确率':>9}{'精确率':>9}{'召回率':>9}"
             f"{'F1':>9}{'误报率':>9}{'漏报率':>9}",
             "-" * 74]

    for r in rows:
        lines.append(
            f"{r['threshold']:>6.2f}"
            f"{_pct(r['accuracy']):>9}"
            f"{_pct(r['precision']):>9}"
            f"{_pct(r['recall']):>9}"
            f"{_pct(r['f1']):>9}"
            f"{_pct(r['fpr']):>9}"
            f"{_pct(r['fnr']):>9}")

    lines.append("")

    # 找出 F1 最优点
    valid = [r for r in rows if r["f1"] is not None]
    if valid:
        best = max(valid, key=lambda x: x["f1"])
        lines.append(f"F1 最优：阈值 {best['threshold']:.2f}（F1={_pct(best['f1'])}）")

        # 找出漏报率 <= 5% 的最低阈值
        safe = [r for r in valid if r["fnr"] is not None and r["fnr"] <= 0.05]
        if safe:
            lowest = min(safe, key=lambda x: x["threshold"])
            lines.append(
                f"漏报率 ≤5% 的最低阈值：{lowest['threshold']:.2f}"
                f"（漏报 {_pct(lowest['fnr'])}，误报 {_pct(lowest['fpr'])}）")
        else:
            lines.append("没有任何阈值能把漏报率压到 5% 以下")

    lines.append("")
    lines.append("=" * 74)
    lines.append("怎么读这张表")
    lines.append("=" * 74)
    lines.append("""
两个方向的代价完全不对称：

  误报（FP）  把对的判成错的 → 使用者觉得评测器「太严」，
              会开始不信任报告，最终忽略告警
  漏报（FN）  把错的判成对的 → **缺陷会漏到生产环境**

所以选阈值时不该只看 F1 或准确率，
而该问：**漏报到生产环境的代价有多大？**

通常宁可容忍一定误报，也要压低漏报。
但这是业务决策，不是技术决策——
所以本工具只给数据，不替你选。
""")
    return "\n".join(lines)


def format_errors(rows: List[Dict[str, Any]],
                  threshold: float) -> str:
    """列出某个阈值下判错的样本"""
    for r in rows:
        if abs(r["threshold"] - threshold) > 1e-6:
            continue

        lines = ["=" * 74,
                 f"阈值 {threshold:.2f} 下的误判样本",
                 "=" * 74, ""]
        if not r["errors"]:
            lines.append("无")
            return "\n".join(lines)

        for e in r["errors"]:
            kind = "漏报（有幻觉却判无）" if e["expect"] else \
                   "误报（无幻觉却判有）"
            lines.append(f"[{kind}] {e['id']}")
            lines.append(f"  {e['reason']}")
            lines.append("")

        return "\n".join(lines)

    return f"未扫描到阈值 {threshold}"


def _pct(v):
    return f"{v:.1%}" if v is not None else "  n/a"


# ============================================================
# CLI
# ============================================================
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="阈值校准工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python -m eval.calibrate                    扫描 0.30~0.80 的阈值
  python -m eval.calibrate --start 0.4 --end 0.7 --step 0.05
  python -m eval.calibrate --errors 0.6       查看 0.6 下判错了哪些
  python -m eval.calibrate --explain          打印局限声明
        """,
    )
    parser.add_argument("--start", type=float, default=0.30)
    parser.add_argument("--end", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--errors", type=float, default=None,
                        help="打印该阈值下判错的样本")
    parser.add_argument("--explain", action="store_true",
                        help="打印局限声明")

    args = parser.parse_args(argv)

    if args.explain:
        print(__doc__)
        return 0

    samples, source = resolve_gold_samples()
    labeled = [s for s in samples if s.label_hallucination is not None]
    usable = [s for s in labeled if s.reviewed]

    print("=" * 74)
    print("阈值校准")
    print("=" * 74)
    print(f"数据源：{source}")
    print(f"有幻觉标签的样本：{len(labeled)} 条"
          f"（其中已复核 {len(usable)} 条）")
    print()

    if not labeled:
        print("没有可用的标注数据，无法校准。")
        print("请先在 goldset/gold_set.json 中完成标注。")
        return 1

    thresholds = []
    v = args.start
    while v <= args.end + 1e-9:
        thresholds.append(round(v, 4))
        v += args.step

    rows = scan(labeled, thresholds)
    print(format_curve(rows))

    if args.errors is not None:
        print()
        print(format_errors(rows, args.errors))

    print()
    print("=" * 74)
    print("重要提醒")
    print("=" * 74)
    print(f"""
本结果基于 **{len(labeled)} 条**样本，其中由项目作者本人标注的占多数。

评测器的检测规则正是为检出这些样本里的缺陷而设计的，
所以在这批数据上表现好是**循环论证**的结果，
不代表它在真实数据上同样可靠。

结论应这样使用：
  ✓ 「阈值在 0.5~0.7 之间时表现差异不大」—— 可作为参考区间
  ✗ 「阈值 0.6 是最优解」—— 30 条样本支撑不了这个结论

要真正校准，见本文件开头的「局限」一节。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
