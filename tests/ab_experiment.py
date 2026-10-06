"""
对照实验：验证防幻觉约束的实际效果
---------------------------------
用途：证明 Prompt 中的防幻觉约束不是"看起来有用"，而是有量化收益。

方法（A/B 实验）：
    对照组：关闭 prompt.enable_hallucination_guard
    实验组：开启（当前配置）
    对两组分别跑同一批评测用例，对比域外拒答率、幻觉率等指标。

用法：
    python tests/ab_experiment.py
"""

import os
import sys
import json
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tests"))

from config_loader import load_config, load_cases  # noqa: E402


AB_REPORT = {
    "json": "ab_experiment.json",
}


def run_group(label, enable_guard, cases):
    """
    以指定配置运行一次评测。

    通过环境变量开关控制被测系统的 Prompt 行为，
    避免修改配置文件影响其他运行。
    """
    import app.rag as rag
    from evaluator import run_all_eval

    # 运行时切换防幻觉约束
    rag.ENABLE_HALLUCINATION_GUARD = enable_guard

    print(f"\n{'=' * 64}")
    print(f"实验组：{label}（防幻觉约束 = {'开启' if enable_guard else '关闭'}）")
    print("=" * 64)

    summary = run_all_eval(cases=cases, verbose=False)
    m = summary["metrics"]

    print(f"  域外拒答率    {m['refusal_rate_out_domain']:.0%}")
    print(f"  无幻觉率      {m['hallucination_free_rate']:.0%}")
    print(f"  域内准确率    {m['accuracy_in_domain']:.0%}")
    print(f"  总体通过率    {m['overall_pass_rate']:.0%}")
    print(f"  检索命中率    {m['retrieval_hit_rate']:.0%}")

    return {
        "label": label,
        "enable_guard": enable_guard,
        "metrics": m,
        "cases": [
            {
                "case_id": c["case_id"],
                "category": c["category"],
                "question": c["question"],
                "actual_output": c["actual_output"],
                "hallucination_free": c["hallucination_free"],
                "hallucination_reason": c["hallucination_reason"],
                "passed": c["passed"],
            }
            for c in summary["cases"]
        ],
    }


def main():
    cfg = load_config()
    cases = load_cases()

    print("=" * 64)
    print("对照实验：防幻觉约束的实际效果验证")
    print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型：{os.getenv('OPENAI_MODEL_NAME')}")
    print(f"用例：{len(cases)} 条")
    print("=" * 64)

    # 实验组：开启防幻觉约束
    exp = run_group("实验组", True, cases)
    time.sleep(1)

    # 对照组：关闭防幻觉约束
    ctl = run_group("对照组", False, cases)

    # ---------- 对比 ----------
    em, cm = exp["metrics"], ctl["metrics"]

    print("\n" + "=" * 64)
    print("对照结果")
    print("=" * 64)
    print(f"{'指标':<16}{'对照组(无约束)':>16}{'实验组(有约束)':>16}{'差异':>14}")

    rows = [
        ("域外拒答率", "refusal_rate_out_domain", True),
        ("无幻觉率", "hallucination_free_rate", True),
        ("域内准确率", "accuracy_in_domain", True),
        ("总体通过率", "overall_pass_rate", True),
        ("检索命中率", "retrieval_hit_rate", True),
    ]

    diffs = {}
    for label, key, is_pct in rows:
        a, b = cm[key], em[key]
        diff = b - a
        diffs[key] = {
            "control": a, "experiment": b, "diff": round(diff, 3),
        }
        sign = "+" if diff > 0 else ""
        print(f"{label:<16}{a:>15.0%}{b:>16.0%}{sign + format(diff, '.0%'):>14}")

    # 典型案例对比
    print("\n" + "=" * 64)
    print("典型案例（域外用例，系统本应拒答）")
    print("=" * 64)

    ctl_map = {c["case_id"]: c for c in ctl["cases"]}
    shown = 0
    for ec in exp["cases"]:
        if ec["category"] != "out_domain":
            continue
        cc = ctl_map.get(ec["case_id"])
        if not cc:
            continue

        # 只展示有差异的案例
        if cc["hallucination_free"] == ec["hallucination_free"]:
            continue

        print(f"\n问题：{ec['question']}")
        print(f"  对照组：{cc['actual_output'][:70] or '（拒答）'}")
        print(f"  实验组：{ec['actual_output'][:70] or '（拒答）'}")
        shown += 1
        if shown >= 3:
            break

    if shown == 0:
        print("（两组行为完全一致，说明当前场景下该约束影响有限）")

    # ---------- 结论 ----------
    refusal_gain = diffs["refusal_rate_out_domain"]["diff"]
    halluc_gain = diffs["hallucination_free_rate"]["diff"]

    print("\n" + "=" * 64)
    print("实验结论")
    print("=" * 64)
    if refusal_gain > 0:
        print(f"  防幻觉约束使域外拒答率提升 {refusal_gain:.0%}")
        print(f"  无幻觉率提升 {halluc_gain:.0%}")
        print("  结论：该约束对防止模型编造内容有实质作用，建议保留。")
    else:
        print("  未观察到正向差异，可能原因：")
        print("    1. 当前模型本身拒答倾向较强，约束未体现额外价值")
        print("    2. 域外用例召回率过低，模型未进入生成环节")
        print("    3. 需扩大域外用例数量以获得统计显著性")
    print("=" * 64)

    # 保存
    out = os.path.join(PROJECT_ROOT, "reports", AB_REPORT["json"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": os.getenv("OPENAI_MODEL_NAME"),
            "case_count": len(cases),
            "diff": diffs,
            "control_group": ctl,
            "experiment_group": exp,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果：{out}")


if __name__ == "__main__":
    main()
