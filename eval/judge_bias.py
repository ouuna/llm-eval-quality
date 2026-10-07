"""
Judge Bias 自检工具
--------------------------------
为什么需要它
------------
LLM-as-a-Judge 最大的风险是：**评分不可信但看不出来**。
报告上写着「faithfulness 4.2/5」，读报告的人无从判断这个数字能不能信。

三类必须实测的偏差（不能假设不存在）：

  1. **刻度漂移**
     temperature=0 理论上应确定性，但服务商不保证严格确定性。
     同一输入多次评分差异大 → 该 Judge 不可靠。
     早先假定「设了 temperature=0 就确定了」，这是假设，不是事实。

  2. **长度偏好**
     长的回答看起来更「详实」，容易得高分。

  3. **位置偏好**
     A/B 交换顺序，评分应保持一致。
     若总是「排在前面的得高分」→ 有位置偏好，评分不可信。
     这是 LLM-as-a-Judge 论文里的标准检测方法。

为什么做成独立 CLI 而不是每条用例都跑
------------------------------------
每个检查要多次调用 LLM（一致性 3 次、位置偏好 2×2 次）。
每条用例都跑会让评测成本翻好几倍。

正确做法是：**在评测之前先验证 Judge 本身可信**，
可信才启用它做评测；不可信就报告出来并降级。
这才是「按需配置」的合理顺序。

用法
----
    python -m eval.judge_bias                跑全部检查
    python -m eval.judge_bias --consistency  只跑一致性
    python -m eval.judge_bias --position     只跑位置偏好
    python -m eval.judge_bias --length       只跑长度偏好
"""

import argparse
from typing import Any, Dict, List

# 判定阈值。写明来源，不藏在代码里。
STABILITY_TOLERANCE = 0.2      # 重复评分极差超过它算漂移
POSITION_BIAS_THRESHOLD = 0.3  # 位置造成的分差超过它算有偏好
LENGTH_BIAS_THRESHOLD = 0.5    # 长短答案分差超过它算有偏好


def _samples() -> List[Dict[str, Any]]:
    """
    用于自检的样本。

    刻意用项目自己的知识库内容——
    这样不需要额外准备数据，也不会引入新的领域。
    样本量小是刻意的：这是「验证工具是否可信」，
    不是评测被测系统。
    """
    return [
        {
            "id": "JB01",
            "question": "什么是等价类划分？",
            "context": "等价类划分：把输入域划分为若干互不相交的子集。",
            "good": "等价类划分是把输入域划分为若干互不相交子集的方法，"
                    "每个子集内的输入具有相同预期结果。",
            "bad": "等价类划分就是随便分一下。",
        },
        {
            "id": "JB02",
            "question": "回归测试的作用是什么？",
            "context": "回归测试：修改代码后重新执行相关测试用例，"
                       "确认没有引入新缺陷。",
            "good": "修改代码后重新执行相关测试用例，确认没有引入新缺陷。",
            "bad": "回归测试就是随便测一下。",
        },
    ]


def run_consistency(client, repeats: int = 3) -> Dict[str, Any]:
    """刻度漂移检测"""
    from eval.evaluators.judge import check_judge_consistency

    results = []
    for s in _samples():
        r = check_judge_consistency(
            client, s["question"], s["good"], s["context"],
            repeats=repeats, tolerance=STABILITY_TOLERANCE)
        results.append(r)

    usable = [r for r in results if r.get("available")]
    if not usable:
        return {
            "available": False,
            "reason": "所有样本的有效评分不足，Judge 可能无法调用",
            "details": results,
        }

    # 全部样本都稳定才算稳定
    all_stable = all(r.get("stable") for r in usable)
    spreads = [r.get("spread", 0) for r in usable]

    return {
        "available": True,
        "stable": all_stable,
        "samples": len(usable),
        "max_spread": max(spreads),
        "mean_spread": round(sum(spreads) / len(spreads), 3),
        "tolerance": STABILITY_TOLERANCE,
        "reason": (
            f"{len(usable)} 个样本最大极差 {max(spreads):.3f}"
            f"（阈值 {STABILITY_TOLERANCE}）"
            f"→ {'稳定' if all_stable else '存在刻度漂移，该 Judge 评分不可信'}"
        ),
        "details": results,
    }


def run_position_bias(client, rounds: int = 2) -> Dict[str, Any]:
    """位置偏好检测"""
    from eval.evaluators.judge import check_position_bias

    results = []
    for s in _samples():
        r = check_position_bias(
            client, s["question"], s["context"],
            s["good"], s["bad"], rounds=rounds)
        results.append(r)

    usable = [r for r in results if r.get("available")]
    if not usable:
        return {
            "available": False,
            "reason": "位置偏好检测无有效结果",
            "details": results,
        }

    max_effect = max(abs(r.get("position_effect", 0)) for r in usable)
    biased = max_effect > POSITION_BIAS_THRESHOLD

    return {
        "available": True,
        "biased": biased,
        "max_effect": round(max_effect, 3),
        "threshold": POSITION_BIAS_THRESHOLD,
        "reason": (
            f"最大位置效应 {max_effect:.3f}"
            f"（阈值 {POSITION_BIAS_THRESHOLD}）"
            f"→ {'存在位置偏好，评分不可信' if biased else '未发现明显位置偏好'}"
        ),
        "details": results,
    }


def run_length_bias(client) -> Dict[str, Any]:
    """
    长度偏好检测（简化版）。

    方法：同一问题，喂「详尽但可能冗长」与「简洁但完整」的回答，
    看 Judge 是否系统性偏好长的那个。

    局限：真实的长度偏好研究需要控制信息量，
    这里只能用「详尽 vs 简洁」近似，结论仅供参考。
    """
    samples = []
    for s in _samples():
        verbose = s["good"] + " " + "另外，" * 12 + "除此之外没有别的补充内容。"
        r_good = client.judge(s["question"], s["good"], s["context"])
        r_verbose = client.judge(s["question"], verbose, s["context"])

        if not (r_good.available and r_verbose.available):
            continue
        if r_good.final_score is None or r_verbose.final_score is None:
            continue

        delta = r_verbose.final_score - r_good.final_score
        samples.append({
            "id": s["id"],
            "short_score": r_good.final_score,
            "verbose_score": r_verbose.final_score,
            "delta": round(delta, 3),
        })

    if not samples:
        return {
            "available": False,
            "reason": "无有效评分，无法检测长度偏好",
        }

    max_delta = max(abs(s["delta"]) for s in samples)
    biased = max_delta > LENGTH_BIAS_THRESHOLD

    return {
        "available": True,
        "biased": biased,
        "max_delta": round(max_delta, 3),
        "threshold": LENGTH_BIAS_THRESHOLD,
        "reason": (
            f"冗长回答最高多拿 {max_delta:.3f} 分"
            f"（阈值 {LENGTH_BIAS_THRESHOLD}）"
            f"→ {'可能存在长度偏好' if biased else '未发现明显长度偏好'}"
        ),
        "caveat": "本检测用「详尽 vs 简洁」近似，未严格控制信息量，"
                  "结论仅供参考",
        "samples": samples,
    }


def format_report(checks: Dict[str, Any]) -> str:
    """渲染自检报告"""
    lines = ["=" * 68,
             "LLM Judge Bias 自检",
             "=" * 68, ""]

    for name, label in (("consistency", "刻度漂移"),
                        ("position_bias", "位置偏好"),
                        ("length_bias", "长度偏好")):
        c = checks.get(name)
        lines.append(f"【{label}】")
        if not c:
            lines.append("  未执行")
            lines.append("")
            continue
        if not c.get("available"):
            lines.append(f"  不可用：{c.get('reason')}")
            lines.append("")
            continue
        lines.append(f"  {c['reason']}")
        if c.get("caveat"):
            lines.append(f"  说明：{c['caveat']}")
        lines.append("")

    # 总体判定
    lines.append("=" * 68)
    lines.append("总体判定")
    lines.append("=" * 68)

    usable = [c for c in checks.values()
              if c and c.get("available")]
    if not usable:
        lines.append("")
        lines.append("所有检查均不可用——Judge 很可能无法调用。")
        lines.append("评测应降级为不使用 Judge，并报告原因。")
    else:
        problems = []
        if checks.get("consistency", {}).get("available") and \
                not checks["consistency"].get("stable"):
            problems.append("刻度漂移")
        if checks.get("position_bias", {}).get("available") and \
                checks["position_bias"].get("biased"):
            problems.append("位置偏好")
        if checks.get("length_bias", {}).get("available") and \
                checks["length_bias"].get("biased"):
            problems.append("长度偏好")

        if problems:
            lines.append("")
            lines.append(f"检出问题：{'、'.join(problems)}")
            lines.append("")
            lines.append("**建议：不要把这个 Judge 用于质量门禁。**")
            lines.append("它给出的分数会误导排查方向——")
            lines.append("读报告的人会把评分当成客观结论。")
        else:
            lines.append("")
            lines.append("未检出明显偏差，Judge 评分可作为参考。")
            lines.append("注意：「未检出」不等于「不存在」，")
            lines.append("本检测的样本量很小（2 个），只能证伪不能证真。")

    lines.append("")
    lines.append("=" * 68)
    lines.append("为什么这件事重要")
    lines.append("=" * 68)
    lines.append("""
LLM Judge 的失败模式与普通代码不同：
它的输出**看起来总是合理的**——五维打分、给出理由、列出证据。
你无法从报告上看出它是瞎编的。

所以「Judge 本身可不可信」必须靠实测，
而不能因为「它每次都输出了格式正确的结果」就默认它对。
""")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="LLM Judge Bias 自检")
    parser.add_argument("--consistency", action="store_true",
                        help="只跑刻度漂移检测")
    parser.add_argument("--position", action="store_true",
                        help="只跑位置偏好检测")
    parser.add_argument("--length", action="store_true",
                        help="只跑长度偏好检测")
    parser.add_argument("--repeats", type=int, default=3,
                        help="一致性检测的重复次数")
    args = parser.parse_args(argv)

    from eval.evaluators.judge import JudgeClient
    from eval import env_loader

    if env_loader.missing_required():
        print("=" * 68)
        print("未配置 API Key，无法自检 Judge")
        print("=" * 68)
        print(f"缺少：{', '.join(n for n, _ in env_loader.missing_required())}")
        print("配置后重试：python -m eval.judge_bias")
        return 0    # 不算失败：没配置不是 Judge 的问题

    client = JudgeClient()
    if not client.is_available():
        print("[跳过] Judge 客户端不可用（可能未配置 Judge 模型）")
        return 0

    # 未指定任何开关则全跑
    all_checks = not (args.consistency or args.position or args.length)

    checks: Dict[str, Any] = {}
    if all_checks or args.consistency:
        print("[1/3] 刻度漂移检测...", flush=True)
        checks["consistency"] = run_consistency(client, repeats=args.repeats)
    if all_checks or args.position:
        print("[2/3] 位置偏好检测...", flush=True)
        checks["position_bias"] = run_position_bias(client)
    if all_checks or args.length:
        print("[3/3] 长度偏好检测...", flush=True)
        checks["length_bias"] = run_length_bias(client)

    print()
    print(format_report(checks))

    # 有检出问题时返回非 0，便于 CI 拦截
    for name, c in checks.items():
        if not c or not c.get("available"):
            continue
        if name == "consistency" and not c.get("stable"):
            return 1
        if name in ("position_bias", "length_bias") and c.get("biased"):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
