"""
评测器可靠性验证
---------------------------------
用人工标注的 Gold Set 反向测量自动评测器的能力。

回答的问题：**"自动评测器本身是否可靠？"**

核心指标
--------
- Accuracy   整体准确率
- Precision  精确率：判"有幻觉"中真正有幻觉的比例 ← 误报控制
- Recall     召回率：真实有幻觉中被检出的比例 ← **漏报控制，最重要**
- F1         二者调和
- FPR        假阳性率：把无幻觉误判为有幻觉的比例
- FNR        假阴性率：把有幻觉漏判的比例 ← **危险指标**

为什么 Recall 最关键
-------------------
假阳性（FP）导致误报，用户会觉得评测器"太严"；
假阴性（FN）导致漏报，**缺陷会漏到生产环境**。

两者的代价完全不对称，因此必须同时报告，
且门禁应对 FN 设置更严的阈值。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import os

from eval.evaluators.faithfulness import detect_hallucination
from eval.schemas.result import Severity, ViolationType
from eval.datasets.gold_set import GoldSample, load_gold_set

PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 混淆矩阵
# ============================================================
@dataclass
class ConfusionMatrix:
    """二分类混淆矩阵"""
    tp = 0    # 真有幻觉，评测器判有 → 正确检出
    fp = 0    # 实际无幻觉，评测器判有 → 误报
    tn = 0    # 实际无幻觉，评测器判无 → 正确放过
    fn = 0    # 真有幻觉，评测器判无 → **漏报（最危险）**

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def metrics(self) -> Dict[str, Any]:
        t = self.total
        if t == 0:
            return {"usable_samples": 0, "note": "无可用样本，Gold Set 未复核"}

        acc = (self.tp + self.tn) / t
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else None
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else None

        f1 = None
        if precision and recall and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)

        fpr = self.fp / (self.fp + self.tn) if (self.fp + self.tn) else None
        fnr = self.fn / (self.tp + self.fn) if (self.tp + self.fn) else None

        return {
            "usable_samples": t,
            "accuracy": round(acc, 4),
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "fpr": round(fpr, 4) if fpr is not None else None,
            "fnr": round(fnr, 4) if fnr is not None else None,
            "matrix": {
                "tp": self.tp, "fp": self.fp,
                "tn": self.tn, "fn": self.fn,
            },
        }


# ============================================================
# 单样本评测器运行
# ============================================================
def run_evaluator_on_sample(sample: GoldSample,
                            use_semantic: bool = False,
                            semantic_evaluator=None) -> Dict[str, Any]:
    """
    在单个 Gold 样本上运行声明级验证

    返回评测器的判定结果与理由
    """
    result = detect_hallucination(
        answer=sample.answer,
        context=sample.context,
        required_facts=sample.required_facts,
        forbidden_facts=sample.forbidden_facts,
        should_refuse=bool(sample.label_should_refuse),
    )

    out = {
        "id": sample.id,
        "predicted_hallucination": bool(result["has_hallucination"]),
        "reason": result["reason"],
        "coverage": result["coverage"],
        "forbidden_hits": result["forbidden_hits"],
        "unsupported_count": len(result["unsupported_claims"]),
        "missing_required": result.get("missing_required", []),
    }

    if use_semantic and semantic_evaluator is not None and sample.context:
        sem = semantic_evaluator.groundedness(sample.answer, sample.context)
        out["semantic_groundedness"] = (
            sem.get("score") if sem.get("available") else None
        )

    return out


# ============================================================
# 批量验证
# ============================================================
def validate_evaluator(samples: List[GoldSample],
                      use_semantic: bool = False,
                      semantic_evaluator=None) -> Dict[str, Any]:
    """
    用 Gold Set 验证评测器可靠性

    返回完整验证报告
    """
    usable = [s for s in samples if s.is_usable]

    if not usable:
        st = {
            "usable_samples": 0,
            "note": (
                "Gold Set 尚无人工复核样本，无法验证评测器。"
                "这是预期行为——未复核的数据不能作为真值。"
            ),
        }
        return {
            "status": "insufficient_data",
            "metrics": st,
            "false_positives": [],
            "false_negatives": [],
            "per_sample": [],
        }

    cm = ConfusionMatrix()
    per_sample = []
    fp_cases, fn_cases = [], []

    for s in usable:
        pred = run_evaluator_on_sample(s, use_semantic, semantic_evaluator)
        truth = bool(s.label_hallucination)
        got = pred["predicted_hallucination"]

        if truth and got:
            cm.tp += 1
        elif truth and not got:
            cm.fn += 1
            fn_cases.append({**pred, "note": s.note, "answer": s.answer,
                             "category": s.category})
        elif (not truth) and got:
            cm.fp += 1
            fp_cases.append({**pred, "note": s.note, "answer": s.answer,
                             "category": s.category})
        else:
            cm.tn += 1

        per_sample.append({
            **pred,
            "ground_truth": truth,
            "correct": truth == got,
            "category": s.category,
            "note": s.note,
        })

    metrics = cm.metrics()
    m = metrics.get("accuracy")

    return {
        "status": "ok",
        "metrics": metrics,
        "false_positives": fp_cases,     # 误报：无幻觉却判有
        "false_negatives": fn_cases,     # 漏报：有幻觉却判无
        "per_sample": per_sample,
        "verdict": _verdict(metrics),
    }


def _verdict(metrics: Dict[str, Any]) -> str:
    """
    根据指标给出结论

    刻意使用固定阈值，且阈值本身写明，便于后续讨论是否调整
    """
    if not metrics.get("usable_samples"):
        return "无法评估：Gold Set 未复核"

    p, r = metrics.get("precision"), metrics.get("recall")
    fpr, fnr = metrics.get("fpr"), metrics.get("fnr")

    lines = []

    # 漏报是更危险的方向，阈值更严
    if r is not None and r < 0.8:
        lines.append(f"召回率 {r:.0%} 偏低——存在漏报，缺陷可能流入生产")
    if fnr is not None and fnr > 0.2:
        lines.append(f"漏报率 {fnr:.0%} 超过 20%，不可用于门禁拦截")

    if p is not None and p < 0.8:
        lines.append(f"精确率 {p:.0%} 偏低——误报过多，评测器可信度不足")
    if fpr is not None and fpr > 0.3:
        lines.append(f"误报率 {fpr:.0%} 过高，实用价值受限")

    if not lines:
        lines.append("各项指标达标，可考虑用于质量门禁")

    return "；".join(lines)


# ============================================================
# 已知局限分类分析
# ============================================================
def analyze_failure_patterns(report: Dict[str, Any]) -> Dict[str, List[dict]]:
    """
    对误报与漏报做模式归类

    用途：报告不仅要说"漏了3 条"，还要说"漏的都是哪类"
    """
    fn_by_cat: Dict[str, List[dict]] = {}
    fp_by_cat: Dict[str, List[dict]] = {}

    for c in report.get("false_negatives", []):
        cat = c.get("category", "unknown")
        fn_by_cat.setdefault(cat, []).append(c)

    for c in report.get("false_positives", []):
        cat = c.get("category", "unknown")
        fp_by_cat.setdefault(cat, []).append(c)

    return {
        "false_negative_patterns": [
            {"category": k, "count": len(v),
             "ids": [x["id"] for x in v]}
            for k, v in sorted(fn_by_cat.items(), key=lambda x: -len(x[1]))
        ],
        "false_positive_patterns": [
            {"category": k, "count": len(v),
             "ids": [x["id"] for x in v]}
            for k, v in sorted(fp_by_cat.items(), key=lambda x: -len(x[1]))
        ],
    }


# ============================================================
# 报告输出
# ============================================================
def save_validation_report(report: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


def format_validation_report(report: Dict[str, Any],
                             gold_stats: Dict[str, Any] = None) -> str:
    """生成可读的验证报告文本"""
    L = []
    L.append("=" * 66)
    L.append("评测器可靠性验证报告")
    L.append("=" * 66)

    if gold_stats:
        L.append("Gold Set 状态：")
        L.append(f"  样本总数{'':6}{gold_stats['total']}")
        L.append(f"  已人工复核     {gold_stats['reviewed']}"
                 f"（{gold_stats['review_rate']:.0%}）")
        L.append(f"  可用样本       {gold_stats['usable']}")
        L.append(f"  正/负样本      {gold_stats['positive_hallucination']}"
                 f" / {gold_stats['negative_no_hallucination']}")
        L.append("")

    if report["status"] != "ok":
        L.append(f"状态：{report['status']}")
        L.append(f"说明：{report['metrics'].get('note', '')}")
        L.append("")
        L.append("⚠️  必须先完成人工复核，才能得到有意义的验证结果。")
        L.append("=" * 66)
        return "\n".join(L)

    m = report["metrics"]
    L.append("混淆矩阵")
    L.append("  " + "-" * 40)
    L.append(f"  TP（正确检出幻觉）  {m['matrix']['tp']}")
    L.append(f"  FN（漏报·最危险）  {m['matrix']['fn']}")
    L.append(f"  FP（误报）          {m['matrix']['fp']}")
    L.append(f"  TN（正确放过）      {m['matrix']['tn']}")
    L.append("")
    L.append("指标")
    L.append("  " + "-" * 40)
    L.append(f"  Accuracy   {_fmt(m['accuracy'])}")
    L.append(f"  Precision  {_fmt(m['precision'])}")
    L.append(f"  Recall     {_fmt(m['recall'])}   ←漏报控制")
    L.append(f"  F1         {_fmt(m['f1'])}")
    L.append(f"  FPR        {_fmt(m['fpr'])}   ← 误报控制")
    L.append(f"  FNR        {_fmt(m['fnr'])}   ← 漏报控制")
    L.append("")
    L.append(f"结论：{report['verdict']}")
    L.append("")

    patterns = analyze_failure_patterns(report)
    if patterns["false_negative_patterns"]:
        L.append("漏报模式（评测器查不出的缺陷类型）")
        L.append("  " + "-" * 40)
        for p in patterns["false_negative_patterns"]:
            L.append(f"  {p['category']:<20} {p['count']} 条  {', '.join(p['ids'])}")
        L.append("")

    if patterns["false_positive_patterns"]:
        L.append("误报模式（评测器过严的类型）")
        L.append("  " + "-" * 40)
        for p in patterns["false_positive_patterns"]:
            L.append(f"  {p['category']:<20} {p['count']} 条  {', '.join(p['ids'])}")
        L.append("")

    if report["false_negatives"]:
        L.append("漏报样本明细")
        L.append("  " + "-" * 40)
        for c in report["false_negatives"]:
            L.append(f"  [{c['id']}] {c['note'][:50]}")
            L.append(f"      评测器理由：{c['reason'][:60]}")
        L.append("")

    L.append("=" * 66)
    return "\n".join(L)


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "unavailable"
    return f"{v:.1%}"


# ============================================================
# 数据源
# ============================================================

# 人工标注结果的存放位置。
#
# 为什么不直接用代码里的 SAMPLES
# ----------------------------
# gold_set.py 里的 SAMPLES 是自动预填的模板，全部 reviewed=False，
# 按设计不参与验证统计（is_usable 要求 reviewed=True）。
# 真正的人工标注写在 goldset/gold_set.json 里。
#
# 早先这里直接 import SAMPLES，导致用户辛苦完成的标注从未被读取，
# 验证永远返回 insufficient_data —— 而 reports/ 里那份漂亮的准确率
# 是手工作业跑出来的，与代码实际行为不符。
#
# 这类「报告与代码不一致」比直接报错危险得多：
# 它会让评测器看起来已经验证过了。
DEFAULT_GOLD_SET_PATH = os.path.join(
    PROJECT_ROOT, "goldset", "gold_set.json"
)


def resolve_gold_samples(path: str = None) -> tuple:
    """
    返回 (样本列表, 数据来源说明)。

    优先读人工标注的 JSON；文件不存在时才退回内置模板，
    并在来源说明里讲清楚用的是哪一个——不静默降级。
    """
    target = path or DEFAULT_GOLD_SET_PATH

    if not os.path.exists(target):
        from eval.datasets.gold_set import SAMPLES
        return list(SAMPLES), (
            f"内置模板（未找到 {target}）"
            f"——全部未复核，无法作为真值"
        )

    try:
        with open(target, "r", encoding="utf-8") as f:
            samples = [GoldSample.from_dict(d) for d in json.load(f)]
    except (OSError, ValueError, KeyError) as e:
        from eval.datasets.gold_set import SAMPLES
        return list(SAMPLES), (
            f"内置模板（{target} 解析失败：{type(e).__name__}: {e}）"
            f"——全部未复核，无法作为真值"
        )

    reviewers = {s.reviewed_by for s in samples if s.reviewed_by}
    who = "、".join(sorted(reviewers)) if reviewers else "未署名"
    return samples, f"{target}（复核者：{who}）"


# ============================================================
# 标注一致性（Inter-Annotator Agreement, IAA）
# ============================================================
def cohens_kappa(labels_a: List[Any], labels_b: List[Any]) -> Dict[str, Any]:
    """
    Cohen's Kappa —— 两位标注者的一致性指标。

    为什么用 Kappa 而不是一致率
    ---------------------------
    一致率（agreement）有致命缺陷：如果两个人都"随便标 true"，
    在正样本占 90% 的数据上一致率也能高达 90%，
    但这个一致毫无信息量。Kappa 扣除了"碰巧一致"的部分，
    所以基线准确率高的场景下 Kappa 才有意义。

    返回字段
    --------
    kappa          Kappa 值，0~1。无法计算时（见 reason）为 None
    agreement      观察一致率，作为对照
    n              参与计算的样本对数
    reason         无法计算 Kappa 时的原因（正常计算时为空字符串）
    """
    if len(labels_a) != len(labels_b):
        return {
            "kappa": None, "agreement": None, "n": 0,
            "reason": "两标注者样本数不一致，无法配对计算",
        }

    n = len(labels_a)
    if n == 0:
        return {
            "kappa": None, "agreement": None, "n": 0,
            "reason": "空输入，无样本可计算",
        }

    # 单类别输入：所有标注完全相同，一致率=1 但 Kappa 无定义（分母为0）
    # ——这是 Kappa 的经典"悖论"，必须显式说明而非抛 ZeroDivisionError
    unique = sorted(set(labels_a) | set(labels_b))
    if len(unique) < 2:
        return {
            "kappa": None, "agreement": 1.0, "n": n,
            "reason": "单类别输入（所有标注一致），Kappa 无定义（分母为零）",
        }

    # 观察一致率
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = agree / n

    # 期望一致率（各标注者各类别的边缘分布之积求和）
    p_e = 0.0
    for cat in unique:
        pa = labels_a.count(cat) / n
        pb = labels_b.count(cat) / n
        p_e += pa * pb

    if p_e == 1.0:
        # 理论期望一致率=1 时 Kappa 无定义
        return {
            "kappa": None, "agreement": round(p_o, 4), "n": n,
            "reason": "期望一致率已达 1.0，Kappa 无定义",
        }

    kappa = (p_o - p_e) / (1.0 - p_e)
    return {
        "kappa": round(kappa, 4),
        "agreement": round(p_o, 4),
        "n": n,
        "reason": "",
    }


def compare_annotations(path_a: str, path_b: str) -> Dict[str, Any]:
    """
    对比两位标注者的 Gold Set 标注文件，输出差异清单。

    用途：解决 Gold Set 循环论证（P0-2）的关键工具——
    有了第二位标注者的独立结果，才能计算一致性、
    发现"评测器规则是否为这批样本量身定制"。

    返回字段
    --------
    status           ok / error（某个文件读不到时）
    total_a / total_b 两个文件的样本数
    agreed           一致数
    disagreed        不一致数
    kappa            基于 label_hallucination 的 Cohen's Kappa
    differences      不一致样本明细（题目、两个标签、各自理由）
    error            出错原因（status=error 时）
    """
    samples_a = load_gold_set(path_a)
    samples_b = load_gold_set(path_b)

    if samples_a is None or samples_b is None:
        missing = []
        if samples_a is None:
            missing.append(path_a)
        if samples_b is None:
            missing.append(path_b)
        return {
            "status": "error",
            "error": f"标注文件读不到或解析失败：{'、'.join(missing)}",
            "total_a": 0, "total_b": 0,
            "agreed": 0, "disagreed": 0,
            "kappa": None,
            "differences": [],
        }

    map_a = {s.id: s for s in samples_a}
    map_b = {s.id: s for s in samples_b}

    common_ids = [sid for sid in map_a if sid in map_b]

    differences = []
    labels_a, labels_b = [], []
    for sid in common_ids:
        sa, sb = map_a[sid], map_b[sid]
        a_label = sa.label_hallucination
        b_label = sb.label_hallucination
        labels_a.append(a_label)
        labels_b.append(b_label)

        if a_label != b_label:
            differences.append({
                "id": sid,
                "question": sa.question,
                "label_a": a_label,
                "label_b": b_label,
                "note_a": sa.note,
                "note_b": sb.note,
            })

    agreed = len(common_ids) - len(differences)

    # 只对双方都给了标签的样本算 Kappa（None 表示未标注，跳过）
    paired_a, paired_b = [], []
    for a, b in zip(labels_a, labels_b):
        if a is not None and b is not None:
            paired_a.append(a)
            paired_b.append(b)
    kappa = cohens_kappa(paired_a, paired_b)

    return {
        "status": "ok",
        "total_a": len(samples_a),
        "total_b": len(samples_b),
        "agreed": agreed,
        "disagreed": len(differences),
        "kappa": kappa,
        "differences": differences,
    }


if __name__ == "__main__":
    from eval.datasets.gold_set import statistics as gold_stats_fn

    gold_samples, source = resolve_gold_samples()

    print("=" * 66)
    print("数据源")
    print("=" * 66)
    print(f"  {source}")

    stats = gold_stats_fn(gold_samples)
    report = validate_evaluator(gold_samples)
    report["data_source"] = source

    print(format_validation_report(report, stats))
