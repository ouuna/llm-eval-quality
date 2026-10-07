"""
Baseline 对比
---------------------------------
检测"这次比上次退化了"，是质量门禁的重要补充。

为什么需要
----------
绝对阈值只能回答"现在好不好"，回答不了"有没有变差"。
真实项目中，模型换了、Prompt 改了、检索策略调整了，
指标可能从 0.95 掉到 0.91——绝对阈值仍达标，但已经退化了。

对比规则
--------
1. **按方向判断好坏**：幻觉率下降是改善，正确率下降是退化
2. **超过退化阈值才算失败**：小幅波动属正常噪声
3. **必须显式报告退化幅度**，不能只说"有变化"
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASELINE = os.path.join(PROJECT_ROOT, "reports", "baseline.json")


# 指标方向：true 表示越大越好
METRIC_DIRECTION = {
    "overall_pass_rate": True,
    "hallucination_rate": False,
    "refusal_accuracy": True,
    "faithfulness": True,
    "answer_correctness": True,
    "completeness": True,
    "relevance": True,
    "retrieval_hit_rate": True,
    "consistency": True,
    "score_std": False,
    "p95_latency_ms": False,
    "avg_cost": False,
}


@dataclass
class Delta:
    """单个指标的对比结果"""
    name: str
    baseline_value: Optional[float]
    current_value: Optional[float]
    direction_better: str          # "up" / "down"
    regression: bool = False
    magnitude: float = 0.0         # 变化幅度（相对百分比）
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "baseline": self.baseline_value,
            "current": self.current_value,
            "direction": self.direction_better,
            "regression": self.regression,
            "magnitude": round(self.magnitude, 4),
            "description": self.description,
        }


@dataclass
class BaselineComparison:
    """整体对比结果"""
    deltas: List[Delta] = field(default_factory=list)
    has_regression: bool = False
    baseline_source: str = ""
    comparable: bool = True
    incomparable_reason: str = ""

    @property
    def regressions(self) -> List[Delta]:
        return [d for d in self.deltas if d.regression]

    @property
    def improvements(self) -> List[Delta]:
        return [d for d in self.deltas
                if not d.regression and d.magnitude > 0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_regression": self.has_regression,
            "baseline_source": self.baseline_source,
            "comparable": self.comparable,
            "incomparable_reason": self.incomparable_reason,
            "regressions": [d.to_dict() for d in self.regressions],
            "improvements": [d.to_dict() for d in self.improvements],
            "deltas": [d.to_dict() for d in self.deltas],
        }

    def text(self) -> str:
        lines = ["=" * 66, "Baseline Comparison", "=" * 66]

        # 顺序很重要：不可比的判断必须排在「无数据」之前。
        # 反过来的话，mock 基线这种「有文件但无可比指标」的情况
        # 会走进无数据分支，把真正需要看到的提示吞掉。
        if not self.comparable:
            lines.append(f"⚠️ 基线不可比：{self.incomparable_reason}")
            lines.append("")
            lines.append("已跳过逐项对比。本次结果不参与退化判定。")
            lines.append("=" * 66)
            return "\n".join(lines)

        if not self.deltas:
            lines.append("无可对比的基线数据")
            return "\n".join(lines)

        lines.append(f"基线来源：{self.baseline_source}")
        lines.append("")

        header = (f"{'指标':<24}{'基线':>12}{'当前':>12}"
                  f"{'变化':>10}  判定")
        lines.append(header)
        lines.append("-" * 66)

        for d in sorted(self.deltas, key=lambda x: -x.magnitude):
            if d.baseline_value is None or d.current_value is None:
                lines.append(f"{d.name:<24}{'-':>12}{'-':>12}"
                             f"{'-':>10}  数据不足")
                continue

            diff = d.current_value - d.baseline_value
            if d.baseline_value != 0:
                pct = diff / abs(d.baseline_value) * 100
                mtxt = f"{pct:+.1f}%"
            else:
                pct = 0.0
                mtxt = "n/a"

            mark = ("退化" if d.regression
                    else "改善" if d.magnitude > 0 else "持平")
            lines.append(f"{d.name:<24}{d.baseline_value:>12.4g}"
                         f"{d.current_value:>12.4g}{mtxt:>10}  {mark}")

        lines.append("")
        if self.has_regression:
            lines.append(f"⚠️ 检出 {len(self.regressions)} 项退化")
            for d in self.regressions:
                lines.append(f"   {d.name}: {d.description}")
        else:
            lines.append("✓ 无退化")

        lines.append("=" * 66)
        return "\n".join(lines)


def check_comparable(baseline_meta: Dict[str, Any],
                     current_meta: Dict[str, Any]) -> tuple:
    """
    判断基线与当前结果是否可比。

    为什么必须校验
    --------------
    基线对比的前提是「同一个东西的两个版本」。
    如果拿 Mock 跑出来的基线去比真实 API 的结果，
    或者拿 8 条的 smoke 基线去比 27 条的 smoke+full，
    算出来的差值没有任何含义。

    但这类对比照样会输出一张格式整齐、数字漂亮的表格，
    人眼扫过去只会觉得"哦，有对比"，很难意识到它是无效的。
    所以这里宁可拒绝对比，也不产出误导性的数字。
    """
    b_model = (baseline_meta or {}).get("model")
    c_model = (current_meta or {}).get("model")

    b_ds = (baseline_meta or {}).get("dataset")
    c_ds = (current_meta or {}).get("dataset")

    # Mock 与真实模型之间不可比
    if b_model != c_model and "mock" in (str(b_model), str(c_model)):
        return False, (
            f"基线来自 {'Mock' if 'mock' in str(b_model) else 'Mock'}数据"
            f"（model={b_model}），当前为真实 API 调用（model={c_model}）。"
            f"Mock 是人工构造的固定输出，与真实模型输出不可比。"
            f"请先用真实API 跑一次以建立基线。"
        )

    if b_ds != c_ds:
        return False, (
            f"数据集不同：基线为 {b_ds}，当前为 {c_ds}。"
            f"不同数据集的用例难度与覆盖面不同，指标不可直接比较。"
        )

    return True, ""


def compare(current: Dict[str, float],
             baseline: Dict[str, float],
             tolerance: float = 0.05,
             baseline_source: str = "",
             baseline_meta: Dict[str, Any] = None,
             current_meta: Dict[str, Any] = None) -> BaselineComparison:
    """
    对比当前与基线

    参数
    ----
    tolerance  相对变化容忍度，默认 5%
               超过此幅度且方向为劣化 → 判为退化

    baseline_meta / current_meta
        两边的元信息（model / dataset）。提供时会先校验可比性；
        不可比则跳过逐项对比，has_regression 恒为 False。
    """
    comparison = BaselineComparison(baseline_source=baseline_source)

    if baseline_meta is not None and current_meta is not None:
        ok, reason = check_comparable(baseline_meta, current_meta)
        if not ok:
            comparison.comparable = False
            comparison.incomparable_reason = reason
            return comparison

    for name, cur in current.items():
        if not isinstance(cur, (int, float)):
            continue

        base = baseline.get(name)
        if not isinstance(base, (int, float)):
            continue

        better_up = METRIC_DIRECTION.get(name, True)
        diff = cur - base
        magnitude = abs(diff / base * 100) if base else 0.0

        # 方向判定
        if abs(diff) < 1e-12:
            direction = "flat"
        elif (diff > 0) == better_up:
            direction = "up"      # 改善
        else:
            direction = "down"    # 退化方向

        # 超过容忍度且方向为劣化 → 判为退化
        regression = (magnitude > tolerance * 100) and (direction == "down")

        desc = ""
        if regression:
            desc = (f"从 {base:.4g} 变为 {cur:.4g}，"
                    f"变化 {diff:+.4g}（{magnitude:.1f}%），超过容忍度 {tolerance:.0%}")

        comparison.deltas.append(Delta(
            name=name, baseline_value=base, current_value=cur,
            direction_better=direction, regression=regression,
            magnitude=magnitude, description=desc,
        ))

    comparison.has_regression = any(d.regression for d in comparison.deltas)
    return comparison


# ============================================================
# 基线存取
# ============================================================
def save_baseline(metrics: Dict[str, float], path: str = None,
                  meta: Dict[str, Any] = None) -> str:
    """保存当前指标作为基线"""
    path = path or DEFAULT_BASELINE
    os.makedirs(os.path.dirname(path), exist_ok=True)

    payload = {
        "metrics": {k: v for k, v in metrics.items()
                    if isinstance(v, (int, float))},
        "meta": meta or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_baseline(path: str = None) -> Dict[str, Any]:
    """加载基线；不存在返回空"""
    path = path or DEFAULT_BASELINE
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """从报告 JSON 中提取指标"""
    out = {}
    for name, m in (results.get("metrics") or {}).items():
        v = m.get("value") if isinstance(m, dict) else None
        if isinstance(v, (int, float)) and m.get("available", True):
            out[name] = v
    return out


if __name__ == "__main__":
    # 演示
    current = {
        "overall_pass_rate": 0.96, "hallucination_rate": 0.03,
        "faithfulness": 0.91, "answer_correctness": 0.88,
        "completeness": 0.92, "relevance": 0.79,
    }
    baseline = {
        "overall_pass_rate": 0.98, "hallucination_rate": 0.02,
        "faithfulness": 0.95, "answer_correctness": 0.93,
        "completeness": 0.94, "relevance": 0.81,
    }
    cmp = compare(current, baseline, tolerance=0.03,
                  baseline_source="demo")
    print(cmp.text())
