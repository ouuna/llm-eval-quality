"""
Quality Gate —— 分层质量门禁
---------------------------------
把指标阈值与判定逻辑从 pytest 测试中抽离，成为独立组件。

为什么要独立
------------
1. **多入口复用**：CLI、pytest、CI、Dashboard 都需要同一套门禁逻辑
2. **判定原因要可读**：门禁失败必须说清"哪个指标失败、当前值多少、
   阈值多少、哪些样本失败、为什么失败"
3. **可配置**：阈值放 YAML，不同项目不同标准
4. **分层**：不同严重级别的门禁可独立启用

门禁分层
--------
    critical   能力底线，不达标必须阻断（幻觉、拒答、正确性）
    quality    质量目标，允许小幅不达标
    performance 性能与成本
    stability  稳定性与回归

核心原则
--------
**门禁失败必须给出可执行的诊断信息，而不是"不达标"三个字。**
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import os
import sys
import json

# 复用已有的零依赖 YAML 解析器（PyYAML 存在时其会自动优先使用）
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tests"))
from config_loader import _load_yaml_text  # noqa: E402


# ============================================================
# 配置
# ============================================================
DEFAULT_GATE_CONFIG = {
    "critical": {
        "hallucination_rate": {"max": 0.05,
                        "desc": "幻觉率必须低于此值"},
        "refusal_accuracy": {"min": 0.95,
                       "desc": "域外拒答率必须高于此值"},
        "faithfulness": {"min": 0.90,
                    "desc": "有据性必须高于此值"},
        "answer_correctness": {"min": 0.90,
                         "desc": "答案正确性必须高于此值"},
        "regression_pass_rate": {"min": 1.00,
                           "desc": "回归用例必须 100% 通过"},
    },
    "quality": {
        "overall_pass_rate": {"min": 0.95, "desc": "总体通过率"},
        "completeness": {"min": 0.90, "desc": "回答完整率"},
        "relevance": {"min": 0.60, "desc": "答案相关性"},
        "retrieval_hit_rate": {"min": 0.60, "desc": "检索命中率"},
    },
    "performance": {
        "p95_latency_ms": {"max": 10000, "desc": "P95 延迟上限"},
        "avg_cost": {"max": 0.5, "desc": "平均成本上限（单位元）"},
    },
    "stability": {
        "consistency": {"min": 0.85, "desc": "输出一致性"},
        "score_std": {"max": 0.15, "desc": "分数波动上限"},
    },
}


def load_gate_config(path: str = None) -> Dict[str, Any]:
    """加载门禁配置；文件不存在则用默认"""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return _load_yaml_text(f.read())
    return DEFAULT_GATE_CONFIG


# ============================================================
# 单项检查结果
# ============================================================
@dataclass
class GateCheck:
    """单条门禁检查结果"""
    name: str
    layer: str
    passed: bool
    current_value: Optional[float]
    threshold_value: Optional[float]
    comparison: str                # "min" / "max"
    description: str = ""
    failed_samples: List[str] = field(default_factory=list)
    sample_count: int = 0
    available: bool = True         # 指标是否可计算
    skip_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "layer": self.layer,
            "passed": self.passed,
            "current_value": self.current_value,
            "threshold": self.threshold_value,
            "comparison": self.comparison,
            "description": self.description,
            "failed_samples": self.failed_samples[:10],
            "failed_sample_count": len(self.failed_samples),
            "sample_count": self.sample_count,
            "available": self.available,
            "skip_reason": self.skip_reason,
            "message": self.message,
        }

    @property
    def message(self) -> str:
        """人类可读的诊断信息"""
        if not self.available:
            return f"[{self.name}] 跳过：{self.skip_reason}"

        if self.passed:
            return (f"[{self.name}] 通过：{self.current_value:.4f} "
                    f"{self._op()} {self.threshold_value}")

        base = (f"[{self.name}] 未达标：{self.current_value:.4f} "
                f"{self._op()} {self.threshold_value}")
        if self.failed_samples:
            samples = ", ".join(self.failed_samples[:5])
            more = (f" 等 {len(self.failed_samples)} 个"
                    if len(self.failed_samples) > 5 else "")
            base += f"\n失败样本：{samples}{more}"
        return base

    def _op(self) -> str:
        return ">=" if self.comparison == "min" else "<="


# ============================================================
# 门禁评估器
# ============================================================
class QualityGate:
    """
    分层质量门禁

    用法
    ----
    gate = QualityGate()
    report = gate.evaluate(metrics, failed_samples_by_metric)
    if not report.passed:
        for check in report.failed_checks:
            print(check.message)
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEFAULT_GATE_CONFIG

    def evaluate(self,
                 metrics: Dict[str, Any],
                 failed_samples: Dict[str, List[str]] = None,
                 enabled_layers: List[str] = None) -> "GateReport":
        """
        执行门禁评估

        参数
        ----
        metrics            指标名 → 数值（None 表示不可用）
        failed_samples     指标名 → 失败样本 id 列表
        enabled_layers     启用的层，默认全部
        """
        failed_samples = failed_samples or {}
        layers = enabled_layers or list(self.config.keys())

        checks: List[GateCheck] = []

        for layer in layers:
            rules = self.config.get(layer, {})
            for name, rule in rules.items():
                checks.append(self._check(name, layer, rule,
                                          metrics, failed_samples))

        return GateReport(checks, metrics)

    def _check(self, name: str, layer: str, rule: Dict[str, Any],
               metrics: Dict[str, Any],
               failed_samples: Dict[str, List[str]]) -> GateCheck:
        # 找min 还是 max
        if "min" in rule:
            threshold, comparison = rule["min"], "min"
        elif "max" in rule:
            threshold, comparison = rule["max"], "max"
        else:
            threshold, comparison = None, "min"

        value = metrics.get(name)

        # 指标不可用
        if value is None:
            return GateCheck(
                name=name, layer=layer, passed=True, current_value=None,
                threshold_value=threshold, comparison=comparison,
                description=rule.get("desc", ""),
                available=False,
                skip_reason=("指标未提供或不可用（按需配置此项门禁）"
                             if threshold is not None else "无阈值配置"),
                sample_count=0,
            )

        if comparison == "min":
            passed = value >= threshold
        else:
            passed = value <= threshold

        return GateCheck(
            name=name, layer=layer, passed=passed, current_value=value,
            threshold_value=threshold, comparison=comparison,
            description=rule.get("desc", ""),
            failed_samples=failed_samples.get(name, []),
            sample_count=metrics.get(f"{name}_count", 0),
        )


# ============================================================
# 门禁报告
# ============================================================
@dataclass
class GateReport:
    checks: List[GateCheck]
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> List[GateCheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def skipped_checks(self) -> List[GateCheck]:
        return [c for c in self.checks if not c.available]

    def by_layer(self) -> Dict[str, List[GateCheck]]:
        out: Dict[str, List[GateCheck]] = {}
        for c in self.checks:
            out.setdefault(c.layer, []).append(c)
        return out

    def summary(self) -> Dict[str, Any]:
        total = len(self.checks)
        failed = len(self.failed_checks)
        skipped = len(self.skipped_checks)
        return {
            "passed": self.passed,
            "total_checks": total,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": round((total - failed - skipped) / total, 4)
            if total else 0,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "checks": [c.to_dict() for c in self.checks],
            "metrics": self.metrics,
        }

    def text(self) -> str:
        """生成可读的诊断报告"""
        lines = []
        s = self.summary()
        lines.append("=" * 68)
        lines.append("Quality Gate")
        lines.append("=" * 68)
        lines.append(f"结论：{'PASSED' if self.passed else 'FAILED'}")
        lines.append(f"检查项：{s['total_checks']}  "
                     f"失败 {s['failed']}  跳过 {s['skipped']}")
        lines.append("")

        for layer, checks in self.by_layer().items():
            layer_failed = sum(1 for c in checks if not c.passed)
            mark = "✗" if layer_failed else "✓"
            lines.append(f"  {mark} {layer}")
            for c in checks:
                if c.available:
                    icon = "  ✓" if c.passed else "  ✗"
                    lines.append(f"  {icon} {c.message}")
                else:
                    lines.append(f"  - {c.name}（跳过）")
            lines.append("")

        if self.failed_checks:
            lines.append("=" * 68)
            lines.append("失败详情与修复方向")
            lines.append("=" * 68)
            directions = {
                "hallucination_rate": "加强 Prompt 防幻觉约束；"
                                "检查是否有 forbidden_facts 覆盖",
                "refusal_accuracy": "补充域外用例；确认 Prompt 含拒答指令",
                "faithfulness": "检查检索召回率；提升上下文相关性",
                "answer_correctness": "完善 required_facts 标注；"
                                "检查被测系统 Prompt",
                "regression_pass_rate": "查看失败用例，可能是代码回退",
                "completeness": "补充 key_points 标注；检查是否只答了一半",
                "relevance": "检查检索是否召回正确片段",
                "retrieval_hit_rate": "扩充知识库；改进检索策略",
                "consistency": "降低 temperature；检查是否存在随机性",
            }
            for c in self.failed_checks:
                d = directions.get(c.name, "查看对应指标定义与实现")
                lines.append(f"  {c.name}: {d}")
            lines.append("=" * 68)

        return "\n".join(lines)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path


if __name__ == "__main__":
    # 演示：构造一组指标并评估
    gate = QualityGate()

    demo_metrics = {
        "hallucination_rate": 0.03,
        "refusal_accuracy": 0.98,
        "faithfulness": 0.92,
        "answer_correctness": 0.85,     # 故意不达标
        "regression_pass_rate": 1.0,
        "overall_pass_rate": 0.91,      # 故意不达标
        "completeness": 0.95,
        "relevance": 0.78,
        "retrieval_hit_rate": 0.68,
        "p95_latency_ms": 3200,
        "avg_cost": 0.012,
        "consistency": 0.95,
        "score_std": 0.05,
    }

    failed = {
        "answer_correctness": ["case-03", "case-07", "case-12"],
        "overall_pass_rate": ["case-03", "case-07", "case-12", "case-19"],
    }

    report = gate.evaluate(demo_metrics, failed)
    print(report.text())
