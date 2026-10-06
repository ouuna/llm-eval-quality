"""
统一评测结果 Schema
---------------------------------
所有 evaluator 必须输出本模块定义的 CaseResult 与 EvalReport，
以保证 CLI / JSON / HTML / CI / Dashboard 使用同一套结果结构。

核心设计
--------
1. **status 三态而非两态**
   passed / failed / error

   这是修复 P0-1 的关键设计。旧实现只有"通过/不通过"两态，
   导致 API 调用失败时 actual_output 为空串，
   而 check_hallucination 首行 `if not output.strip(): return True`
   把空串判为"无幻觉"→ 误判为通过。

   新设计：调用失败必须标记为 error，且 error 状态永远不计入 passed。

2. **violations 结构化**
   每条违规记录 violation_type / severity / message / evidence。
   violation_type 对应缺陷分类（RETRIEVAL_ERROR / HALLUCINATION / ...），
   用于生成 Pareto 缺陷统计。

3. **evidence 与 violations 分离**
   evidence 是"支撑判定的原材料"（如引用的上下文片段、命中的事实），
   violations 是"判定结论"。分开便于报告展示与问题追溯。

4. **不可用指标显式标注**
   token_usage / cost 在 API 未返回时记为 None，而非填 0。
   填 0 会让 P95 延迟、均值成本等统计失真。
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum
import json
import time


# ============================================================
# 枚举定义
# ============================================================
class Status(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"      # 评测过程本身出错，与 SUT 质量无关


class Severity(str, Enum):
    CRITICAL = "critical"   # 阻断发布
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ViolationType(str, Enum):
    """缺陷分类，用于 Pareto 统计"""
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"           # 检索未召回或召回错误
    HALLUCINATION = "HALLUCINATION"               # 包含无依据内容
    INCOMPLETE_ANSWER = "INCOMPLETE_ANSWER"       # 遗漏关键要点
    WRONG_ANSWER = "WRONG_ANSWER"                 # 答案与参考不符
    REFUSAL_ERROR = "REFUSAL_ERROR"               # 该拒答却作答 / 不该拒答却拒答
    CONTEXT_CONFLICT = "CONTEXT_CONFLICT"         # 未识别上下文矛盾
    PROMPT_INJECTION = "PROMPT_INJECTION"         # 越权诱导被遵从
    IRRELEVANT = "IRRELEVANT"                     # 答非所问
    LATENCY_REGRESSION = "LATENCY_REGRESSION"     # 延迟超标
    COST_REGRESSION = "COST_REGRESSION"           # 成本超标
    EVALUATOR_ERROR = "EVALUATOR_ERROR"           # 评测器自身异常
    SUT_ERROR = "SUT_ERROR"                       # 被测系统报错


# 状态与违规的映射规则
FAILING_VIOLATIONS = {
    ViolationType.RETRIEVAL_ERROR,
    ViolationType.HALLUCINATION,
    ViolationType.INCOMPLETE_ANSWER,
    ViolationType.WRONG_ANSWER,
    ViolationType.REFUSAL_ERROR,
    ViolationType.CONTEXT_CONFLICT,
    ViolationType.PROMPT_INJECTION,
    ViolationType.IRRELEVANT,
    ViolationType.LATENCY_REGRESSION,
    ViolationType.COST_REGRESSION,
}

ERROR_VIOLATIONS = {
    ViolationType.EVALUATOR_ERROR,
    ViolationType.SUT_ERROR,
}


# ============================================================
# 数据结构
# ============================================================
@dataclass
class Violation:
    """一条违规记录"""
    type: str
    severity: str
    message: str
    evidence: List[str] = field(default_factory=list)
    # 判定所依据的中间量，便于追溯
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.severity not in {s.value for s in Severity}:
            raise ValueError(f"非法 severity: {self.severity}")
        if self.type not in {v.value for v in ViolationType}:
            raise ValueError(f"非法 violation type: {self.type}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenUsage:
    """
    Token 用量与成本

    设计要点：API 未返回时各字段为 None，绝不填 0。
    填 0 会让"平均成本""总 token"等统计失真，
    使报告看起来比实际便宜，违反"不要伪造数据"的原则。
    """
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None          # 估算成本，单位 CNY
    available: bool = False              # API 是否提供了用量信息

    @classmethod
    def unavailable(cls) -> "TokenUsage":
        return cls(available=False)

    @classmethod
    def from_api(cls, usage: Optional[dict], cost_per_1k: Optional[float] = None):
        """从 API 返回的 usage 字段构造"""
        if not usage:
            return cls.unavailable()

        it = usage.get("prompt_tokens")
        ot = usage.get("completion_tokens")
        tt = usage.get("total_tokens") or ((it or 0) + (ot or 0))

        cost = None
        if cost_per_1k is not None and tt:
            cost = round(tt / 1000 * cost_per_1k, 6)

        return cls(
            input_tokens=it,
            output_tokens=ot,
            total_tokens=tt,
            cost=cost,
            available=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CaseResult:
    """
    单条用例的评测结果

    字段对应需求文档第十六节
    """
    case_id: str
    status: str = Status.PASSED.value
    scores: Dict[str, float] = field(default_factory=dict)
    violations: List[Violation] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    latency_ms: int = 0
    token_usage: TokenUsage = field(default_factory=TokenUsage.unavailable)
    error: Optional[str] = None

    # ---- 以下为可追溯信息，不参与门禁判定 ----
    category: str = ""
    question: str = ""
    expected_behavior: str = ""
    actual_output: str = ""
    reference_answer: Optional[str] = None
    retrieval_context: List[str] = field(default_factory=list)
    difficulty: str = "medium"
    tags: List[str] = field(default_factory=list)
    evaluators_used: List[str] = field(default_factory=list)
    injected_defect: Optional[str] = None

    # ---------- 构造方法 ----------
    @classmethod
    def passed(cls, case_id: str, **kwargs) -> "CaseResult":
        return cls(case_id=case_id, status=Status.PASSED.value, **kwargs)

    @classmethod
    def failed(cls, case_id: str, violations: List[Violation], **kwargs) -> "CaseResult":
        return cls(
            case_id=case_id, status=Status.FAILED.value,
            violations=violations, **kwargs
        )

    @classmethod
    def error_result(cls, case_id: str, error: str,
                     violation_type: str = ViolationType.SUT_ERROR.value, **kwargs) -> "CaseResult":
        """
        评测过程出错

        这是修复 P0-1 的核心：API 失败不再是"空输出判为无幻觉"，
        而是显式标记为 error 状态，在门禁中必然失败。
        """
        v = Violation(
            type=violation_type,
            severity=Severity.CRITICAL.value,
            message=error,
        )
        return cls(
            case_id=case_id, status=Status.ERROR.value,
            violations=[v], error=error, **kwargs
        )

    # ---------- 判定逻辑 ----------
    @property
    def is_passed(self) -> bool:
        return self.status == Status.PASSED.value

    @property
    def is_failed(self) -> bool:
        return self.status == Status.FAILED.value

    @property
    def is_error(self) -> bool:
        return self.status == Status.ERROR.value

    @property
    def critical_violations(self) -> List[Violation]:
        return [v for v in self.violations if v.severity == Severity.CRITICAL.value]

    def violation_types(self) -> List[str]:
        return [v.type for v in self.violations]

    def add_violation(self, vtype: str, message: str,
                      severity: str = Severity.HIGH.value,
                      evidence: Optional[List[str]] = None,
                      **details):
        """追加违规并自动将状态置为 failed"""
        v = Violation(
            type=vtype, severity=severity, message=message,
            evidence=evidence or [], details=details
        )
        self.violations.append(v)
        if self.status == Status.PASSED.value:
            self.status = Status.FAILED.value
        return v

    def to_dict(self) -> Dict[str, Any]:
        """
        显式序列化，不使用 dataclasses.asdict。

        原因：asdict 会递归遍历所有属性，包括 @property 定义的
        is_passed / is_error / critical_violations 等方法对象，
        导致 json.dumps 时报 "Object of type method is not JSON serializable"。
        """
        return {
            "case_id": self.case_id,
            "status": self.status,
            "scores": dict(self.scores),
            "violations": [v.to_dict() for v in self.violations],
            "evidence": list(self.evidence),
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage.to_dict(),
            "error": self.error,
            "category": self.category,
            "question": self.question,
            "expected_behavior": self.expected_behavior,
            "actual_output": self.actual_output,
            "reference_answer": self.reference_answer,
            "retrieval_context": list(self.retrieval_context),
            "difficulty": self.difficulty,
            "tags": list(self.tags),
            "evaluators_used": list(self.evaluators_used),
            "injected_defect": self.injected_defect,
        }


@dataclass
class MetricSummary:
    """单个指标的汇总"""
    name: str
    value: float
    available: bool = True
    definition: str = ""                # 指标定义，写入报告
    caveat: str = ""                    # 已知误差来源
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    passed: Optional[bool] = None       # None 表示该指标未设门禁

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalReport:
    """整体评测报告"""
    dataset_name: str
    dataset_tier: str
    model: str
    provider: str = "openai_compatible"
    started_at: str = ""
    duration_sec: float = 0.0

    cases: List[CaseResult] = field(default_factory=list)
    metrics: Dict[str, MetricSummary] = field(default_factory=dict)

    # 门禁结论
    gate_passed: bool = True
    gate_failures: List[Dict[str, Any]] = field(default_factory=list)
    gate_message: str = ""

    # 运行元信息
    evaluators: List[str] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    # 运行级错误文本（如配置缺失、模型不可用）
    # 注意：与下面的 errors property（用例错误计数）不同，故命名区分
    run_errors: List[str] = field(default_factory=list)

    # ---------- 统计 ----------
    @property
    def total(self) -> int:
        return len(self.cases)

    def count(self, status: str) -> int:
        return sum(1 for c in self.cases if c.status == status)

    @property
    def passed(self) -> int:
        return self.count(Status.PASSED.value)

    @property
    def failed(self) -> int:
        return self.count(Status.FAILED.value)

    @property
    def errors(self) -> int:
        return self.count(Status.ERROR.value)

    @property
    def pass_rate(self) -> float:
        """通过率。error 不计入分母 —— 错误不是"质量表现"而是"评测未完成" """
        valid = self.passed + self.failed
        return round(self.passed / valid, 4) if valid else 0.0

    def latency_stats(self) -> Dict[str, Optional[float]]:
        """延迟统计。全部不可用时返回 None 而非 0"""
        lats = [c.latency_ms for c in self.cases if c.latency_ms > 0]
        if not lats:
            return {"count": 0, "mean": None, "p50": None,
                    "p95": None, "max": None, "available": False}

        s = sorted(lats)
        n = len(s)

        def pct(p: float) -> float:
            if n == 1:
                return float(s[0])
            idx = min(n - 1, int(round((p / 100) * (n - 1))))
            return float(s[idx])

        return {
            "count": n,
            "mean": round(sum(s) / n, 1),
            "p50": pct(50),
            "p95": pct(95),
            "max": float(s[-1]),
            "available": True,
        }

    def token_stats(self) -> Dict[str, Any]:
        """Token 与成本统计。API 未提供时 available=False"""
        us = [c.token_usage for c in self.cases if c.token_usage.available]
        if not us:
            return {"available": False, "total_tokens": None,
                    "avg_tokens": None, "total_cost": None, "avg_cost": None}

        totals = [u.total_tokens for u in us if u.total_tokens is not None]
        costs = [u.cost for u in us if u.cost is not None]

        return {
            "available": True,
            "count": len(us),
            "total_tokens": sum(totals) if totals else None,
            "avg_tokens": round(sum(totals) / len(totals), 1) if totals else None,
            "total_cost": round(sum(costs), 6) if costs else None,
            "avg_cost": round(sum(costs) / len(costs), 6) if costs else None,
        }

    def failure_pareto(self) -> List[Dict[str, Any]]:
        """缺陷 Pareto 统计：按类型计数与占比，降序"""
        counter: Dict[str, int] = {}
        for c in self.cases:
            for vt in c.violation_types():
                counter[vt] = counter.get(vt, 0) + 1

        total_v = sum(counter.values()) or 1
        rows = [
            {"type": k, "count": v, "rate": round(v / total_v, 4)}
            for k, v in sorted(counter.items(), key=lambda x: -x[1])
        ]
        cum = 0
        for r in rows:
            cum += r["rate"]
            r["cumulative"] = round(cum, 4)
        return rows

    def category_stats(self) -> List[Dict[str, Any]]:
        """按类别统计"""
        out = []
        cats = sorted({c.category for c in self.cases if c.category})
        for cat in cats:
            sub = [c for c in self.cases if c.category == cat]
            p = sum(1 for c in sub if c.is_passed)
            f = sum(1 for c in sub if c.is_failed)
            e = sum(1 for c in sub if c.is_error)
            out.append({
                "category": cat,
                "total": len(sub),
                "passed": p, "failed": f, "error": e,
                "pass_rate": round(p / (p + f), 4) if (p + f) else 0.0,
            })
        return out

    def summary(self) -> Dict[str, Any]:
        """汇总统计"""
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "error": self.errors,
            "pass_rate": self.pass_rate,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": {
                "dataset": self.dataset_name,
                "tier": self.dataset_tier,
                "model": self.model,
                "provider": self.provider,
                "started_at": self.started_at,
                "duration_sec": self.duration_sec,
                "evaluators": self.evaluators,
            },
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "error": self.errors,
                "pass_rate": self.pass_rate,
            },
            "latency": self.latency_stats(),
            "token_usage": self.token_stats(),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "gate": {
                "passed": self.gate_passed,
                "failures": self.gate_failures,
                "message": self.gate_message,
            },
            "failure_pareto": self.failure_pareto(),
            "category_stats": self.category_stats(),
            "run_errors": self.run_errors,
            "cases": [c.to_dict() for c in self.cases],
        }

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    # ---------- 文本摘要 ----------
    def text(self) -> str:
        """别名：与 GateReport.text() 命名保持一致"""
        return self.summary_text()

    def summary_text(self) -> str:
        """CLI 使用的文本摘要"""
        lines = []
        lines.append("=" * 62)
        lines.append("Evaluation Summary")
        lines.append("=" * 62)
        lines.append(f"Dataset: {self.dataset_name} ({self.dataset_tier})")
        lines.append(f"Model:   {self.model}")
        lines.append(f"Time:    {self.started_at}")
        lines.append("")
        lines.append(f"Cases:   {self.total}")
        lines.append(f"Passed:  {self.passed}")
        lines.append(f"Failed:  {self.failed}")
        lines.append(f"Errors:  {self.errors}")
        lines.append(f"Pass Rate: {self.pass_rate:.2%}")
        lines.append("")

        if self.metrics:
            lines.append("-" * 62)
            lines.append("Metrics")
            lines.append("-" * 62)
            for name, m in self.metrics.items():
                if not m.available:
                    lines.append(f"  {name:24} unavailable")
                    continue
                mark = "" if m.passed is None else ("  [PASS]" if m.passed else "  [FAIL]")
                lines.append(f"  {name:24} {m.value}{mark}")
            lines.append("")

        lat = self.latency_stats()
        if lat["available"]:
            lines.append("-" * 62)
            lines.append("Performance")
            lines.append("-" * 62)
            lines.append(f"  Latency mean {lat['mean']:.0f}ms"
                         f"  P50 {lat['p50']:.0f}ms  P95 {lat['p95']:.0f}ms")
        else:
            lines.append("  Latency: unavailable")

        tok = self.token_stats()
        if tok["available"]:
            lines.append(f"  Tokens total {tok['total_tokens']}"
                         f"  avg {tok['avg_tokens']}")
        else:
            lines.append("  Tokens: unavailable")
        lines.append("")

        pareto = self.failure_pareto()
        if pareto:
            lines.append("-" * 62)
            lines.append("Failure Pareto")
            lines.append("-" * 62)
            for r in pareto:
                lines.append(f"  {r['type']:24} {r['count']:>3}"
                             f"  {r['rate']:>6.1%}  (cum {r['cumulative']:.0%})")
            lines.append("")

        lines.append("-" * 62)
        lines.append(f"Quality Gate: {'PASSED' if self.gate_passed else 'FAILED'}")
        if not self.gate_passed:
            lines.append("Reason:")
            for f_ in self.gate_failures:
                lines.append(f"  - {f_.get('message', f_)}")
        lines.append("=" * 62)
        return "\n".join(lines)


# ============================================================
# 辅助函数
# ============================================================
def make_error_result(case_id: str, error: str,
                      violation_type: str = ViolationType.SUT_ERROR.value,
                      **kwargs) -> CaseResult:
    return CaseResult.error_result(case_id, error, violation_type, **kwargs)


def validate_result(r: CaseResult) -> List[str]:
    """
    校验结果对象的自洽性

    捕获的不一致情况正是"评测器自身缺陷"的信号：
      - status=passed 但有 FAILING 类违规
      - status=failed 但无违规
      - status=error 但 error 字段为空
    """
    issues = []
    failing = {v.value for v in FAILING_VIOLATIONS}
    erroring = {v.value for v in ERROR_VIOLATIONS}

    has_failing = any(v.type in failing for v in r.violations)
    has_erroring = any(v.type in erroring for v in r.violations)

    if r.is_passed and has_failing:
        issues.append(f"[{r.case_id}] status=passed 但存在 FAILING 违规")

    if r.is_passed and has_erroring:
        issues.append(f"[{r.case_id}] status=passed 但存在 ERROR 违规")

    if r.is_failed and not has_failing:
        issues.append(f"[{r.case_id}] status=failed 但无 FAILING 类违规")

    if r.is_error and not r.error:
        issues.append(f"[{r.case_id}] status=error 但 error 字段为空")

    if r.is_passed and r.error:
        issues.append(f"[{r.case_id}] status=passed 但存在 error 字段")

    return issues
