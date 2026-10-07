"""
评测器统一返回结构
--------------------------------
为什么需要这一层
----------------
改之前，五个评测函数返回**三种互不兼容的结构**：

    evaluate_correctness  → dataclass（score / is_correct / reason / method）
    evaluate_completeness → dict   （score / is_complete / missing_facts）
    evaluate_relevance    → dict   （score / is_relevant / method）
    evaluate_refusal      → dict   （is_correct / refused，**没有 score**）
    detect_hallucination  → dict   （has_hallucination / claims / coverage）

问题很实际：
  · 调用方要写 `if r.is_correct` 还是 `if r["is_relevant"]`？—— 只能逐个记
  · 四个函数里只有三个有 `score`，无法统一聚合
  · 一个都有 `evaluator` 字段，报告里说不清这个分数是谁给的
  · 「可插拔 Evaluator」这个需求无法实现——
    因为换个实现就要改所有调用方的字段名

所以定义一个统一的 `EvalResult`，所有评测器都返回它。
**返回统一结构不等于信息变少**：
各评测器特有的字段（claims、missing_facts、variants…）
统一放进 `details` 子字典，按需取用。

字段设计
--------
    score      0~1 的分数，None 表示不可用
    passed     通过与否，None 表示不可判定
    reason     人类可读的判定理由（必须能回答「为什么」）
    evaluator  哪个评测器给的这份结果
    details    该评测器特有的补充信息

`score=None` 与 `score=0` 的区别很重要：
前者是「没测到」，后者是「测了，结果是 0」。
把两者混成0，会让门禁把「无法判定」当成「质量完美」。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalResult:
    """
    所有评测器的统一返回结构

    用法
    ----
        r = evaluate_correctness(answer, ...)
        r.score        # 0~1 或 None
        r.passed       # True / False / None
        r.reason       # 判定理由
        r.details      # 评测器特有的补充信息
    """

    score: Optional[float] = None
    passed: Optional[bool] = None
    reason: str = ""
    evaluator: str = ""

    # 补充信息。刻意用 dict 而不是一堆固定字段——
    # 不同评测器需要的东西差异很大，
    # 硬塞成固定字段只会得到几十个大部分为 None 的字段。
    details: Dict[str, Any] = field(default_factory=dict)

    # ---- 构造辅助 ----
    @classmethod
    def ok(cls, score, passed, reason, evaluator, **details):
        """标准构造"""
        return cls(score=score, passed=passed, reason=reason,
                   evaluator=evaluator, details=details)

    @classmethod
    def unavailable(cls, reason, evaluator):
        """
        不可用。

        刻意与 `ok(score=0)` 区分：
        「没测到」和「测了是 0」必须能分辨。
        """
        return cls(score=None, passed=None, reason=reason,
                   evaluator=evaluator, details={"available": False})

    # ---- 便捷访问 ----
    @property
    def available(self) -> bool:
        return self.score is not None

    @property
    def method(self) -> str:
        """判定方法（aliases / required_facts / ...）"""
        return self.details.get("method", "")

    @property
    def violations(self) -> List[str]:
        """该结果附带的违规标记"""
        return self.details.get("violations", [])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "reason": self.reason,
            "evaluator": self.evaluator,
            "available": self.available,
            "details": self.details,
        }

    def __str__(self):
        s = "unavailable" if self.score is None else f"{self.score:.3f}"
        mark = "✓" if self.passed else ("✗" if self.passed is False else "-")
        return (f"[{self.evaluator}] {mark} {s}  {self.reason[:60]}")


# 便于测试与文档：列出所有评测器的标准名称
EVALUATOR_NAMES = (
    "correctness",
    "completeness",
    "relevance",
    "refusal",
    "faithfulness",
    "semantic",
    "judge",
)


def to_result(raw: Any, evaluator: str) -> EvalResult:
    """
    把既有评测器的返回值包装成 EvalResult。

    用途：分步迁移。
    新的评测器直接返回 EvalResult；
    尚未改造的用这个函数包一层，调用方拿到的结构就一致了。

    这样做比重写所有评测器风险低——
    逐个验证每一步，测试随时能跑。
    """
    if isinstance(raw, EvalResult):
        # 补上 evaluator 字段（若原调用没填）
        if not raw.evaluator:
            raw.evaluator = evaluator
        return raw

    if isinstance(raw, dict):
        # 抽走核心字段，**其余全部进 details**。
        #
        # 早先把 method 也排除了，理由是「它不属于核心字段」——
        # 但 EvalResult.method 属性恰恰从 details["method"] 取值，
        # 排除之后 r.method 永远是空字符串。
        # 结果是报告里所有指标的「判定方法」全部消失，
        # 而这个字段恰恰是排查问题最需要的信息。
        core = ("score", "passed", "reason", "available", "evaluator")
        details = {k: v for k, v in raw.items() if k not in core}
        return EvalResult(
            score=raw.get("score"),
            passed=_extract_passed(raw),
            reason=raw.get("reason", ""),
            evaluator=raw.get("evaluator") or evaluator,
            details=details,
        )

    # dataclass 等对象
    details = {}
    if hasattr(raw, "to_dict"):
        details = dict(raw.to_dict())
    elif hasattr(raw, "__dict__"):
        details = {k: v for k, v in vars(raw).items()
                   if not k.startswith("_")}

    return EvalResult(
        score=getattr(raw, "score", None),
        passed=_extract_passed(raw),
        reason=getattr(raw, "reason", ""),
        evaluator=evaluator,
        details=details,
    )


# 历史上用过的「通过」字段名。列在这里而不是各处判断，
# 是为了让「别名」这件事集中可见。
_PASSED_ALIASES = (
    "passed", "is_correct", "is_complete", "is_relevant",
    "is_consistent", "ok",
)


def _extract_passed(raw) -> Optional[bool]:
    """从多种形态里提取「是否通过」"""
    if isinstance(raw, dict):
        for key in _PASSED_ALIASES:
            if key in raw:
                return bool(raw[key])
        return None
    for key in _PASSED_ALIASES:
        if hasattr(raw, key):
            return bool(getattr(raw, key))
    return None
