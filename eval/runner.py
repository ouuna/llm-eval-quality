"""
评测执行编排
---------------------------------
把 Provider、Evaluator、Schema、Quality Gate 串成完整链路：

    数据集 → SUT → 多维评测 → 指标汇总 → 质量门禁

这是评测系统的入口层，负责协调而非实现具体指标。
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

from eval.schemas.result import (
    CaseResult, EvalReport, MetricSummary, TokenUsage,
    Severity, Status, Violation, ViolationType,
)
from eval.schemas.dataset import Dataset, EvalCase
from eval.evaluators.faithfulness import detect_hallucination
from eval.evaluators.correctness import (
    evaluate_correctness, evaluate_completeness, evaluate_relevance,
    evaluate_refusal, detect_context_conflict,
)
from eval.quality_gate.gate import QualityGate, load_gate_config
from eval.reporting.html_report import save_html
from eval.reporting.csv_report import save_csv
from eval import env_loader


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUALITY_GATE_CONFIG = os.path.join(PROJECT_ROOT, "configs", "quality_gate.yaml")


@dataclass
class RunResult:
    """一次评测的完整产出"""
    report: EvalReport
    html_path: Optional[str] = None
    json_path: Optional[str] = None
    csv_path: Optional[str] = None

    def save_all(self, output_dir: str) -> Dict[str, str]:
        paths = {}
        os.makedirs(output_dir, exist_ok=True)
        self.html_path = save_html(self.report, output_dir)
        self.json_path = self.report.save_json(
            os.path.join(output_dir, "eval_results.json"))
        self.csv_path = save_csv(self.report, output_dir)
        return {"html": self.html_path, "json": self.json_path,
                "csv": self.csv_path}


# ============================================================
# 单用例评测
# ============================================================
def evaluate_case(case: EvalCase, provider,
                  use_judge: bool = False,
                  use_semantic: bool = False,
                  semantic_evaluator=None,
                  judge_client=None) -> CaseResult:
    """
    评测单条用例

    完整链路：
        1. 调用被测系统
        2. 声明级幻觉检测
        3. 正确性/完整性/相关性
        4. 拒答判定
        5. 上下文冲突（仅冲突类用例）
        6. 可选：语义评测、LLM Judge
    """
    # ---- Step 1: 调用被测系统 ----
    try:
        resp = provider.ask(case.question)
    except Exception as e:
        return CaseResult.error_result(
            case.id, f"Provider 调用失败：{type(e).__name__}: {e}",
            ViolationType.SUT_ERROR.value,
            category=case.category, question=case.question,
            expected_behavior=case.expected_behavior,
            difficulty=case.difficulty, tags=list(case.tags),
            reference_answer=case.ground_truth.reference_answer,
        )

    # ---- Step 2: Provider 报错必须显式标记 ----
    # 需求第二十六条：不要把 API failure 当成 pass
    if resp.error:
        return CaseResult.error_result(
            case.id, f"被测系统返回错误：{resp.error}",
            ViolationType.SUT_ERROR.value,
            category=case.category, question=case.question,
            expected_behavior=case.expected_behavior,
            difficulty=case.difficulty, tags=list(case.tags),
            latency_ms=resp.latency_ms,
            token_usage=resp.token_usage,
            reference_answer=case.ground_truth.reference_answer,
            retrieval_context=list(resp.contexts),
        )

    answer = resp.answer or ""
    context = resp.context_str
    gt = case.ground_truth

    result = CaseResult(
        case_id=case.id,
        category=case.category,
        question=case.question,
        expected_behavior=case.expected_behavior,
        difficulty=case.difficulty,
        tags=list(case.tags),
        actual_output=answer,
        reference_answer=gt.reference_answer,
        retrieval_context=list(resp.contexts),
        latency_ms=resp.latency_ms,
        token_usage=resp.token_usage,
        evaluators_used=["faithfulness", "correctness", "relevance"],
    )

    # ---- Step 3: 声明级幻觉检测 ----
    faith = detect_hallucination(
        answer=answer,
        context=context,
        required_facts=gt.required_facts,
        forbidden_facts=gt.forbidden_facts,
        should_refuse=case.should_refuse,
    )
    result.scores["faithfulness"] = round(faith["coverage"], 4)
    result.evidence.extend([c.text for c in faith["claims"][:5]])

    if faith["has_hallucination"]:
        result.add_violation(
            ViolationType.HALLUCINATION.value,
            faith["reason"],
            severity=Severity.CRITICAL.value,
            evidence=[c.text for c in faith["unsupported_claims"][:3]],
        )

    # ---- Step 4: 正确性 ----
    if case.expected_behavior == "answer":
        corr = evaluate_correctness(
            answer, gt.required_facts, gt.forbidden_facts,
            gt.acceptable_answers, gt.reference_answer,
        )
        result.scores["correctness"] = corr.score
        result.evidence.extend([
            f"正确性判定：{corr.reason}（method={corr.method}）"])

        if not corr.is_correct:
            vtype = (ViolationType.WRONG_ANSWER.value
                     if corr.wrong_facts
                     else ViolationType.INCOMPLETE_ANSWER.value)
            result.add_violation(
                vtype, corr.reason, severity=Severity.HIGH.value,
                missing_facts=corr.missing_facts,
                wrong_facts=corr.wrong_facts,
            )

        # ---- Step 5: 完整性 ----
        comp = evaluate_completeness(answer, gt.required_facts, context)
        result.scores["completeness"] = comp["score"]
        if not comp["is_complete"] and comp["missing_facts"]:
            result.add_violation(
                ViolationType.INCOMPLETE_ANSWER.value,
                comp["reason"], severity=Severity.MEDIUM.value,
                evidence=comp["missing_facts"],
            )

    # ---- Step 6: 相关性 ----
    rel = evaluate_relevance(
        answer, gt.reference_answer, gt.required_facts, case.question, context)
    result.scores["relevance"] = rel["score"]
    if not rel["is_relevant"]:
        result.add_violation(
            ViolationType.IRRELEVANT.value, rel["reason"],
            severity=Severity.HIGH.value,
        )

    # ---- Step 7: 拒答判定 ----
    ref = evaluate_refusal(
        answer, case.should_refuse, gt.forbidden_facts)
    result.scores["refusal"] = 1.0 if ref["is_correct"] else 0.0
    if not ref["is_correct"]:
        vtype = (ViolationType.REFUSAL_ERROR.value
                 if ref["mode"] == "false_answer"
                 else ViolationType.IRRELEVANT.value)
        result.add_violation(vtype, ref["reason"],
                             severity=Severity.CRITICAL.value)
        if ref.get("forbidden_hits"):
            result.evidence.append(
                f"拒答场景命中的禁止内容：{ref['forbidden_hits']}")

    # ---- Step 8: 上下文冲突（仅冲突类用例）----
    if case.category == "context_conflict" and context:
        conflict = detect_context_conflict(context, answer)
        result.scores["conflict_detected"] = 1.0 if conflict["has_conflict"] else 0.0
        if conflict["has_conflict"]:
            result.evidence.append(
                f"检出上下文冲突：{conflict.get('conflicts')}")
            if not conflict.get("acknowledged"):
                result.add_violation(
                    ViolationType.CONTEXT_CONFLICT.value,
                    f"存在上下文冲突但模型未识别：{conflict['reason']}",
                    severity=Severity.HIGH.value,
                )

    # ---- Step 9: 语义评测（可选）----
    if use_semantic and semantic_evaluator is not None:
        result.evaluators_used.append("semantic")
        sem = semantic_evaluator.groundedness(answer, context)
        if sem.get("available"):
            result.scores["semantic_groundedness"] = sem["score"]
        else:
            result.scores["semantic_groundedness"] = None
            result.evidence.append(
                f"语义评测不可用：{sem.get('reason')}")

    # ---- Step 10: LLM Judge（可选）----
    if use_judge and judge_client is not None:
        result.evaluators_used.append("judge")
        jr = judge_client.judge(
            case.question, answer, context, gt.reference_answer,
            focus=f"重点检查：{case.purpose}",
        )
        if jr.available:
            result.scores.update({f"judge_{k}": v
                                 for k, v in jr.scores.items()})
            result.evidence.append(f"Judge 判定：{jr.reason}")
            if jr.hallucination:
                result.add_violation(
                    ViolationType.HALLUCINATION.value,
                    f"LLM Judge 判定为幻觉：{jr.reason}",
                    severity=Severity.HIGH.value,
                    evidence=jr.evidence[:3],
                )
        else:
            # Judge 失败必须进入 evaluator_error，不能当通过
            result.add_violation(
                ViolationType.EVALUATOR_ERROR.value,
                f"Judge 判定失败：{jr.error}",
                severity=Severity.HIGH.value,
                error_kind=jr.error_kind,
            )

    return result


# ============================================================
# 批量评测
# ============================================================
def run_evaluation(dataset: Dataset,
                   use_judge: bool = False,
                   use_semantic: bool = False,
                   use_mock: bool = False,
                   stability_repeat: int = 3,
                   provider_name: str = None,
                   gate_config: str = None) -> RunResult:
    """
    执行完整评测

    参数
    ----
    dataset           数据集
    use_judge         启用 LLM Judge
    use_semantic      启用 Embedding 评测
    use_mock          使用 Mock Provider（无需 API）
    stability_repeat  稳定性重复次数
    """
    started = time.time()

    # ---- Provider ----
    if use_mock:
        from eval.providers.mock import MockProvider
        provider = MockProvider(sequence=["correct"] * len(dataset))
    elif provider_name:
        from eval.providers.sut import get_provider
        provider = get_provider(provider_name)
    else:
        from eval.providers.sut import get_provider
        provider = get_provider("rag")

    # ---- 可选组件 ----
    semantic_evaluator = None
    if use_semantic and not use_mock:
        from eval.evaluators.semantic import SemanticEvaluator
        semantic_evaluator = SemanticEvaluator()

    judge_client = None
    if use_judge and not use_mock:
        from eval.evaluators.judge import JudgeClient
        judge_client = JudgeClient()

    # ---- 逐用例评测 ----
    results: List[CaseResult] = []
    for i, case in enumerate(dataset.cases, 1):
        r = evaluate_case(
            case, provider,
            use_judge=use_judge,
            use_semantic=use_semantic,
            semantic_evaluator=semantic_evaluator,
            judge_client=judge_client,
        )
        results.append(r)

    # ---- 稳定性 ----
    stability = _measure_stability(dataset, provider, stability_repeat,
                                   use_mock)

    # ---- 汇总指标 ----
    metrics, failed_by_metric = _aggregate(results, stability)

    # ---- 组装报告 ----
    report = EvalReport(
        dataset_name=dataset.name,
        dataset_tier=dataset.tier,
        model=("mock" if use_mock else env_loader.get_model_name("unknown")),
        provider=(provider.name if hasattr(provider, "name") else "unknown"),
        started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        duration_sec=round(time.time() - started, 1),
        cases=results,
        evaluators=["faithfulness", "correctness",
                    "completeness", "relevance", "refusal"]
        + (["semantic"] if use_semantic else [])
        + (["judge"] if use_judge else []),
        run_errors=[],
    )

    for name, value in metrics.items():
        report.metrics[name] = MetricSummary(
            name=name, value=value if value is not None else 0.0,
            available=value is not None,
            definition=report.evaluators and name or name,
        )

    # ---- 质量门禁 ----
    gate = QualityGate(load_gate_config(gate_config or QUALITY_GATE_CONFIG))
    gate_report = gate.evaluate(
        {k: v for k, v in metrics.items() if v is not None},
        failed_by_metric,
    )
    report.gate_passed = gate_report.passed
    report.gate_failures = [c.to_dict() for c in gate_report.failed_checks]
    report.gate_message = gate_report.text()

    return RunResult(report=report)


# ============================================================
# 内部工具
# ============================================================
def _measure_stability(dataset, provider, repeat, use_mock) -> Dict[str, Any]:
    """测量输出稳定性"""
    if repeat < 2:
        return {"available": False, "reason": "重复次数不足"}

    from eval.evaluators.faithfulness import content_chars

    # 选前3 条可答的用例做稳定性测试
    targets = [c for c in dataset.cases
               if c.expected_behavior == "answer"][:3]
    if not targets:
        return {"available": False, "reason": "无可测用例"}

    scores = []
    for case in targets:
        answers = []
        for _ in range(repeat):
            try:
                resp = provider.ask(case.question)
                if not resp.error:
                    answers.append(resp.answer)
            except Exception:
                pass

        if len(answers) < 2:
            continue

        pairs = []
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                a, b = content_chars(answers[i]), content_chars(answers[j])
                if a and b:
                    pairs.append(len(a & b) / len(a | b))
        if pairs:
            scores.append(sum(pairs) / len(pairs))

    if not scores:
        return {"available": False, "reason": "有效样本不足"}

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return {
        "available": True,
        "consistency": round(mean, 4),
        "std": round(variance ** 0.5, 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "sample_count": len(scores),
    }


def _aggregate(results: List[CaseResult],
               stability: Dict[str, Any]) -> tuple:
    """汇总指标，返回 (metrics, failed_by_metric)"""
    ok = [r for r in results]
    valid = [r for r in ok if r.status != Status.ERROR.value]

    def avg(key: str, default=None):
        vals = [r.scores.get(key) for r in valid
                if r.scores.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else default

    # 幻觉率 = 有 HALLUCINATION 违规的比例
    halluc_count = sum(
        1 for r in valid
        if ViolationType.HALLUCINATION.value in r.violation_types())
    halluc_rate = round(halluc_count / len(valid), 4) if valid else None

    # 拒答准确率
    refusal_vals = [r.scores.get("refusal") for r in valid
                    if r.scores.get("refusal") is not None]
    refusal_acc = (round(sum(refusal_vals) / len(refusal_vals), 4)
                   if refusal_vals else None)

    metrics = {
        "overall_pass_rate": round(
            sum(1 for r in valid if r.is_passed) / len(valid), 4) if valid else None,
        "hallucination_rate": halluc_rate,
        "refusal_accuracy": refusal_acc,
        "faithfulness": avg("faithfulness"),
        "answer_correctness": avg("correctness"),
        "completeness": avg("completeness"),
        "relevance": avg("relevance"),
        "retrieval_hit_rate": round(
            sum(1 for r in ok if r.retrieval_context) / len(ok), 4) if ok else None,
        "consistency": stability.get("consistency") if stability.get("available") else None,
        "score_std": stability.get("std") if stability.get("available") else None,
        "p95_latency_ms": (lambda l: round(sorted(l)[int(len(l) * 0.95)]
                                       if l else None))(
            [r.latency_ms for r in ok if r.latency_ms > 0]),
    }

    # 成本（仅当 API 提供）
    token_stats = EvalReport(dataset_name="", dataset_tier="",
                             model="").token_stats()
    tok = [r.token_usage for r in ok if r.token_usage.available]
    metrics["avg_cost"] = (round(
        sum(t.cost for t in tok if t.cost is not None) / len(tok), 6)
        if tok and any(t.cost is not None for t in tok) else None)

    # 失败样本归类
    failed_by_metric: Dict[str, List[str]] = {}
    for r in ok:
        if r.is_failed:
            for vt in r.violation_types():
                failed_by_metric.setdefault(vt, []).append(r.case_id)
    if halluc_rate is not None and halluc_rate > 0:
        failed_by_metric["hallucination_rate"] = \
            failed_by_metric.get(ViolationType.HALLUCINATION.value, [])
    if refusal_acc is not None and refusal_acc < 1.0:
        failed_by_metric["refusal_accuracy"] = [
            r.case_id for r in ok
            if r.scores.get("refusal") is not None
            and r.scores["refusal"] < 1.0]

    return metrics, failed_by_metric
