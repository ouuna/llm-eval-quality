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
from eval import thresholds


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
    #
    # 阈值从eval.thresholds 取，而不是各评测器的默认值。
    # 此前config.yaml 写了 evaluation.hallucination_threshold: 0.6
    # 却从不生效——真正用的是函数签名里的默认 0.35。
    # 改配置没反应属于最坏的一类问题：看起来可调，实际不可调。
    faith = detect_hallucination(
        answer=answer,
        context=context,
        required_facts=gt.required_facts,
        forbidden_facts=gt.forbidden_facts,
        should_refuse=case.should_refuse,
        supported_threshold=thresholds.get_threshold(
            "hallucination_supported"),
        partial_threshold=thresholds.get_threshold(
            "hallucination_partial"),
        forbidden_match=thresholds.get_threshold("forbidden_match"),
        required_hit=thresholds.get_threshold("required_fact_hit"),
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
            threshold=thresholds.get_threshold("correctness_pass"),
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
        comp = evaluate_completeness(
            answer, gt.required_facts, context,
            threshold=thresholds.get_threshold("completeness_pass"),
        )
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

        # 9a. 有据性（答案↔上下文）
        sem = semantic_evaluator.groundedness(answer, context)
        if sem.get("available"):
            result.scores["semantic_groundedness"] = sem["score"]
        else:
            result.scores["semantic_groundedness"] = None
            result.evidence.append(
                f"语义评测不可用：{sem.get('reason')}")

        # 9b. 语义事实覆盖率（答案↔required_facts）
        #
        # 这一项此前是**死代码**：方法写好了、阈值 0.65 调过了，
        # 但 runner 只调 groundedness，导致它从未参与任何判定。
        #
        # 它的价值：词级匹配认不出同义改写
        # （「切分」vs「划分」在字符层面几乎无交集），
        # 而语义相似度能认出来。
        # 两者互补：词级给基线，语义补同义改写。
        if gt.required_facts:
            fc = semantic_evaluator.fact_coverage(answer,
                                                  gt.required_facts)
            if fc.get("available"):
                result.scores["semantic_fact_coverage"] = fc["score"]
                # 语义层发现词级漏掉的事实，是很有价值的诊断信息
                sem_covered = fc.get("covered_facts") or []
                if sem_covered:
                    result.evidence.append(
                        f"语义层额外识别到要点：{sem_covered[:3]}")
            else:
                result.scores["semantic_fact_coverage"] = None
                result.evidence.append(
                    f"语义事实覆盖不可用：{fc.get('reason')}")

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
            if jr.final_score is not None:
                result.scores["judge_final"] = jr.final_score
            result.evidence.append(f"Judge 判定：{jr.reason}")
            if jr.hallucination:
                result.add_violation(
                    ViolationType.HALLUCINATION.value,
                    f"LLM Judge 判定为幻觉：{jr.reason}",
                    severity=Severity.HIGH.value,
                    evidence=jr.evidence[:3],
                )
            elif jr.passed is False:
                # 判定为有幻觉之外的质量问题（如完整性差）。
                #
                # 此前只处理 `jr.hallucination`，
                # 结果是「有幻觉=False 但五维均低于阈值」的情况
                # 完全不产生违规——Judge 打了分却没人看。
                weak = jr.weak_dimensions
                result.add_violation(
                    ViolationType.INCOMPLETE_ANSWER.value,
                    (f"LLM Judge 判定不达标，薄弱维度：{'、'.join(weak)}"
                     if weak else
                     f"LLM Judge 综合分偏低（{jr.final_score}）"),
                    severity=Severity.MEDIUM.value,
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
# 统计辅助
# ============================================================
def _percentile(values: List[float], p: float) -> Optional[float]:
    """
    取第 p 百分位（p 取0~100）。

    为什么不用 `sorted(l)[int(len(l) * p / 100)]`
    ---------------------------------------
    那样写当 `int(len(l) * p / 100)` 正好等于 `len(l)` 时会越界
    （比如 len=20、p=95 时 int(19.0)=19 侥幸安全，
    但 len=100、p=99 时 int(99)=99 也安全，
    len=20、p=100 时 int(20)=20 就越界了）。

    更麻烦的是项目里曾有两处不同的 P95 实现：
    schemas/result.py 用 `round((p/100)*(n-1))`，
    runner.py 用 `int(len*p/100)`，两处结果可能不同。
    同一个指标在报告和门禁里算出两个值，是很难查的问题。

    现在统一走 eval.metrics.percentile。
    """
    from eval.metrics import percentile
    return percentile(values, p)


def _error_rate(results: List[CaseResult]) -> Optional[float]:
    """
    错误率 = error 条数 / 总条数。

    分母是全部结果（包含 error 本身），这点很关键：
    如果像 pass_rate 那样把 error 排除出分母，
    「全部调用失败」会得到 0 而不是 1，
    看起来反而比「一半失败」更健康——完全说反了。
    """
    if not results:
        return None
    errors = sum(1 for r in results if r.status == Status.ERROR.value)
    return round(errors / len(results), 4)


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

    # ---- 数据集前置校验 ----
    #
    # validate_dataset 早就写好了，但一直没有调用点——
    # 于是脏数据（比如 expected_behavior 与 ground_truth 矛盾）
    # 会一路跑完整个评测，最后报出一个看不懂的指标。
    #
    # 评测前拦住它，错误信息能直接指出是哪条用例的哪个字段不对。
    from eval.schemas.dataset import validate_dataset
    ds_issues = validate_dataset(dataset)
    if ds_issues:
        print("[WARN] 数据集存在问题，部分结果可能不可信：")
        for issue in ds_issues[:10]:
            print(f"  - {issue}")
        if len(ds_issues) > 10:
            print(f"  ...另有 {len(ds_issues) - 10} 项")
        print()

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
            name=name,
            # value 字段在无数据时填0.0，靠 available=False 区分。
            # 这不是理想设计（0.0 本身是合法数值），
            # 但 MetricSummary.value 的类型是 float 且必填，
            # 改签名会波及序列化格式与报告渲染。
            # 真正的修法是让下游一律先看 available——
            # 门禁已经这样做了，见 gate._check()。
            value=value if value is not None else 0.0,
            available=value is not None,
            # definition 此前写的是 `report.evaluators and name or name`，
            # 那是个恒等于 name 的无意义表达式——
            # 所以指标的算法说明一直是空的，报告里看不出分数怎么算的。
            # 现在真正从 thresholds 取定义与局限说明。
            definition=thresholds.definition_of(name),
            caveat=thresholds.caveat_of(name),
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

    # ---- 报告自检 ----
    #
    # validate_result 同样早就写好了但没人调。
    # 它检查的是「报告自己是否自洽」，比如：
    #   · 标成 passed 却带着 FAILING 级违规
    #   · 标成 failed 却没有任何违规记录
    #   · 标成 error 却没有 error 信息
    # 这类矛盾会让读报告的人被误导——
    # 比如「全部通过」的报告里有一条 critical 违规，
    # 排查时会先怀疑指标算错，而不是怀疑状态标记错。
    from eval.schemas.result import validate_result
    self_check = []
    for c in results:
        self_check.extend(validate_result(c))
    if self_check:
        print("[WARN] 结果自检发现不一致（可能是评测逻辑的 bug）：")
        for issue in self_check[:10]:
            print(f"  - {issue}")
        if len(self_check) > 10:
            print(f"  ...另有 {len(self_check) - 10} 项")
        print()

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
        "p95_latency_ms": _percentile(
            [r.latency_ms for r in ok if r.latency_ms > 0], 95),
        # 错误率：调用失败（含超时、5xx、解析错误）的占比
        #
        # 此前门禁里根本没有这条规则，于是「API 全挂」和
        # 「质量达标」在报告上看起来一样。
        # 它必须独立于 pass_rate：pass_rate 的分母已排除 error，
        # 所以一个全是error 的run 会得到 None 而不是 0——
        # 那种情况下恰恰需要 error_rate 来说明「到底失败了多少」。
        "error_rate": _error_rate(ok),
    }

    # 成本（仅当 API 提供）
    #
    # 此前这里有一行：
    #     token_stats = EvalReport(dataset_name="", ...).token_stats()
    # 专门实例化一个空报告只为调一次 token_stats()，
    # 而它的结果从未被使用——紧接着下面又自己算了一遍。
    # 属于纯冗余，且白白构造对象。已删除。
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


# ============================================================
# 统一评测结果出口
# ============================================================
def collect_evaluator_results(case, answer, context, faith, corr, comp,
                              rel, refusal, semantic_result=None,
                              judge_result=None) -> List:
    """
    把一次评测中各评测器的结果收敛成统一的 EvalResult 列表。

    需求五要求「每种evaluator 返回统一结构」。
    这层是它的落地点：无论内部用什么算法实现，
    对外一律是 EvalResult，调用方不用记每个函数各自的字段名。

    同时提供一个「总体判定」：任一评测器不通过则整体不通过。
    这与 run_evaluation 的聚合口径一致（任一FAILING 违规即失败）。
    """
    from eval.evaluators.base import EvalResult, to_result

    out: List[EvalResult] = []

    # 幻觉检测：原始 dict → 统一结构
    if faith is not None:
        out.append(EvalResult(
            score=faith.get("coverage"),
            passed=not faith.get("has_hallucination"),
            reason=faith.get("reason", ""),
            evaluator="faithfulness",
            details=dict(faith),
        ))

    # 正确性：已是统一结构（CorrectnessResult 继承 EvalResult）
    if corr is not None:
        out.append(to_result(corr, "correctness"))

    # 完整性 / 相关性 / 拒答：旧 dict → 统一结构
    if comp is not None:
        out.append(to_result(comp, "completeness"))
    if rel is not None:
        out.append(to_result(rel, "relevance"))
    if refusal is not None:
        out.append(to_result(refusal, "refusal"))

    # 可选组件
    if semantic_result is not None:
        out.append(to_result(semantic_result, "semantic"))
    if judge_result is not None:
        out.append(to_result(judge_result, "judge"))

    return out


def overall_verdict(results: List) -> Dict[str, Any]:
    """
    汇总统一结构的结果，给出总体判定。

    判定口径：任一评测器明确不通过（passed=False）则整体不通过。
    `passed=None`（无法判定）不计入失败——
    与 pass_rate 排除 error 的思路一致：
    「没测到」不等于「没通过」。
    """
    from eval.evaluators.base import EvalResult

    failed = [r for r in results if r.passed is False]
    undecided = [r for r in results if r.passed is None]

    return {
        "passed": not failed,
        "failed_evaluators": [r.evaluator for r in failed],
        "undecided_evaluators": [r.evaluator for r in undecided],
        "count": len(results),
        "reasons": {r.evaluator: r.reason for r in failed},
    }
