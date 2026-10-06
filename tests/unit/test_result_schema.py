"""
Result Schema 单元测试
---------------------------------
不依赖真实 LLM API，全部使用构造数据。

重点覆盖
--------
1. error 状态不计入通过（修复 P0-1）
2. Token 不可用时记 None 而非 0
3. 违规自动将状态置为 failed
4. 延迟统计的 P50/P95
5. Result 自洽性校验能捕获不一致
"""

import pytest

from eval.schemas.result import (
    CaseResult, EvalReport, MetricSummary, TokenUsage, Violation,
    Severity, Status, ViolationType, validate_result,
)


# ============================================================
# 1. error 状态处理（P0-1 修复验证）
# ============================================================
class TestErrorHandling:
    """验证 API 失败不再被当作通过"""

    def test_error_result_not_passed(self):
        r = CaseResult.error_result("case_01", "API 调用超时")
        assert r.is_error
        assert not r.is_passed
        assert not r.is_failed
        assert r.status == Status.ERROR.value

    def test_error_has_critical_violation(self):
        r = CaseResult.error_result("case_01", "API 调用超时")
        assert len(r.critical_violations) == 1
        assert r.critical_violations[0].severity == Severity.CRITICAL.value

    def test_error_can_use_custom_violation_type(self):
        """评测器自身异常与被测系统异常应区分"""
        r = CaseResult.error_result("case_01", "判定逻辑异常",
                             violation_type=ViolationType.EVALUATOR_ERROR.value)
        assert r.violation_types() == [ViolationType.EVALUATOR_ERROR.value]

    def test_pass_rate_excludes_error(self):
        """
        error 不计入分母

        理由：error 是"评测未完成"，不是"质量表现"。
        混入分母会让通过率被评测故障稀释，掩盖真实质量。
        """
        report = EvalReport(dataset_name="t", dataset_tier="smoke", model="m")
        report.cases = [
            CaseResult.passed("c1"),
            CaseResult.passed("c2"),
            CaseResult.error_result("c3", "超时"),
        ]
        assert report.passed == 2
        assert report.errors == 1
        # 2/(2+0) = 1.0，不因 error 而拉低
        assert report.pass_rate == 1.0

    def test_error_regression_case_not_passed(self):
        """
        对应 REGRESSION: REG-004
        API 异常场景必须判为 error 而非 pass
        """
        r = CaseResult.error_result("REG-004", "调用大模型失败，已重试 3 次")
        assert r.is_error
        assert r.error is not None


# ============================================================
# 2. Token / Cost 可用性
# ============================================================
class TestTokenUsage:
    """验证不可用数据记None 而非 0"""

    def test_unavailable_defaults_to_none(self):
        u = TokenUsage.unavailable()
        assert not u.available
        assert u.total_tokens is None
        assert u.cost is None

    def test_from_api_returns_unavailable_when_missing(self):
        u = TokenUsage.from_api(None)
        assert not u.available
        assert u.total_tokens is None

    def test_from_api_with_valid_usage(self):
        u = TokenUsage.from_api(
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        )
        assert u.available
        assert u.total_tokens == 150
        assert u.cost is None      # 未提供单价则不估算

    def test_from_api_computes_cost(self):
        u = TokenUsage.from_api(
            {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
            cost_per_1k=0.001,
        )
        assert u.cost == pytest.approx(0.0015)

    def test_cost_computed_from_total_when_api_omits_it(self):
        u = TokenUsage.from_api(
            {"prompt_tokens": 100, "completion_tokens": 50},
            cost_per_1k=0.001,
        )
        assert u.total_tokens == 150
        assert u.cost == pytest.approx(0.00015)

    def test_report_token_stats_unavailable(self):
        """全部不可用时应报告 unavailable，而不是 0"""
        report = EvalReport(dataset_name="t", dataset_tier="smoke", model="m")
        report.cases = [CaseResult.passed("c1"), CaseResult.passed("c2")]
        stats = report.token_stats()
        assert stats["available"] is False
        assert stats["total_tokens"] is None
        assert stats["avg_cost"] is None


# ============================================================
# 3. 违规处理
# ============================================================
class TestViolation:
    def test_add_violation_flips_status(self):
        r = CaseResult.passed("c1")
        assert r.is_passed
        r.add_violation(ViolationType.HALLUCINATION.value, "包含无依据内容")
        assert r.is_failed
        assert not r.is_passed

    def test_add_violation_records_evidence(self):
        r = CaseResult.passed("c1")
        r.add_violation(
            ViolationType.HALLUCINATION.value, "实体调换",
            evidence=["上下文：北京是首都", "回答：中国的首都是上海"],
        )
        v = r.violations[0]
        assert len(v.evidence) == 2
        assert "上海" in v.evidence[1]

    def test_invalid_severity_rejected(self):
        with pytest.raises(ValueError):
            Violation(type=ViolationType.HALLUCINATION.value,
                      severity="不存在的等级", message="x")

    def test_invalid_violation_type_rejected(self):
        with pytest.raises(ValueError):
            Violation(type="NOT_A_REAL_TYPE", severity=Severity.HIGH.value, message="x")

    def test_failed_with_explicit_violations(self):
        v = Violation(type=ViolationType.WRONG_ANSWER.value,
                      severity=Severity.HIGH.value, message="答案错误")
        r = CaseResult.failed("c1", [v])
        assert r.is_failed


# ============================================================
# 4. 统计
# ============================================================
class TestStatistics:
    def test_latency_stats_unavailable_when_all_zero(self):
        report = EvalReport(dataset_name="t", dataset_tier="smoke", model="m")
        report.cases = [CaseResult.passed("c1"), CaseResult.passed("c2")]
        lat = report.latency_stats()
        assert lat["available"] is False
        assert lat["mean"] is None

    def test_latency_stats_percentiles(self):
        report = EvalReport(dataset_name="t", dataset_tier="smoke", model="m")
        report.cases = [CaseResult.passed(f"c{i}", latency_ms=v)
                        for i, v in enumerate([100, 200, 300, 400, 500])]
        lat = report.latency_stats()
        assert lat["available"]
        assert lat["mean"] == 300.0
        assert lat["p50"] == 300.0
        assert lat["max"] == 500.0

    def test_latency_single_case(self):
        report = EvalReport(dataset_name="t", dataset_tier="smoke", model="m")
        report.cases = [CaseResult.passed("c1", latency_ms=250)]
        lat = report.latency_stats()
        assert lat["p50"] == 250.0
        assert lat["p95"] == 250.0

    def test_failure_pareto_sorted_and_cumulative(self):
        report = EvalReport(dataset_name="t", dataset_tier="smoke", model="m")
        r1 = CaseResult.passed("c1")
        r1.add_violation(ViolationType.HALLUCINATION.value, "a")
        r1.add_violation(ViolationType.RETRIEVAL_ERROR.value, "b")
        r2 = CaseResult.passed("c2")
        r2.add_violation(ViolationType.HALLUCINATION.value, "a")
        report.cases = [r1, r2]

        pareto = report.failure_pareto()
        assert pareto[0]["type"] == ViolationType.HALLUCINATION.value
        assert pareto[0]["count"] == 2
        assert pareto[0]["rate"] == pytest.approx(2 / 3, abs=0.01)
        assert pareto[-1]["cumulative"] == pytest.approx(1.0, abs=0.01)

    def test_category_stats(self):
        report = EvalReport(dataset_name="t", dataset_tier="smoke", model="m")
        report.cases = [
            CaseResult.passed("c1", category="in_domain"),
            CaseResult.failed("c2", [Violation(ViolationType.WRONG_ANSWER.value,
                                               Severity.HIGH.value, "x")],
                              category="in_domain"),
            CaseResult.error_result("c3", "超时", category="out_domain"),
        ]
        stats = report.category_stats()
        by_cat = {s["category"]: s for s in stats}
        assert by_cat["in_domain"]["total"] == 2
        assert by_cat["in_domain"]["pass_rate"] == 0.5
        assert by_cat["out_domain"]["error"] == 1

    def test_empty_report_is_valid(self):
        report = EvalReport(dataset_name="t", dataset_tier="smoke", model="m")
        assert report.total == 0
        assert report.pass_rate == 0.0
        assert report.failure_pareto() == []
        assert report.latency_stats()["available"] is False


# ============================================================
# 5. 自洽性校验
# ============================================================
class TestValidation:
    """
    验证 validate_result 能捕获"状态与违规不一致"

    这类不一致正是评测器自身缺陷的信号。
    旧实现里"空输出→无幻觉→通过"就是这种不一致的极端案例。
    """

    def test_passed_with_failing_violation_is_invalid(self):
        r = CaseResult.passed("c1")
        r.violations.append(Violation(ViolationType.HALLUCINATION.value,
                                      Severity.HIGH.value, "x"))
        # 手动把状态改回 passed，模拟不一致
        r.status = Status.PASSED.value
        issues = validate_result(r)
        assert any("FAILING" in i for i in issues)

    def test_failed_without_violation_is_invalid(self):
        r = CaseResult.failed("c1", [])
        issues = validate_result(r)
        assert any("无 FAILING" in i for i in issues)

    def test_error_without_error_field_is_invalid(self):
        r = CaseResult.error_result("c1", "超时")
        r.error = None
        issues = validate_result(r)
        assert any("error 字段为空" in i for i in issues)

    def test_passed_with_error_field_is_invalid(self):
        r = CaseResult.passed("c1")
        r.error = "残留错误"
        issues = validate_result(r)
        assert any("error 字段" in i for i in issues)

    def test_consistent_result_passes(self):
        r = CaseResult.error_result("c1", "超时")
        assert validate_result(r) == []

        r2 = CaseResult.passed("c2")
        assert validate_result(r2) == []


# ============================================================
# 6. 序列化
# ============================================================
class TestSerialization:
    def test_case_result_to_dict(self):
        r = CaseResult.passed("c1", category="in_domain", latency_ms=100)
        d = r.to_dict()
        assert d["case_id"] == "c1"
        assert d["status"] == "passed"
        assert d["token_usage"]["available"] is False
        assert d["violations"] == []

    def test_report_to_dict_and_json_serializable(self):
        import json
        report = EvalReport(dataset_name="smoke", dataset_tier="smoke",
                            model="glm-4-flash")
        report.cases = [CaseResult.passed("c1", category="in_domain")]
        report.metrics["faithfulness"] = MetricSummary(
            name="faithfulness", value=0.95,
            definition="答案是否有上下文支撑", caveat="字符级方法有盲区",
        )
        report.gate_passed = False
        report.gate_failures = [{"metric": "faithfulness", "message": "0.95 < 0.98"}]

        d = report.to_dict()
        # 必须可 JSON 序列化
        s = json.dumps(d, ensure_ascii=False)
        back = json.loads(s)
        assert back["summary"]["total"] == 1
        assert back["gate"]["passed"] is False
        assert back["metrics"]["faithfulness"]["value"] == 0.95

    def test_summary_text_contains_key_info(self):
        report = EvalReport(dataset_name="smoke", dataset_tier="smoke",
                            model="glm-4-flash")
        report.cases = [
            CaseResult.passed("c1"),
            CaseResult.error_result("c2", "超时"),
        ]
        report.gate_passed = False
        report.gate_failures = [{"message": "error 1 条"}]

        text = report.summary_text()
        assert "Evaluation Summary" in text
        assert "Errors:  1" in text
        assert "Quality Gate: FAILED" in text
        assert "unavailable" in text      # token 不可用应显式说明
