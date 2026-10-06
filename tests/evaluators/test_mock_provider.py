"""
Mock Provider 与离线评测测试
---------------------------------
**这些测试完全不需要 API Key、不发任何网络请求。**

这意味着：
- 本地随时可跑，秒级完成
- CI 无需配置任何密钥
- 可以构造真实环境难以复现的异常路径

重点验证需求第二十一条的 8 种故障注入，
以及"评测器在各种被测行为下的正确判定"。
"""

import pytest

from eval.providers.mock import (
    MockProvider, SCENARIOS, Scenario,
    scenario_provider, make_error_provider, make_sequence_provider,
    make_evaluator_scenario_provider,
)
from eval.evaluators.faithfulness import detect_hallucination
from eval.evaluators.correctness import (
    evaluate_correctness, evaluate_completeness, evaluate_relevance,
    evaluate_refusal,
)


# ============================================================
# 1. Mock Provider 自身行为
# ============================================================
class TestMockProviderBasics:
    def test_always_available(self):
        """Mock 永远可用——这正是它的价值"""
        assert MockProvider().is_available()

    def test_fixed_scenario(self):
        p = scenario_provider("entity_swap")
        r = p.ask("自动化测试框架通常分为哪几层？")
        assert r.answer == SCENARIOS["entity_swap"].answer
        assert r.contexts
        assert r.error is None

    def test_error_scenario(self):
        p = scenario_provider("timeout")
        r = p.ask("任意问题")
        assert r.error, "异常场景必须返回 error"
        assert not r.ok

    def test_sequence_mode(self):
        p = make_sequence_provider("correct", "entity_swap")
        a = p.ask("问题")
        b = p.ask("问题")
        assert a.answer != b.answer, "顺序模式应轮换返回"

    def test_callable_mode(self):
        def responder(q):
            return Scenario(name="custom", answer=f"回答：{q}",
                            contexts=["上下文"], expected_hallucination=False)
        p = MockProvider(responder=responder)
        r = p.ask("测试问题")
        assert r.answer == "回答：测试问题"

    def test_sequence_and_responder_conflict(self):
        with pytest.raises(ValueError):
            MockProvider(sequence=["correct"],
                         responder=lambda q: SCENARIOS["correct"])

    def test_unknown_scenario(self):
        with pytest.raises(KeyError):
            MockProvider(scenario="不存在的场景")

    def test_token_usage_mocked(self):
        """Mock 模拟 token 返回，用于验证统计逻辑"""
        r = scenario_provider("correct").ask("问题")
        assert r.token_usage.available
        assert r.token_usage.total_tokens > 0

    def test_token_unavailable_for_empty(self):
        r = scenario_provider("empty").ask("问题")
        assert not r.token_usage.available
        assert r.token_usage.total_tokens is None

    def test_raw_carries_scenario_name(self):
        r = scenario_provider("wrong").ask("问题")
        assert r.raw["scenario"] == "wrong"

    def test_latency_recorded(self):
        p = MockProvider(scenario="correct", default_latency_ms=250)
        r = p.ask("问题")
        assert r.latency_ms == 250


# ============================================================
# 2. 八种故障注入 → 评测器判定（需求第二十一条核心）
# ============================================================
class TestFaultInjection:
    """
    逐场景验证评测器判定

    这是"测试评测系统本身"的核心：
    已知被测对象会返回什么，验证评测器是否判对。
    """

    @pytest.mark.parametrize(
        "scenario_name", sorted(SCENARIOS)
    )
    def test_evaluator_matches_expectation(self, scenario_name):
        sc = SCENARIOS[scenario_name]
        p = scenario_provider(scenario_name)
        resp = p.ask("测试问题")

        # 异常场景：分两类
        #   1) error 字段型（timeout / api_error）—— 必须能识别
        #   2) 特殊输入型（invalid_json）—— 供 Judge 测试，
        #      声明级验证不涉及，不在此断言
        if sc.expected_hallucination is None:
            if scenario_name == "invalid_json":
                pytest.skip("invalid_json 供 Judge 解析测试，不适用声明级验证")
            assert resp.error, f"{scenario_name} 应返回 error"
            return

        r = detect_hallucination(
            resp.answer, resp.context_str,
            should_refuse=(scenario_name in ("refusal", "empty")),
        )
        assert r["has_hallucination"] == sc.expected_hallucination, \
            f"{scenario_name} 判定错误：{r['reason']}"

    def test_correct_answer_passed(self):
        r = detect_hallucination(
            SCENARIOS["correct"].answer,
            SCENARIOS["correct"].contexts[0],
        )
        assert not r["has_hallucination"]

    def test_entity_swap_detected(self):
        """Phase 0 曾漏报的场景，Mock 应能稳定复现"""
        r = detect_hallucination(
            SCENARIOS["entity_swap"].answer,
            SCENARIOS["entity_swap"].contexts[0],
            forbidden_facts=["报告层"],
        )
        assert r["has_hallucination"]
        assert r["forbidden_hits"]

    def test_hallucinated_name_detected(self):
        """Phase 6 曾漏报的场景"""
        r = detect_hallucination(
            SCENARIOS["hallucination"].answer,
            SCENARIOS["hallucination"].contexts[0],
        )
        assert r["has_hallucination"], f"漏报：{r['reason']}"

    def test_numeric_error_detected(self):
        """Phase 6 曾漏报的场景"""
        r = detect_hallucination(
            SCENARIOS["numeric_error"].answer,
            SCENARIOS["numeric_error"].contexts[0],
        )
        assert r["has_hallucination"], f"漏报：{r['reason']}"

    def test_paraphrase_not_flagged(self):
        """同义改写不得误报"""
        r = detect_hallucination(
            SCENARIOS["paraphrase"].answer,
            SCENARIOS["paraphrase"].contexts[0],
        )
        assert not r["has_hallucination"], f"误报：{r['reason']}"

    def test_refusal_not_flagged(self):
        r = detect_hallucination(
            SCENARIOS["refusal"].answer, "", should_refuse=True)
        assert not r["has_hallucination"]


# ============================================================
# 3. 错误处理：不能把失败当通过（需求第二十六条）
# ============================================================
class TestErrorHandling:
    """
    最关键的一组测试

    需求原文：不要把 API failure 当成 pass
    """

    @pytest.mark.parametrize("scenario_name", ["timeout", "api_error"])
    def test_error_scenario_marked_as_error(self, scenario_name):
        """异常场景必须能通过 response.error 识别出来"""
        resp = scenario_provider(scenario_name).ask("问题")
        assert resp.error is not None
        assert not resp.ok

    def test_error_does_not_become_pass(self):
        """
        回归：Phase 0 的P0-001

        旧实现：API 失败 → 空串 → 判为"无幻觉" → 通过
        新实现：必须能区分「异常」与「空回答」
        """
        # 错误场景
        resp = scenario_provider("api_error").ask("问题")
        assert resp.error, "错误场景必须携带 error 字段"

        # 空回答场景（无错误）
        resp_empty = scenario_provider("empty").ask("问题")
        assert resp_empty.error is None

        # 两者必须可区分
        assert resp.ok != resp_empty.ok, \
            "error 场景与空回答场景必须可区分"

    def test_custom_error_provider(self):
        p = make_error_provider("自定义错误信息")
        r = p.ask("问题")
        assert r.error == "自定义错误信息"

    def test_error_repeated_calls_still_error(self):
        p = make_error_provider("持续错误")
        for _ in range(5):
            assert p.ask("问题").error == "持续错误"


# ============================================================
# 4. 稳定性检测（离线可测）
# ============================================================
class TestStabilityOffline:
    """稳定性检测本来需要调多次 API，Mock 使其可离线测试"""

    def test_inconsistent_answers_detectable(self):
        p = make_evaluator_scenario_provider(
            correct_answer="自动化测试框架分为基础层、用例层、数据层。",
            hallucinated_answer="自动化测试框架分为基础层、用例层、报告层。",
            context="自动化测试框架分层：基础层、用例层、数据层。",
        )
        answers = [p.ask("问题").answer for _ in range(4)]
        assert len(set(answers)) == 2, "应产生两种不同答案"

    def test_consistent_answers_stable(self):
        p = scenario_provider("correct")
        answers = [p.ask("问题").answer for _ in range(5)]
        assert len(set(answers)) == 1, "固定场景应始终返回相同答案"


# ============================================================
# 5. 多指标联动的离线测试
# ============================================================
class TestMultiMetricOffline:
    def test_correct_answer_passes_all_metrics(self):
        sc = SCENARIOS["correct"]
        ctx = "\n\n".join(sc.contexts)
        r = detect_hallucination(sc.answer, ctx)
        c = evaluate_correctness(sc.answer,
                                 required_facts=["基础层", "用例层", "数据层"])
        rel = evaluate_relevance(sc.answer, reference_answer=ctx)
        comp = evaluate_completeness(sc.answer,
                                      required_facts=["基础层", "用例层", "数据层"])

        assert not r["has_hallucination"]
        assert c.is_correct
        assert rel["is_relevant"]
        assert comp["is_complete"]

    def test_hallucination_fails_correctness(self):
        sc = SCENARIOS["entity_swap"]
        c = evaluate_correctness(sc.answer,
                                 required_facts=["基础层", "用例层", "数据层"])
        assert not c.is_correct, "实体调换应导致正确性不达标"

    def test_wrong_answer_fails_relevance(self):
        sc = SCENARIOS["wrong"]
        ctx = "\n\n".join(sc.contexts)
        rel = evaluate_relevance(sc.answer, reference_answer=ctx)
        assert not rel["is_relevant"] or rel["score"] < 0.5

    def test_refusal_correctness(self):
        """拒答场景：expected_behavior=refuse 时不应判为错误"""
        sc = SCENARIOS["refusal"]
        r = evaluate_refusal(sc.answer, should_refuse=True)
        assert r["is_correct"]
        assert r["mode"] == "correct_refusal"

    def test_empty_answer_over_refusal(self):
        """上下文有内容却空回答 → 过度拒答"""
        sc = SCENARIOS["empty"]
        r = evaluate_refusal(sc.answer, should_refuse=False)
        assert not r["is_correct"]


# ============================================================
# 6. 注册到 Provider 表
# ============================================================
class TestProviderRegistration:
    def test_mock_registered(self):
        from eval.providers.sut import get_provider, available_providers
        from eval.providers.mock import register_all
        register_all()
        assert "mock" in available_providers()

    def test_get_mock_provider(self):
        from eval.providers.sut import get_provider
        from eval.providers.mock import register_all
        register_all()
        p = get_provider("mock", scenario="correct")
        assert isinstance(p, MockProvider)
        assert p.ask("问题").answer == SCENARIOS["correct"].answer
