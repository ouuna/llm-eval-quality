"""
正确性与相关性单元测试
---------------------------------
验证两个 P0 缺陷已修复：

P0-2  expected_output 不参与判定 → 现在 required_facts 参与
P0-3  相关性方向反了              → 现在算answer↔required/reference
"""

import pytest

from eval.evaluators.correctness import (
    evaluate_correctness, evaluate_completeness, evaluate_relevance,
    evaluate_refusal, detect_context_conflict,
)


# ============================================================
# P0-2 验证：正确性真正生效
# ============================================================
class TestCorrectness:
    """旧实现：域内用例只要用词在上下文出现就算通过"""

    CTX = "自动化测试框架分层：基础层负责请求封装，用例层负责断言，数据层负责测试数据。"
    REQUIRED = ["基础层", "用例层", "数据层"]

    def test_correct_answer_passes(self):
        r = evaluate_correctness(
            "自动化测试框架分为基础层、用例层、数据层。",
            self.REQUIRED,
        )
        assert r.is_correct
        assert r.score == 1.0
        assert not r.missing_facts

    def test_missing_fact_fails(self):
        """旧实现会通过（用词都在上下文里），新实现必须判错"""
        r = evaluate_correctness(
            "自动化测试框架分为基础层、用例层。",
            self.REQUIRED,
        )
        assert not r.is_correct, "缺少必需事实应判错"
        assert "数据层" in r.missing_facts
        assert r.score == pytest.approx(2 / 3, abs=0.01)

    def test_irrelevant_answer_fails(self):
        """
        核心回归：答非所问必须判错
        旧实现下这句话的字符覆盖率很高（都是常用字），会通过
        """
        r = evaluate_correctness(
            "测试框架分为基础层、用例层、数据层。",
            self.REQUIRED,
            reference_answer="自动化测试框架分为基础层、用例层、数据层。",
        )
        assert r.is_correct  # 事实都在，只是不提"自动化"二字

        r2 = evaluate_correctness(
            "今天天气不错，适合出门散步。",
            self.REQUIRED,
        )
        assert not r2.is_correct, "完全无关的回答必须判错"
        assert r2.score == 0.0

    def test_forbidden_fact_wins_over_required(self):
        """命中禁止事实时直接判错，优先级高于 required 覆盖"""
        r = evaluate_correctness(
            "自动化测试框架分为基础层、用例层、数据层、业务层。",
            self.REQUIRED,
            forbidden_facts=["业务层"],
        )
        assert not r.is_correct
        assert "业务层" in r.wrong_facts
        assert r.reason.startswith("答案包含禁止内容")

    def test_acceptable_answers_used_as_fallback(self):
        r = evaluate_correctness(
            "改动代码后重新执行相关测试用例，确认没有引入新缺陷。",
            required_facts=[],
            acceptable_answers=["修改代码后重新执行相关测试用例，确认没有引入新缺陷"],
        )
        assert r.method == "acceptable_answers"
        assert r.is_correct

    def test_degraded_method_is_explicitly_marked(self):
        """只有 reference 时必须标注判定依据较弱"""
        r = evaluate_correctness(
            "基础层、用例层、数据层。",
            required_facts=[],
            reference_answer="自动化测试框架分为基础层、用例层、数据层。",
        )
        assert r.method == "reference_similarity_degraded"
        assert "判定依据较弱" in r.reason

    def test_insufficient_ground_truth_reported(self):
        """既无 required 也无 reference 时必须明确报"依据不足"，不编造分数"""
        r = evaluate_correctness("随便写点什么", required_facts=[])
        assert r.method == "insufficient_ground_truth"
        assert not r.is_correct

    def test_empty_answer(self):
        r = evaluate_correctness("", self.REQUIRED)
        assert not r.is_correct
        assert r.score == 0.0


# ============================================================
# P0-3 验证：相关性方向已修正
# ============================================================
class TestRelevanceDirectionFix:
    """
    旧实现：|question_chars ∩ answer_chars| / |question_chars|
    问题：完全无关的回答只要复述问题词就能得高分
    """

    def test_unrelated_answer_no_longer_gets_high_score(self):
        """
        核心回归：答非所问不得高分

        旧实现下这句话会因复述问题词而得高分
        """
        r = evaluate_relevance(
            "今天天气不错。",
            question="如何提高代码质量？",
            reference_answer="提高代码质量需要遵循规范、补充测试、进行代码评审。",
        )
        assert r["score"] < 0.5, f"无关回答不应得高分，实际 {r['score']}"
        assert not r["is_relevant"]

    def test_relevant_answer_high_score(self):
        r = evaluate_relevance(
            "提高代码质量需要遵循编码规范。",
            question="如何提高代码质量？",
            reference_answer="提高代码质量需要遵循规范、补充测试、进行代码评审。",
        )
        assert r["score"] > 0.5
        assert r["is_relevant"]

    def test_required_facts_is_primary_method(self):
        """有 required_facts 时应作为主要依据"""
        r = evaluate_relevance(
            "框架分为基础层、用例层、数据层。",
            required_facts=["基础层", "用例层", "数据层"],
            reference_answer="框架分为基础层、用例层、数据层。",
        )
        assert r["method"] == "required_facts_coverage"
        assert r["score"] == 1.0

    def test_answer_reference_similarity_direction(self):
        """相似度应是 answer∩reference / reference（问"是否答到点上"）"""
        r = evaluate_relevance(
            "自动化测试框架分为基础层、用例层、数据层。",
            reference_answer="自动化测试框架分为基础层、用例层、数据层。",
        )
        assert r["method"] == "answer_reference_similarity"
        assert r["score"] > 0.8

    def test_degraded_to_context(self):
        """只有 context 时必须标注降级"""
        r = evaluate_relevance(
            "基础层、用例层、数据层。",
            context="自动化测试框架分层：基础层、用例层、数据层。",
        )
        assert r["method"] == "answer_context_similarity_degraded"
        assert "未提供 required_facts" in r["reason"]

    def test_no_ground_truth_explicit(self):
        r = evaluate_relevance("任意回答")
        assert r["method"] == "insufficient_ground_truth"
        assert not r["is_relevant"]


# ============================================================
# 完整性
# ============================================================
class TestCompleteness:
    REQUIRED = ["冒烟测试验证核心功能", "回归测试修改代码后重新执行"]

    def test_complete_answer(self):
        r = evaluate_completeness(
            "冒烟测试验证核心功能是否可用，回归测试修改代码后重新执行用例。",
            self.REQUIRED,
        )
        assert r["is_complete"]
        assert r["score"] == 1.0

    def test_partial_answer(self):
        """对比型问题只答一半 → 不完整（这是旧实现漏掉的场景）"""
        r = evaluate_completeness(
            "冒烟测试验证核心功能是否可用。",
            self.REQUIRED,
        )
        assert not r["is_complete"]
        assert r["score"] == pytest.approx(0.5, abs=0.01)
        assert len(r["missing_facts"]) == 1

    def test_empty_answer(self):
        r = evaluate_completeness("", self.REQUIRED)
        assert not r["is_complete"]
        assert r["score"] == 0.0

    def test_no_required_facts_skipped(self):
        r = evaluate_completeness("任意回答", [])
        assert r["is_complete"]
        assert "跳过" in r["reason"]


# ============================================================
# 拒答
# ============================================================
class TestRefusal:
    def test_correct_refusal(self):
        r = evaluate_refusal("参考资料中未提及。", should_refuse=True)
        assert r["is_correct"]
        assert r["mode"] == "correct_refusal"

    def test_empty_is_refusal(self):
        r = evaluate_refusal("", should_refuse=True)
        assert r["is_correct"]

    def test_false_answer_is_error(self):
        """该拒答却作答 —— 最严重的失败模式"""
        r = evaluate_refusal(
            "可以 pip install python 来安装。",
            should_refuse=True,
            forbidden_facts=["pip install"],
        )
        assert not r["is_correct"]
        assert r["mode"] == "false_answer"
        assert r["forbidden_hits"] == ["pip install"]

    def test_over_refusal(self):
        """不该拒答却拒答 —— 方向不同的失败"""
        r = evaluate_refusal("未提及", should_refuse=False)
        assert not r["is_correct"]
        assert r["mode"] == "over_refusal"

    def test_correct_answer_when_should_answer(self):
        r = evaluate_refusal("等价类划分是...", should_refuse=False)
        assert r["is_correct"]
        assert r["mode"] == "correct_answer"

    def test_modes_are_distinguished(self):
        """两种失败模式必须可区分 —— 修复方向不同"""
        false_answer = evaluate_refusal("随便答", should_refuse=True)
        over_refusal = evaluate_refusal("未提及", should_refuse=False)
        assert false_answer["mode"] != over_refusal["mode"]


# ============================================================
# 上下文冲突
# ============================================================
class TestContextConflict:
    def test_no_conflict(self):
        r = detect_context_conflict("冒烟测试：验证核心功能是否可用。")
        assert not r["has_conflict"]

    def test_conflicting_context(self):
        ctx = ("冒烟测试：验证核心功能是否可用。"
               "冒烟测试：完整回归全部功能。")
        r = detect_context_conflict(ctx)
        assert r["has_conflict"], f"应检出冲突：{r}"
        assert r["conflicts"]

    def test_acknowledgement_detected(self):
        ctx = ("冒烟测试：验证核心功能是否可用。"
               "冒烟测试：完整回归全部功能。")
        r = detect_context_conflict(ctx, answer="两条信息存在冲突，无法确定。")
        assert r.get("acknowledged") is True

    def test_blind_answer_not_acknowledged(self):
        ctx = ("冒烟测试：验证核心功能是否可用。"
               "冒烟测试：完整回归全部功能。")
        r = detect_context_conflict(ctx, answer="验证核心功能是否可用。")
        assert r.get("acknowledged") is False


# ============================================================
# 确定性
# ============================================================
class TestDeterminism:
    @pytest.mark.parametrize("answer", [
        "自动化测试框架分为基础层、用例层、数据层。",
        "自动化测试框架分为基础层、用例层。",
        "完全无关的回答。",
    ])
    def test_correctness_deterministic(self, answer):
        req = ["基础层", "用例层", "数据层"]
        r1 = evaluate_correctness(answer, req)
        r2 = evaluate_correctness(answer, req)
        assert r1.score == r2.score
        assert r1.is_correct == r2.is_correct

    def test_relevance_deterministic(self):
        args = ("回答", ["事实一", "事实二"], "参考答案")
        r1 = evaluate_relevance(*args)
        r2 = evaluate_relevance(*args)
        assert r1["score"] == r2["score"]
