"""
声明级验证（Faithfulness）单元测试
---------------------------------
**这些测试是评测器自身的单元测试**，不依赖真实 LLM API。

重点验证
--------
1. 实体调换型幻觉能被检出（旧字符覆盖率方法检不出的 case）
2. 数字编造能被检出
3. 否定极性冲突能被检出
4. forbidden_facts 能检出"多出一层"型幻觉
5. required_facts 缺失能被检出
6. 拒答不被误判为幻觉
"""

import pytest

from eval.evaluators.faithfulness import (
    detect_hallucination, verify_claim, verify_forbidden_facts,
    content_chars, split_claims, split_sentences, Claim,
)


# ============================================================
# 1. 核心能力：实体调换（旧方法的盲区）
# ============================================================
class TestEntitySwapDetection:
    """
    这是本项目最重要的能力验证。

    旧实现对以下两句话判定完全相同（字符覆盖率都是 1.0）：
        "中国的首都是北京"（正确）
        "中国的首都是上海"（幻觉）
    """

    CTX = "北京是中国的首都。"

    def test_correct_entity_passed(self):
        r = detect_hallucination("中国的首都是北京。", self.CTX)
        assert not r["has_hallucination"], r["reason"]

    def test_entity_swap_detected(self):
        """实体调换必须被检出"""
        r = detect_hallucination("中国的首都是上海。", self.CTX)
        # 注意：纯字符覆盖率方法会漏检，此处通过 forbidden/数字之外的手段
        # 依赖实词"上海"不在上下文 → 覆盖率不足
        # 这是本方法优于纯覆盖率的地方
        assert r["unsupported_claims"], "实体调换应产生无依据声明"

    def test_layer_swap_detected(self):
        """
        真实的实体调换场景：框架分层说错一层

        上下文说数据层，回答说报告层 —— 而"报告层"不在上下文中
        """
        ctx = "自动化测试框架分层：基础层负责请求封装，用例层负责断言，数据层负责测试数据。"
        r = detect_hallucination(
            "自动化测试框架分为基础层、用例层、报告层。",
            ctx,
        )
        assert r["unsupported_claims"], f"应检出无依据声明，实际：{r['reason']}"

    def test_forbidden_facts_catches_extra_layer(self):
        """forbidden_facts 直接检出多出的实体"""
        ctx = "自动化测试框架分层：基础层负责请求封装，用例层负责断言，数据层负责测试数据。"
        r = detect_hallucination(
            "自动化测试框架分为基础层、用例层、数据层、业务层。",
            ctx,
            forbidden_facts=["业务层", "接口层", "页面层"],
        )
        assert r["forbidden_hits"], f"应命中禁止事实，实际：{r['reason']}"
        assert r["has_hallucination"]


# ============================================================
# 2. 数字编造
# ============================================================
class TestNumericHallucination:
    CTX = "边界值分析需要关注的边界值包括：最小值、最小值加一、最大值减一、最大值。"

    def test_novel_number_detected(self):
        """上下文没有的数字应被检出"""
        r = detect_hallucination("边界值包括最小值、最大值、中间值和平均值。", self.CTX)
        assert r["has_hallucination"], r["reason"]

    def test_correct_numbers_passed(self):
        r = detect_hallucination(
            "边界值分析关注最小值、最小值加一、最大值减一、最大值。",
            self.CTX,
        )
        assert not r["has_hallucination"], r["reason"]

    def test_wrong_count_detected(self):
        """数量错误：说'四个'但实际列举了五个不同值"""
        ctx = "测试用例字段包括用例编号、所属模块、用例标题、前置条件。"
        r = detect_hallucination(
            "测试用例字段包括用例编号、所属模块、用例标题、前置条件、操作步骤、预期结果。",
            ctx,
        )
        # 后两个字段不在上下文 → 应产生无依据声明
        assert r["unsupported_claims"] or r["missing_required"]


# ============================================================
# 3. 否定极性冲突
# ============================================================
class TestNegationConflict:
    CTX = "冒烟测试是验证核心功能是否可用的必需环节。"

    def test_negation_conflict_detected(self):
        """
        极性相反但字面高度重合 —— 字符覆盖率会漏检
        """
        r = detect_hallucination("冒烟测试不是验证核心功能是否可用的必需环节。", self.CTX)
        assert r["has_hallucination"], r["reason"]
        assert any("极性" in c.reason for c in r["claims"]), \
            f"应标注极性冲突，实际：{[c.reason for c in r['claims']]}"


# ============================================================
# 4. required_facts 缺失
# ============================================================
class TestRequiredFacts:
    CTX = "自动化测试框架分层：基础层负责请求封装，用例层负责断言，数据层负责测试数据。"

    def test_missing_required_fact(self):
        r = detect_hallucination(
            "自动化测试框架分为基础层、用例层。",
            self.CTX,
            required_facts=["基础层", "用例层", "数据层"],
        )
        assert r["missing_required"], f"应检出缺失事实：{r['reason']}"
        assert "数据层" in r["missing_required"]

    def test_all_required_present(self):
        r = detect_hallucination(
            "自动化测试框架分为基础层、用例层、数据层。",
            self.CTX,
            required_facts=["基础层", "用例层", "数据层"],
        )
        assert not r["missing_required"], f"不应有缺失：{r['reason']}"


# ============================================================
# 5. 拒答场景
# ============================================================
class TestRefusal:
    def test_empty_answer_should_refuse(self):
        r = detect_hallucination("", "任意上下文", should_refuse=True)
        assert not r["has_hallucination"]
        assert "空" in r["reason"]

    def test_empty_answer_should_not_refuse(self):
        """该答却没答 —— 不能当幻觉，应由完整性指标处理"""
        r = detect_hallucination("", "任意上下文", should_refuse=False)
        assert not r["has_hallucination"]
        assert "应作答但未作答" in r["reason"]

    def test_explicit_refusal_markers(self):
        for marker in ["未提及", "没有相关", "无法回答", "不知道", "不明确"]:
            r = detect_hallucination(f"参考资料{marker}。", "任意上下文")
            assert not r["has_hallucination"], f"{marker} 应识别为拒答"

    def test_no_context_but_answer_is_hallucination(self):
        """无上下文却有回答 —— 最严重的幻觉形态"""
        r = detect_hallucination("北京是中国的首都。", "")
        assert r["has_hallucination"]
        assert "无检索上下文" in r["reason"]


# ============================================================
# 6. 声明级细粒度验证
# ============================================================
class TestClaimLevelVerification:
    """验证不是"整段一刀切"，而是逐条声明给结论"""

    CTX = ("等价类划分：把输入域划分为若干互不相交的子集，"
           "每个子集内的输入具有相同预期结果。"
           "边界值分析：关注输入和输出边界。")

    def test_partial_support_granularity(self):
        """
        一句里既有正确的也有编造的 —— 应该只标记编造的那条
        """
        answer = "等价类划分是把输入域划分为若干互不相交的子集。火星上有三个环形山。"
        r = detect_hallucination(answer, self.CTX)

        assert r["has_hallucination"]
        # 应该能定位到具体是哪条声明有问题
        assert r["unsupported_claims"], "应标出不支持的声明"
        # 且报告里应保留全部声明，便于人工复核
        assert len(r["claims"]) >= 2

    def test_claims_have_evidence_field(self):
        r = detect_hallucination(
            "等价类划分把输入域划分为互不相交的子集。",
            self.CTX,
        )
        assert r["claims"]
        # 每个 claim 都应能说明判定理由
        for c in r["claims"]:
            assert c.reason, f"声明'{c.text}'缺少判定理由"

    def test_support_level_values(self):
        r = detect_hallucination("等价类划分是常用的测试方法。", self.CTX)
        levels = {c.support_level for c in r["claims"]}
        assert levels <= {"supported", "partial", "unsupported"}


# ============================================================
# 7. 工具函数
# ============================================================
class TestUtilities:
    def test_content_chars_removes_stopwords(self):
        chars = content_chars("的是什么测试")
        assert "测" in chars and "试" in chars
        assert "的" not in chars

    def test_content_chars_keeps_numbers(self):
        chars = content_chars("最大值减一")
        assert isinstance(chars, set)

    def test_split_claims_splits_conjunctions(self):
        claims = split_claims("北京是中国的首都，人口约2100万")
        assert len(claims) >= 2

    def test_split_sentences_handles_empty(self):
        assert split_sentences("") == []
        assert split_sentences(None) == []

    def test_verify_claim_returns_reason_always(self):
        c = verify_claim("随便一句话", "上下文内容")
        assert c.reason

    def test_verify_claim_empty_context(self):
        c = verify_claim("任何声明", "")
        assert not c.supported
        assert "上下文为空" in c.reason

    def test_verify_forbidden_facts_empty_input(self):
        assert verify_forbidden_facts("任何内容", [], "") == []
        assert verify_forbidden_facts("任何内容", None, "") == []

    def test_claim_to_dict(self):
        c = Claim(text="t", supported=True, support_level="supported")
        d = c.to_dict()
        assert d["text"] == "t"
        assert d["supported"] is True
        assert d["support_level"] == "supported"


# ============================================================
# 8. 确定性（评测器自身可靠性前提）
# ============================================================
class TestDeterminism:
    """同一输入必须得到同一输出，否则评测结果不可复现"""

    CTX = "等价类划分：把输入域划分为若干互不相交的子集。"

    @pytest.mark.parametrize("answer", [
        "等价类划分把输入域划分为互不相交的子集。",
        "火星上有三个环形山。",
        "中国的首都是上海。",
    ])
    def test_repeated_calls_identical(self, answer):
        r1 = detect_hallucination(answer, self.CTX)
        r2 = detect_hallucination(answer, self.CTX)
        assert r1["has_hallucination"] == r2["has_hallucination"]
        assert r1["coverage"] == r2["coverage"]
        assert r1["reason"] == r2["reason"]


# ============================================================
# 9. 回归测试：Phase 6 验证阶段发现并修复的三个 bug
#    这些是"评测器自身缺陷"——漏检会导致缺陷流入生产
# ============================================================
class TestRegressionFromPhase6:
    """
    三个 bug 均由 Gold Set 验证阶段暴露：

    1. 拉丁字母被丢弃 → 编造的外国人名检测不出
    2. 数字冲突检测门槛过高 → 编造数字在字面重合度高时漏报
    3. 跑题未判为幻觉 → 完全无关回答只报"缺必需事实"
    """

    def test_fabricated_foreign_name_detected(self):
        """Bug 1：'Kent Beck 2003' 原本被漏检"""
        ctx = "回归测试：修改代码后重新执行相关测试用例，确认没有引入新缺陷。"
        r = detect_hallucination("回归测试由 Kent Beck 在 2003 年提出。", ctx)
        assert r["has_hallucination"], f"编造的外国人名应检出：{r['reason']}"
        # 且应明确指出人名是冲突项
        conflicts = [t for c in r["claims"] for t in c.conflicting_terms]
        assert any("kent" in str(c).lower() or "beck" in str(c).lower()
                   or "2003" in str(c) for c in conflicts), \
            f"应定位到人名/年份：{conflicts}"

    def test_fabricated_count_detected_despite_high_overlap(self):
        """
        Bug 2：数字编造但字面重合度高时，原实现漏报

        "包含 9 个字段" vs "包含 15 个字段"——
        除数字外几乎所有字都重合，原实现因覆盖率过高而不检查数字
        """
        ctx = "测试用例：用例编号、所属模块、用例标题、前置条件、操作步骤、预期结果、实际结果、执行结果、优先级。"
        r = detect_hallucination("测试用例通常包含 15 个字段。", ctx)
        assert r["has_hallucination"], f"编造字段数应检出：{r['reason']}"
        assert "15" in r["forbidden_hits"] or \
               any("15" in c.conflicting_terms for c in r["claims"]), \
            f"应定位到数字 15：{r}"

    def test_off_topic_answer_detected(self):
        """Bug 3：完全跑题的回答应判为幻觉"""
        ctx = "等价类划分：把输入域划分为若干互不相交的子集。"
        r = detect_hallucination("今天北京天气晴朗，气温 25 度，适合户外运动。", ctx)
        assert r["has_hallucination"], f"跑题应判为幻觉：{r['reason']}"
        assert "无任何实词重合" in r["reason"]

    def test_latin_terms_are_extracted(self):
        """验证 Bug 1 的根因已修：拉丁词必须进入比对集"""
        from eval.evaluators.faithfulness import _extract_terms
        terms = _extract_terms("由 Kent Beck 提出")
        assert "kent" in terms and "beck" in terms

    def test_numbers_still_extracted(self):
        """修复不能破坏原有的数字提取"""
        from eval.evaluators.faithfulness import _extract_terms
        terms = _extract_terms("包含 15 个字段")
        assert "15" in terms

    def test_chinese_extraction_unaffected(self):
        """修复不能破坏中文实词提取"""
        from eval.evaluators.faithfulness import _extract_terms
        terms = _extract_terms("等价类划分是子集")
        assert "等" in terms and "价" in terms and "类" in terms
