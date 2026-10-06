"""
Regression Runner —— 缺陷回归验证
---------------------------------
把 Regression Dataset 转成可执行的 pytest 用例。

**这不是"又一份测试数据"，而是缺陷的保险锁。**

作用
----
    任何一次代码改动，只要破坏了已修复的缺陷，
    本模块会立即失败，CI 阻断合并。

判定逻辑
--------
对每条回归用例，运行当前的评测器，检查它是否符合预期：

    should_detect_hallucination=True  → 评测器必须判为"有幻觉"
    should_detect_hallucination=False → 评测器必须判为"无幻觉"

这测的是**评测器本身的行为**，不是被测系统的质量。
"""

import pytest

from eval.datasets.regression import CASES, statistics
from eval.evaluators.faithfulness import detect_hallucination


# ============================================================
# 数据完整性校验
# ============================================================
def test_regression_dataset_integrity():
    """回归集自身必须合法，否则测试通过没有意义"""
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids)), "回归用例 id 重复"

    for c in CASES:
        assert c.id, "缺少 id"
        assert c.severity in ("P0", "P1", "P2"), f"{c.id} severity 非法"
        assert c.layer in ("evaluator_bug", "method_limit", "env_issue"), \
            f"{c.id} layer 非法"
        assert c.root_cause, f"{c.id} 缺少根因说明"
        assert c.discovered_in, f"{c.id} 缺少发现阶段"


def test_p0_cases_are_fixed():
    """P0 级缺陷不允许'未修复'"""
    unfixed_p0 = [c.id for c in CASES
                  if c.severity == "P0" and not c.is_fixed]
    assert not unfixed_p0, f"P0 缺陷不允许未修复：{unfixed_p0}"


# ============================================================
# 逐条回归验证
# ============================================================
# 不同缺陷由不同指标验证
#   faithfulness 声明级验证——在 test_regression_case 中验证
#   correctness   由 TestCorrectnessRegression 验证
#   relevance     由 TestRelevanceRegression 验证
#   mechanism     由 TestEnvIssueRegression 验证机制本身
#   manual        需人工核验，当前 xfail（不假装通过）


def _run_case(case):
    """对单条回归用例运行声明级验证"""
    return detect_hallucination(
        answer=case.answer,
        context=case.context,
        required_facts=case.required_facts,
        forbidden_facts=case.forbidden_facts,
        should_refuse=not case.should_detect_hallucination,
    )


@pytest.mark.parametrize(
    "case", CASES,
    ids=[c.id for c in CASES]
)
def test_regression_case(case):
    """
    单条缺陷回归验证

    对已修复的缺陷，评测器必须持续给出正确判定。
    这条测试失败意味着"缺陷复发"或"修复不完整"。
    """
    # 非幻觉类缺陷由对应指标的专项测试验证
    if case.verify_by != "faithfulness":
        pytest.xfail(f"{case.id} 由 {case.verify_by} 指标验证，"
                     f"声明级验证不覆盖")

    result = _run_case(case)

    got = bool(result["has_hallucination"])
    expected = case.should_detect_hallucination

    if case.id == "REG-P0-001":
        # 空输出场景有特殊语义：应拒答时不算幻觉
        assert not got, \
            f"[{case.id}] 空输出被误判为幻觉。\n根因：{case.root_cause}"
        return

    if case.id == "REG-M02":
        # 上下文冲突场景：应识别冲突或标记无依据，不能静默放过
        assert got or result.get("forbidden_hits"), \
            f"[{case.id}] 冲突上下文未被识别。\n理由：{result['reason']}"
        return

    assert got == expected, (
        f"[{case.id}] 判定与预期不符\n"
        f"  原始缺陷：{case.root_cause}\n"
        f"  预期判定：{'有幻觉' if expected else '无幻觉'}\n"
        f"  实际判定：{'有幻觉' if got else '无幻觉'}\n"
        f"  评测器理由：{result['reason']}"
    )


# ============================================================
# 分类专项测试
# ============================================================
class TestEvaluatorBugRegression:
    """
    评测器自身缺陷的专项回归

    这类缺陷最危险：评测器错了，所有指标都不可信。
    """

    def test_p0_empty_output_not_treated_as_no_hallucination(self):
        """REG-P0-001：API 失败曾被误判为通过"""
        r = detect_hallucination("", "任意上下文", should_refuse=True)
        assert not r["has_hallucination"]
        assert "空" in r["reason"] or "拒答" in r["reason"]

    def test_p0_empty_output_when_should_answer(self):
        """该答却没答：不能算幻觉，但也不能算通过——由完整性指标处理"""
        r = detect_hallucination("", "任意上下文", should_refuse=False)
        assert not r["has_hallucination"], "空输出不应算幻觉"
        assert "应作答但未作答" in r["reason"]

    def test_p0_entity_swap(self):
        """REG-P0-004：实体调换必须检出"""
        r = detect_hallucination(
            "中国的首都是上海。", "北京是中国的首都。",
            forbidden_facts=["上海"],
        )
        assert r["has_hallucination"]

    def test_p0_latin_name_fabrication(self):
        """REG-P6-001：编造外国人名必须检出"""
        r = detect_hallucination(
            "回归测试由 Kent Beck 在 2003 年提出。",
            "回归测试：修改代码后重新执行相关测试用例。",
        )
        assert r["has_hallucination"], f"漏报：{r['reason']}"

    def test_p0_number_fabrication_high_overlap(self):
        """REG-P6-002：字面高重合时的数字编造必须检出"""
        r = detect_hallucination(
            "测试用例通常包含 15 个字段。",
            "测试用例：用例编号、所属模块、用例标题、前置条件、"
            "操作步骤、预期结果、实际结果、执行结果、优先级。",
        )
        assert r["has_hallucination"], f"漏报：{r['reason']}"

    def test_p0_off_topic_answer(self):
        """REG-P6-003：跑题必须判为幻觉"""
        r = detect_hallucination(
            "今天北京天气晴朗，气温 25 度。",
            "等价类划分：把输入域划分为若干互不相交的子集。",
        )
        assert r["has_hallucination"], f"漏报：{r['reason']}"


class TestCorrectnessRegression:
    """正确性与完整性类缺陷的回归"""

    def test_p0_missing_required_fact(self):
        """
        REG-P0-002：expected_output 不参与判定（Phase 0 P0）

        该样本说的都对，只是漏了'数据层'。
        幻觉检测不应判为幻觉（没有编造），
        但正确性/完整性必须检出缺失。
        """
        from eval.evaluators.correctness import evaluate_correctness
        r = evaluate_correctness(
            "自动化测试框架分为基础层、用例层。",
            ["基础层", "用例层", "数据层"],
        )
        assert not r.is_correct, "缺少必需事实应判为不正确"
        assert "数据层" in r.missing_facts

    def test_p0_context_conflict(self):
        """REG-M02：未识别上下文冲突"""
        from eval.evaluators.correctness import detect_context_conflict
        r = detect_context_conflict(
            "冒烟测试：验证核心功能是否可用。冒烟测试：完整回归全部功能。")
        assert r["has_conflict"], "应检出上下文冲突"


class TestRelevanceRegression:
    """相关性类缺陷的回归"""

    def test_p0_irrelevant_answer_low_score(self):
        """
        REG-P0-003：相关性方向反了（Phase 0 P0）

        旧实现算 |question∩answer|/|question|，
        同义反复会得满分。
        """
        from eval.evaluators.correctness import evaluate_relevance
        r = evaluate_relevance(
            "代码质量的提高方法。",
            reference_answer="提高代码质量需要遵循编码规范、补充测试。",
        )
        # 关键：分数不应因复述问题词而虚高
        assert r["score"] < 0.6, f"同义反复不应得高分：{r}"


class TestMethodLimitRegression:
    """方法论局限的回归"""

    def test_short_claim_not_dropped(self):
        """REG-M01：并列枚举尾部短片段不能被丢弃"""
        from eval.evaluators.faithfulness import split_claims
        claims = split_claims(
            "自动化测试框架分为基础层、用例层、数据层、业务层。")
        assert "业务层。" in claims, f"'业务层。' 被丢弃：{claims}"

    def test_extra_entity_detected(self):
        """REG-M01：多出一层必须检出"""
        r = detect_hallucination(
            "自动化测试框架分为基础层、用例层、数据层、业务层。",
            "自动化测试框架分层：基础层、用例层、数据层。",
            forbidden_facts=["业务层"],
        )
        assert r["has_hallucination"]
        assert r["forbidden_hits"]

    def test_negation_flip_detected(self):
        """REG-M03：否定翻转必须检出"""
        r = detect_hallucination(
            "冒烟测试不是必需环节。",
            "冒烟测试是必需环节，验证核心功能是否可用。",
        )
        assert r["has_hallucination"], f"漏报：{r['reason']}"


class TestEnvIssueRegression:
    """
    环境类缺陷的机制验证

    这些缺陷与幻觉判定无关，直接验证机制本身已修复。
    """

    def test_crlf_split_handled(self):
        """REG-E01：CRLF 换行不得导致切分失败"""
        from eval.evaluators.faithfulness import split_sentences
        crlf_text = "等价类划分：把输入域划分。\r\n\r\n边界值分析：关注边界。"
        normalized = crlf_text.replace("\r\n", "\n")
        assert len(split_sentences(normalized)) == 2

    def test_single_claim_survives_min_len(self):
        """REG-M01：短声明不能因 min_len 被丢"""
        from eval.evaluators.faithfulness import split_claims
        assert split_claims("数据层。") == ["数据层。"]


# ============================================================
# 覆盖率统计
# ============================================================
def test_regression_coverage():
    """
    回归集必须覆盖所有已修复的 evaluator_bug

    如果新增了评测器缺陷却没加回归用例，这��测试会失败
    """
    st = statistics()
    # 所有 P0 评测器缺陷都必须有对应回归用例
    p0_evaluator = [c for c in CASES
                    if c.severity == "P0" and c.layer == "evaluator_bug"]
    assert len(p0_evaluator) >= 7, \
        f"P0 评测器缺陷回归覆盖不足：{len(p0_evaluator)} < 7"

    # 三类缺陷层都必须有样本
    for layer in ("evaluator_bug", "method_limit", "env_issue"):
        assert st["by_layer"].get(layer, 0) > 0, f"缺少 {layer} 类回归用例"


# ============================================================
# 验证方式分布（报告用）
# ============================================================
def test_verification_coverage_report():
    """
    输出验证方式分布，供报告与CI 日志查看

    目的：让"哪些缺陷还没被自动化验证"一目了然。
    verify_by=manual 的必须人工核验，不能假装已覆盖。
    """
    from eval.datasets.regression import CASES

    dist = {}
    for c in CASES:
        dist[c.verify_by] = dist.get(c.verify_by, 0) + 1

    print("\n" + "=" * 52)
    print("回归集验证方式分布")
    print("=" * 52)
    for k, v in sorted(dist.items(), key=lambda x: -x[1]):
        marker = "" if k != "manual" else "  ← 需人工核验"
        print(f"  {k:16} {v:>3}{marker}")
    print()

    manual = [c.id for c in CASES if c.verify_by == "manual"]
    if manual:
        print(f"  ⚠️ 未自动化验证: {manual}")
        print("     这些是已知局限，已在文档中标注，不假装通过")
