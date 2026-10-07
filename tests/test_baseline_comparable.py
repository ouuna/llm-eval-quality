"""
基线可比性校验测试
------------------
为什么需要
----------
基线对比的前提是「同一个东西的两个版本」。

但早先的实现根本不校验这件事：拿 Mock 跑出来的基线去比真实 API 的结果，
照样能算出一张格式整齐、数字漂亮的对比表。
人眼扫过去只会觉得「哦，有对比」，很难意识到它是无效的。

这类问题比直接报错危险——报错会被发现，无效的数字会被当成结论。
"""

import pytest

from eval.baseline import check_comparable, compare


# ============================================================
# 可比性判定
# ============================================================

def test_同模型同数据集_可比():
    ok, reason = check_comparable(
        {"model": "glm-4-flash", "dataset": "smoke"},
        {"model": "glm-4-flash", "dataset": "smoke"},
    )
    assert ok, reason


def test_mock基线对真实模型_不可比():
    """本项目真实存在的基线是 mock 跑出来的"""
    ok, reason = check_comparable(
        {"model": "mock", "dataset": "smoke"},
        {"model": "glm-4-flash", "dataset": "smoke"},
    )
    assert not ok
    assert "不可比" in reason or "Mock" in reason or "mock" in reason


def test_数据集不同_不可比():
    """8 条的 smoke 基线不能用来比27 条的 smoke+full"""
    ok, reason = check_comparable(
        {"model": "glm-4-flash", "dataset": "smoke"},
        {"model": "glm-4-flash", "dataset": "smoke+full"},
    )
    assert not ok
    assert "数据集不同" in reason


def test_模型不同但都是真实模型_仍判可比():
    """
    换模型是真实项目里最常见的场景，不该被拦下——
    换模型后指标变化恰恰是最需要被看见的。
    """
    ok, reason = check_comparable(
        {"model": "glm-4-flash", "dataset": "smoke"},
        {"model": "glm-4-plus", "dataset": "smoke"},
    )
    assert ok, reason


# ============================================================
# compare 的行为
# ============================================================

def test_不可比时不输出对比表格():
    """
    关键：不可比时不能照样打印一张看起来很专业的表格。
    """
    baseline = {"overall_pass_rate": 0.95}
    current = {"overall_pass_rate": 0.30}

    cmp = compare(
        current, baseline,
        baseline_meta={"model": "mock", "dataset": "smoke"},
        current_meta={"model": "glm-4-flash", "dataset": "smoke"},
    )

    text = cmp.text()
    assert not cmp.comparable
    assert "不可比" in text
    assert "已跳过逐项对比" in text
    # 不能出现数字表格：既不能有"持平"行，也不能有逐项判定结论
    assert "持平" not in text
    assert "改善" not in text


def test_不可比时不产生任何退化判定():
    """
    这是最重要的一条。
    拿 mock 基线比真实结果，数字必然大幅「退化」，
    如果照单全收，CI 会永远报退化，而实际上什么都没变。
    """
    cmp = compare(
        {"overall_pass_rate": 0.30, "hallucination_rate": 0.40},
        {"overall_pass_rate": 0.95, "hallucination_rate": 0.02},
        baseline_meta={"model": "mock", "dataset": "smoke"},
        current_meta={"model": "glm-4-flash", "dataset": "smoke"},
    )

    assert cmp.has_regression is False, \
        "不可比时不得判为退化——那是数据来源不同，不是质量退化"
    assert cmp.regressions == []


def test_不可比时deltas为空():
    cmp = compare(
        {"overall_pass_rate": 0.30},
        {"overall_pass_rate": 0.95},
        baseline_meta={"model": "mock", "dataset": "smoke"},
        current_meta={"model": "glm-4-flash", "dataset": "smoke"},
    )
    assert cmp.deltas == [], "不可比时不应产生任何逐项差值"


def test_可比时正常检出退化():
    """防误报：真的可比且真的退化时，必须检出"""
    cmp = compare(
        {"overall_pass_rate": 0.60},
        {"overall_pass_rate": 0.95},
        baseline_meta={"model": "glm-4-flash", "dataset": "smoke"},
        current_meta={"model": "glm-4-flash", "dataset": "smoke"},
    )
    assert cmp.comparable
    assert cmp.has_regression
    assert "退化" in cmp.text()


def test_不传元信息时保持向后兼容():
    """
    老调用方（测试、demo）不传 meta，不能因此崩掉。
    兼容性本身也是一种需要测的行为。
    """
    cmp = compare({"overall_pass_rate": 0.60},
                  {"overall_pass_rate": 0.95})
    assert cmp.comparable
    assert cmp.has_regression


def test_序列化包含可比性字段():
    """报告落盘后也要能看出这次对比到底算不算数"""
    cmp = compare(
        {"overall_pass_rate": 0.30},
        {"overall_pass_rate": 0.95},
        baseline_meta={"model": "mock", "dataset": "smoke"},
        current_meta={"model": "glm-4-flash", "dataset": "smoke"},
    )
    d = cmp.to_dict()
    assert d["comparable"] is False
    assert d["incomparable_reason"]
    assert d["has_regression"] is False