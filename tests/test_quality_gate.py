"""
pytest 集成层
---------------------------------
将 LLM 质量评测接入 pytest 框架，实现：
    - 逐用例断言与失败定位
    - 质量门禁（不达标时退出码非0，可用于 CI 阻断）
    - Allure / HTML 报告集成

运行：
    pytest tests/ -v
    pytest tests/ -v --html=reports/pytest_report.html

在 CI 中作为质量门禁使用（示例见.github/workflows/eval.yml）：
    pytest tests/ -v --json-report ...
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tests"))

from config_loader import load_cases, validate_cases  # noqa: E402
from evaluator import (  # noqa: E402
    check_hallucination,
    check_completeness,
    relevance_score,
    GATE,
    THRESHOLD_COMPLETENESS,
    THRESHOLD_HALLUCINATION,
)

# 这些用例会真实调用 LLM（通过 tests/evaluator.py → app.rag），
# 因此标记为 live：默认不跑，只有配置了 API Key 才跑。
#
# 为什么需要显式标记
# ----------------
# 它们的依赖（tests/evaluator.py + cases.csv）看不出「需要 API」，
# 按目录判定的自动marker 机制会把它们归进 offline，
# 结果默认运行时集体error：
#     OSError: 缺少配置：OPENAI_API_KEY
#
# 显式标记优先于自动判定，所以这里写了 pytestmark。
pytestmark = pytest.mark.live


# ---- 用例从 cases.csv 加载，可由业务方维护，无需改代码 ----
# 注意：在函数内延迟加载，避免 pytest 导入期执行导致路径解析异常
CASES_FILE = os.path.join(PROJECT_ROOT, "cases.csv")


def _load_cases():
    return load_cases(CASES_FILE)


# ---- 门禁阈值统一来自 config.yaml ----
MIN_PASS_RATE = GATE["overall_pass_rate"]
MIN_REFUSAL_RATE = GATE["refusal_rate_out_domain"]
MIN_HALLUC_FREE = GATE["hallucination_free_rate"]
MIN_RETRIEVAL_RATE = GATE["retrieval_hit_rate"]
MIN_STABILITY = GATE["avg_stability"]
MIN_RELEVANCE = GATE["avg_relevance"]


@pytest.fixture(scope="session")
def cache():
    """会话级缓存，避免重复调用大模型导致耗时与成本翻倍"""
    _c = {}
    return _c


@pytest.fixture(scope="session")
def eval_results(cache):
    """执行一次完整评测，供所有断言用例共享结果"""
    from evaluator import run_all_eval
    return run_all_eval(verbose=False)


def _case(eval_results, case_id):
    for c in eval_results["cases"]:
        if c["case_id"] == case_id:
            return c
    pytest.fail(f"未找到用例 {case_id}")


# ============================================================
# 逐用例断言
# ============================================================
@pytest.mark.parametrize(
    "case", _load_cases(),
    ids=[c["case_id"] for c in _load_cases()]
)
def test_case_quality(case, eval_results):
    """
    单用例质量断言

    判定规则按用例类别区分：
      - in_domain  ：必须无幻觉，且覆盖全部关键要点
      - out_domain ：必须正确拒答（无幻觉）
      - ambiguous  ：不得强行作答（无幻觉）
    """
    r = _case(eval_results, case["case_id"])
    cat = r["category"]

    assert r["hallucination_free"], (
        f"[{r['case_id']}] 检测到幻觉：{r['hallucination_reason']}"
    )

    if cat == "in_domain":
        if case.get("key_points"):
            assert r["is_complete"], (
                f"[{r['case_id']}] 回答不完整，缺失要点：{r['missing_points']}"
            )
        assert r["retrieval_count"] > 0, (
            f"[{r['case_id']}] 域内用例未召回到任何上下文"
        )


# ============================================================
# 整体质量门禁
# ============================================================
def test_no_hallucination(eval_results):
    """门禁：整体无幻觉率达标"""
    m = eval_results["metrics"]
    assert m["hallucination_free_rate"] >= MIN_HALLUC_FREE, (
        f"无幻觉率 {m['hallucination_free_rate']:.0%} 低于门禁 {MIN_HALLUC_FREE:.0%}"
    )


def test_refusal_ability(eval_results):
    """门禁：域外场景必须能正确拒答（防幻觉核心能力）"""
    m = eval_results["metrics"]
    assert m["refusal_rate_out_domain"] >= MIN_REFUSAL_RATE, (
        f"域外拒答率 {m['refusal_rate_out_domain']:.0%} 低于门禁 "
        f"{MIN_REFUSAL_RATE:.0%}—— 存在无依据编造行为，需加强 Prompt 约束"
    )


def test_retrieval_effectiveness(eval_results):
    """门禁：检索命中率达标"""
    m = eval_results["metrics"]
    assert m["retrieval_hit_rate"] >= MIN_RETRIEVAL_RATE, (
        f"检索命中率 {m['retrieval_hit_rate']:.0%} 低于门禁 {MIN_RETRIEVAL_RATE:.0%}"
    )


def test_answer_relevance(eval_results):
    """门禁：答案相关性达标"""
    m = eval_results["metrics"]
    assert m["avg_relevance"] >= MIN_RELEVANCE, (
        f"平均相关性 {m['avg_relevance']} 低于门禁 {MIN_RELEVANCE}"
    )


def test_stability(eval_results):
    """门禁：输出稳定性达标（不稳定输出无法进入生产）"""
    m = eval_results["metrics"]
    assert m["avg_stability"] >= MIN_STABILITY, (
        f"平均稳定性 {m['avg_stability']} 低于门禁 {MIN_STABILITY}"
    )


def test_overall_pass_rate(eval_results):
    """门禁：总体通过率达标"""
    m = eval_results["metrics"]
    assert m["overall_pass_rate"] >= MIN_PASS_RATE, (
        f"总体通过率 {m['overall_pass_rate']:.0%} 低于门禁 {MIN_PASS_RATE:.0%}"
    )


def test_in_domain_accuracy(eval_results):
    """门禁：域内用例准确率达标（有标准答案时必须答对）"""
    m = eval_results["metrics"]
    assert m["accuracy_in_domain"] >= GATE["accuracy_in_domain"], (
        f"域内准确率 {m['accuracy_in_domain']:.0%} 低于门禁"
        f" {GATE['accuracy_in_domain']:.0%}"
    )


def test_case_data_validity():
    """前置校验：用例数据本身必须合法（真实项目中的数据质量门禁）"""
    issues = validate_cases(_load_cases())
    assert not issues, "用例数据存在非法配置：\n" + "\n".join(issues)


# ============================================================
# 报告生成
# ============================================================
def test_generate_reports(eval_results):
    """生成 JSON 与 HTML 报告"""
    from evaluator import save_json, save_html

    json_path = save_json(eval_results)
    html_path = save_html(eval_results)

    assert os.path.exists(json_path), "JSON 报告生成失败"
    assert os.path.exists(html_path), "HTML 报告生成失败"
    assert os.path.getsize(html_path) > 1000, "HTML 报告内容异常"

    print(f"\n  JSON 报告：{json_path}")
    print(f"  HTML 报告：{html_path}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
