"""
Evaluator 统一返回结构测试
--------------------------------
需求五要求：
    每种 evaluator 返回统一结构
    {"score": 0.91, "passed": true, "reason": "...", "evaluator": "embedding"}
    不要只返回 True / False

这组测试守住这个契约。

为什么要测得这么细
------------------
「统一结构」听起来简单，实际最容易出现的情况是：
某个评测器改了一版，返回结构就和其他的不一样了，
而调用方还在按旧的字段名取值——
表现是「悄悄取到 None」，不报错，只是指标一直是 0。

所以这里逐个断言字段存在、类型正确、语义正确。
"""


import pytest

from eval.evaluators.base import (
    EVALUATOR_NAMES, EvalResult, to_result,
)


class TestEvalResult:

    def test_五个核心字段齐全(self):
        """需求五点名的四个字段 + evaluator 标识"""
        r = EvalResult.ok(0.91, True, "覆盖率达标", "correctness")
        assert r.score == 0.91
        assert r.passed is True
        assert r.reason == "覆盖率达标"
        assert r.evaluator == "correctness"

    def test_details承载额外信息(self):
        """评测器特有的信息放details，不丢"""
        r = EvalResult.ok(0.8, True, "ok", "relevance",
                          method="context_similarity", extra=[1, 2])
        assert r.details["method"] == "context_similarity"
        assert r.details["extra"] == [1, 2]

    def test_unavailable与score0必须区分(self):
        """
        需求五的隐含要求，也是本项目反复强调的原则：

            「没测到」≠「测了，结果是 0」

        score=None 表示不可用，score=0.0 表示确实是零。
        混用会让门禁把「无法判定」当成「质量完美」。
        """
        un = EvalResult.unavailable("未配置 API", "judge")
        zero = EvalResult.ok(0.0, False, "确实一分没有", "judge")

        assert un.score is None and not un.available
        assert zero.score == 0.0 and zero.available
        assert un.score != zero.score

    def test_to_dict可序列化(self):
        import json
        r = EvalResult.ok(0.5, False, "跑题", "relevance", method="ctx")
        raw = json.dumps(r.to_dict(), ensure_ascii=False)
        assert "relevance" in raw

    def test_str可读(self):
        r = EvalResult.ok(0.5, False, "跑题", "relevance")
        s = str(r)
        assert "relevance" in s and "跑题" in s


class TestToResultAdapter:
    """适配器：把既有多种形态收敛成统一结构"""

    def test_包装dict(self):
        r = to_result({"score": 0.5, "is_relevant": False,
                       "reason": "跑题", "method": "ctx"}, "relevance")
        assert r.score == 0.5
        assert r.passed is False
        assert r.evaluator == "relevance"
        assert r.details["method"] == "ctx"

    @pytest.mark.parametrize("alias", [
        "passed", "is_correct", "is_complete", "is_relevant",
    ])
    def test_识别全部历史别名(self, alias):
        """
        历史上有四种「通过」字段名。
        适配器必须都认，否则统一结构就统一不了。
        """
        r = to_result({"score": 1.0, alias: True, "reason": "ok"},
                      "correctness")
        assert r.passed is True, f"未识别别名 {alias}"

    def test_包装EvalResult时补evaluator(self):
        r0 = EvalResult.ok(1.0, True, "ok", "")
        r = to_result(r0, "correctness")
        assert r.evaluator == "correctness"

    def test_包装dataclass(self):
        from eval.evaluators.correctness import CorrectnessResult

        r = to_result(CorrectnessResult(score=1.0, passed=True,
                                       reason="ok"), "correctness")
        assert r.score == 1.0
        assert r.evaluator == "correctness"


class TestAllEvaluatorsUnified:
    """
    逐个确认所有评测器都能给出统一结构。

    这是需求五的落地点——不是「有一个 base 类」就够了，
    而是每个实际使用的评测器都能走通。
    """

    def test_正确性(self):
        from eval.evaluators.correctness import evaluate_correctness
        r = evaluate_correctness("划分为基础层、用例层、数据层。",
                                 required_facts=["基础层", "用例层", "数据层"])
        assert isinstance(r, EvalResult)
        assert r.evaluator == "correctness"
        assert r.passed is True

    def test_正确性保留旧字段(self):
        """
        向后兼容检查。

        改造后不能破坏既有调用方——
        runner、tests 都还在用 r.is_correct / r.missing_facts。
        """
        from eval.evaluators.correctness import evaluate_correctness
        r = evaluate_correctness("只有基础层。",
                                 required_facts=["基础层", "用例层"])
        assert r.is_correct is False
        assert "用例层" in r.missing_facts
        assert r.method

    def test_完整性可统一化(self):
        from eval.evaluators.correctness import evaluate_completeness
        raw = evaluate_completeness("覆盖了基础层", ["基础层", "用例层"])
        r = to_result(raw, "completeness")
        assert r.evaluator == "completeness"
        assert r.score is not None

    def test_相关性可统一化(self):
        from eval.evaluators.correctness import evaluate_relevance
        raw = evaluate_relevance("今天天气不错", required_facts=["等价类划分"])
        r = to_result(raw, "relevance")
        assert r.evaluator == "relevance"

    def test_幻觉检测可统一化(self):
        from eval.evaluators.faithfulness import (
            detect_hallucination, to_eval_result,
        )
        raw = detect_hallucination(
            "自动化测试框架分为基础层、用例层、报告层。",
            "自动化测试框架分层：基础层、用例层、数据层。")
        r = to_eval_result(raw)
        assert r.evaluator == "faithfulness"
        assert r.passed is False          # 检出实体调换
        # 原始信息一点都不能丢
        assert r.details["claims"]
        assert "unsupported_claims" in r.details

    def test_拒答可统一化(self):
        from eval.evaluators.correctness import evaluate_refusal
        raw = evaluate_refusal("参考资料中未提及。", should_refuse=True)
        r = to_result(raw, "refusal")
        assert r.evaluator == "refusal"
        assert r.passed is True

    def test_所有评测器名都在注册表里(self):
        """名称注册表要与实际可用的评测器对齐"""
        for name in ("correctness", "completeness", "relevance",
                     "faithfulness"):
            assert name in EVALUATOR_NAMES, f"注册表缺少 {name}"


class TestCollectResults:
    """runner 的统一出口"""

    def test_聚合各评测器结果(self):
        from eval.runner import collect_evaluator_results, overall_verdict

        faith = {"has_hallucination": False, "coverage": 1.0,
                 "reason": "全部有据"}
        corr = {"score": 0.9, "is_correct": True, "reason": "覆盖 3/3"}
        comp = {"score": 1.0, "is_complete": True, "reason": "完整"}
        rel = {"score": 1.0, "is_relevant": True, "reason": "切题"}
        ref = {"is_correct": True, "refused": False, "reason": "正常作答"}

        results = collect_evaluator_results(
            None, "", "", faith, corr, comp, rel, ref)

        assert len(results) == 5
        for r in results:
            assert isinstance(r, EvalResult)
            assert r.evaluator, "每个结果都必须标明来源评测器"

        verdict = overall_verdict(results)
        assert verdict["passed"] is True
        assert verdict["failed_evaluators"] == []

    def test_任一不通过则整体不通过(self):
        from eval.runner import collect_evaluator_results, overall_verdict

        faith = {"has_hallucination": True, "coverage": 0.4, "reason": "无依据"}
        corr = {"score": 0.9, "is_correct": True, "reason": "ok"}
        comp = {"score": 1.0, "is_complete": True, "reason": "ok"}
        rel = {"score": 1.0, "is_relevant": True, "reason": "ok"}
        ref = {"is_correct": True, "refused": False, "reason": "ok"}

        verdict = overall_verdict(
            collect_evaluator_results(None, "", "", faith, corr, comp, rel, ref))

        assert verdict["passed"] is False
        assert "faithfulness" in verdict["failed_evaluators"]
        assert "reasons" in verdict

    def test_无法判定不算失败(self):
        """
        passed=None 不计入失败。

        与 pass_rate 排除 error 同一思路：
        「没测到」不等于「没通过」。
        """
        from eval.runner import overall_verdict

        results = [
            EvalResult.ok(1.0, True, "ok", "correctness"),
            EvalResult.unavailable("未配置", "judge"),
        ]
        verdict = overall_verdict(results)
        assert verdict["passed"] is True
        assert verdict["undecided_evaluators"] == ["judge"]
