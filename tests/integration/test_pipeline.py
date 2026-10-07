"""
端到端集成测试
--------------------------------
验证「部件拼起来能不能跑通」，而不是「单个函数对不对」。

三层链路
--------
    HTTP 服务 → HTTPProvider → Evaluator → 报告/门禁

单层测试全绿、拼起来不通的情况很常见：

  · Provider 读answer，服务端改成了 answer_text → 全链路静默取空
  · Evaluator 正常，runner 传错参数 → 指标算错但没人发现
  · 报告能生成，门禁拿到的指标名对不上 → 门禁永远「跳过」

这些只有真正跑通才暴露。

不用真实API 的部分
----------------
用 Mock Provider 验证评测链路本身（快、稳、可重复）；
真正需要网络的用例用 live 标记单独隔开。
把所有用例都绑在真实 API 上，会让整套测试慢且不稳定——
CI 上没人愿意等20 秒就为了确认一个字段名没写错。
"""

import json

import pytest

from eval.datasets import get_dataset
from eval.providers.mock import MockProvider, scenario_provider
from eval.providers.sut import SUTResponse
from eval.runner import evaluate_case, run_evaluation
from eval.schemas.dataset import EvalCase, GroundTruth
from eval.schemas.result import Status, validate_result


# ============================================================
# 一、Mock 全链路
# ============================================================
class TestMockPipeline:

    """
    用 Mock 跑完整评测链路。

    这类测试不依赖 API Key，可以进 CI 每次提交都跑。
    它验证的是「评测框架自身能否正常工作」。
    """

    def test_单数据集跑通(self):
        ds = get_dataset("smoke")
        result = run_evaluation(ds, use_mock=True, stability_repeat=2)

        rep = result.report
        assert rep.total == len(ds.cases)
        assert rep.passed + rep.failed + rep.errors == rep.total
        assert rep.metrics, "应产出指标"
        assert rep.started_at, "应记录开始时间"

    def test_报告结构完整(self):
        """
        报告是给人和 CI 看的，缺字段会让下游无法消费。

        结构是分层的：meta（运行元信息）/ summary（计数）/
        metrics（指标）/ gate（门禁）/ cases（明细）。
        这里按真实结构断言，不臆测字段名。
        """
        result = run_evaluation(get_dataset("smoke"), use_mock=True,
                                stability_repeat=2)
        data = result.report.to_dict()

        for key in ("meta", "summary", "metrics", "gate",
                    "latency", "cases", "failure_pareto"):
            assert key in data, f"报告缺少 {key}"

        meta = data["meta"]
        for key in ("dataset", "model", "provider", "started_at",
                    "evaluators"):
            assert key in meta, f"meta 缺少 {key}"

        summary = data["summary"]
        for key in ("total", "passed", "failed", "error", "pass_rate"):
            assert key in summary, f"summary 缺少 {key}"

    def test_指标可序列化(self):
        """
        指标必须能 JSON 化——CI 上传的制品是 JSON 文件。

        不可序列化的对象（比如 Decimal、自定义类）会在写文件时才炸，
        而那时评测已经跑完了，白白浪费 API 调用。
        """
        result = run_evaluation(get_dataset("smoke"), use_mock=True,
                                stability_repeat=2)
        raw = result.report.to_json() if hasattr(result.report, "to_json") \
            else json.dumps(result.report.to_dict(), ensure_ascii=False)
        parsed = json.loads(raw)
        assert parsed["metrics"]

    def test_结果自洽(self):
        """
        每条 CaseResult 都必须自洽。

        status=passed 却带着 FAILING 违规，说明判定逻辑有矛盾——
        这种问题在报告里看不出来，只有自检能发现。
        """
        result = run_evaluation(get_dataset("smoke"), use_mock=True,
                                stability_repeat=2)
        for case_result in result.report.cases:
            issues = validate_result(case_result)
            assert not issues, f"{case_result.case_id}: {issues}"

    def test_门禁被实际执行(self):
        """
        门禁必须真的跑过，而不是悄悄跳过。

        此前门禁配了一条 regression_pass_rate 规则，
        但没有任何代码产出该指标——它靠「算不出来就放行」蒙混了半年。
        所以这里要断言门禁确实产生了检查项。
        """
        result = run_evaluation(get_dataset("smoke"), use_mock=True,
                                stability_repeat=2)
        assert result.report.gate_message, "门禁报告不应为空"
        assert "Quality Gate" in result.report.gate_message

    def test_报告保存到磁盘(self, tmp_path):
        """报告要能落盘，否则 CI 上传不了制品"""
        result = run_evaluation(get_dataset("smoke"), use_mock=True,
                                stability_repeat=2)
        paths = result.save_all(str(tmp_path))

        for kind, path in paths.items():
            import os
            assert os.path.exists(path), f"{kind} 未生成：{path}"

    def test_组合数据集可用(self):
        """smoke+full 组合应正常工作"""
        ds = get_dataset("smoke+full")
        assert len(ds.cases) == len(get_dataset("smoke").cases) + \
            len(get_dataset("full").cases)


# ============================================================
# 二、评测链路对故障注入的反应
# ============================================================
class TestPipelineAgainstFaults:

    """
    用 Mock 注入的故障验证评测链路能否正确识别。

    这是「评测系统有没有用」的核心证明：
    如果注入的缺陷都发现不了，那对真实缺陷的评测也不可信。
    """

    @pytest.mark.parametrize("scenario,expect_hallucination", [
        ("correct", False),
        ("paraphrase", False),
        ("refusal", False),
        ("wrong", True),
        ("hallucination", True),
        ("entity_swap", True),
        ("numeric_error", True),
    ])
    def test_单场景判定(self, scenario, expect_hallucination):
        """每种注入故障都应被正确识别"""
        provider = scenario_provider(scenario)
        case = EvalCase(
            id="it_01", category="in_domain",
            question="自动化测试框架分为哪几层？",
            expected_behavior="answer",
            context=["自动化测试框架分层：基础层、用例层、数据层"],
            ground_truth=GroundTruth(
                reference_answer="基础层、用例层、数据层",
                required_facts=["基础层", "用例层", "数据层"],
            ),
        )

        r = evaluate_case(case, provider)
        detected = "HALLUCINATION" in r.violation_types()

        assert detected == expect_hallucination, \
            f"{scenario}: 期望幻觉={expect_hallucination}，实际={detected}"

    @pytest.mark.parametrize("scenario", ["timeout", "api_error"])
    def test_异常场景标记为error(self, scenario):
        """
        异常必须标记为 error，而不是当成「模型没话说」。

        这是全项目最重要的约定。
        """
        provider = scenario_provider(scenario)
        case = EvalCase(
            id="it_02", category="in_domain", question="测试",
            expected_behavior="answer", context=["上下文"],
            ground_truth=GroundTruth(reference_answer="a"),
        )

        r = evaluate_case(case, provider)
        assert r.status == Status.ERROR.value, \
            f"{scenario} 应为 error，实际 {r.status}"

    def test_幻觉率随故障注入上升(self):
        """
        同一批用例，注入幻觉后幻觉率应上升。

        这是评测器有效性的**相对判据**——
        比「绝对准确率」更能说明问题：
        不需要知道正确答案，只需要知道「坏答案被判得更差」。
        """
        ds = get_dataset("smoke")

        clean = run_evaluation(ds, use_mock=True, stability_repeat=2)
        clean_rate = clean.report.metrics["hallucination_rate"].value

        # 全部注入幻觉
        cases = [EvalCase(
            id=c.id, category=c.category, question=c.question,
            expected_behavior=c.expected_behavior, context=c.context,
            ground_truth=c.ground_truth,
        ) for c in ds.cases if c.expected_behavior == "answer"]

        provider = MockProvider(scenario="entity_swap")
        with_bad = [evaluate_case(c, provider) for c in cases]
        bad_rate = sum(
            1 for r in with_bad
            if "HALLUCINATION" in r.violation_types()
        ) / len(with_bad)

        assert bad_rate > clean_rate, (
            f"注入实体调换后幻觉率应上升：{bad_rate} vs 基线 {clean_rate}")

    def test_拒答场景不误判为幻觉(self):
        """
        正确拒答不应被算作幻觉。

        如果拒答被判成幻觉，域外问题的指标会全部失真。
        """
        provider = scenario_provider("refusal")
        case = EvalCase(
            id="it_03", category="out_domain",
            question="Python 怎么安装？",
            expected_behavior="refuse", context=[],
            ground_truth=GroundTruth(forbidden_facts=["pip install"]),
        )

        r = evaluate_case(case, provider)
        assert "HALLUCINATION" not in r.violation_types(), \
            f"正确拒答被判为幻觉：{r.violation_types()}"


# ============================================================
# 三、HTTP 端到端
# ============================================================
class TestHttpEndToEnd:

    """
    完整网络链路：HTTP → Provider → 评测。

    这是唯一能回答「隔着网络之后评测还准不准」的地方。
    """

    def test_健康检查通过(self, http_sut_provider):
        assert http_sut_provider.is_available(), \
            "被测服务应可用"

    def test_http调用能拿到回答(self, http_sut_provider):
        """网络调用能正常拿到非空回答"""
        resp = http_sut_provider.ask("什么是等价类划分？")
        assert resp.error is None, f"调用失败：{resp.error}"
        assert resp.answer.strip(), "回答不应为空"
        assert resp.ok

    def test_http_provider被正确注册(self):
        """Provider 工厂应能创建 http 类型"""
        from eval.providers.sut import get_provider
        p = get_provider("http", base_url="http://127.0.0.1:1")
        assert p.name == "http_sut"

    def test_unknown_provider报错(self):
        from eval.providers.sut import get_provider
        with pytest.raises(KeyError):
            get_provider("nonexistent")

    def test_服务不可达时标记error(self):
        """
        服务挂掉时必须标记 error，不能返回空回答。

        这个用例不需要真实服务：
        指向一个不存在的端口即可。
        """
        from eval.providers.http_provider import HTTPProvider
        p = HTTPProvider(base_url="http://127.0.0.1:1", timeout=3)
        resp = p.ask("测试")

        assert resp.error is not None, "服务不可达必须标记 error"
        assert not resp.ok

    def test_http与本地直调结果一致(self, sut_server):
        """
        对照实验：同一条用例，本地直调与 HTTP 调用应得到相同答案。

        这个对比很有价值：
        如果两者答案不同，说明链路上有信息丢失
        （比如 JSON 编码、字段截断、字符转义问题）。
        """
        from app import rag
        from eval.providers.http_provider import HTTPProvider

        q = "什么是冒烟测试？"

        direct = HTTPProvider(base_url=sut_server, rag_module=rag).ask(q)
        over_http = HTTPProvider(base_url=sut_server, timeout=60).ask(q)

        if direct.error or over_http.error:
            pytest.skip("被测系统未配置，跳过对照实验")

        assert direct.answer == over_http.answer, \
            ("同一问题经 HTTP 调用后答案发生变化，"
             f"说明链路上有信息丢失：\n"
             f"  直调：{direct.answer[:60]}\n"
             f"  HTTP：{over_http.answer[:60]}")

    def test_http评测链路可跑通(self, http_sut_provider):
        """把 HTTP Provider 接到评测器上跑单条用例"""
        case = EvalCase(
            id="it_http", category="in_domain",
            question="自动化测试框架分为哪几层？",
            expected_behavior="answer", context=[],
            ground_truth=GroundTruth(
                reference_answer="基础层、用例层、数据层",
                required_facts=["基础层", "用例层", "数据层"],
            ),
        )

        r = evaluate_case(case, http_sut_provider)
        assert r.status != Status.ERROR.value, \
            f"HTTP 链路应可评测，实际 error：{r.error}"
        assert "faithfulness" in r.scores, "应产出有据性分数"


# ============================================================
# 四、评测框架自身的可测性
# ============================================================
class TestFrameworkTestability:

    """
    元测试：验证「评测框架可被验证」。

    这类测试的价值在于：
    任何人都可以用同样的方法，
    独立复现本项目关于「评测器可靠」的结论，
    而不是只能相信报告里的数字。
    """

    def test_评测器可被独立验证(self):
        """validation 模块应可运行，且数据结构完整"""
        from eval.evaluators.validation import (
            ConfusionMatrix, validate_evaluator,
        )

        cm = ConfusionMatrix()
        cm.tp, cm.fn, cm.fp, cm.tn = 10, 2, 3, 20
        m = cm.metrics()

        for key in ("accuracy", "precision", "recall", "f1", "fpr", "fnr"):
            assert key in m, f"混淆矩阵缺少 {key}"

    def test_变异测试算子可运行(self):
        """
        变异算子应能真实产出变异样本。

        算子写了但没人调用，等于没有。
        """
        from eval.mutation import generate_mutations

        answer = "自动化测试框架分为基础层、用例层、数据层，包含 3 个部分。"
        context = "自动化测试框架分层：基础层、用例层、数据层。"

        muts = generate_mutations(answer, context, max_per_kind=2)
        assert muts, "变异测试应产出变异样本"
        assert all(hasattr(m, "name") for m in muts)

        # 「空答案」变异算子的产物**理应**是空串——
        # 它要验证的是「该答却没答」这种缺陷能否被识别。
        # 所以不能断言所有 mutated 都非空。
        for m in muts:
            if m.name == "空答案":
                assert m.mutated == "", "空答案变异应产出空串"
            else:
                assert m.mutated, f"{m.name} 变异后不应为空"

    def test_变异类型覆盖多种缺陷(self):
        """
        变异算子应覆盖多种缺陷模式。

        如果十类算子里只有一类能产出样本，
        那「检出率 83%」这种数字就没有意义——
        分母太小，什么都能检出。
        """
        from eval.mutation import generate_mutations

        answer = "自动化测试框架分为基础层、用例层、数据层，包含 3 个部分。"
        context = "自动化测试框架分层：基础层、用例层、数据层。"

        muts = generate_mutations(answer, context, max_per_kind=2)
        kinds = {m.name for m in muts}

        assert len(kinds) >= 5, \
            f"实际只覆盖 {len(kinds)} 类变异：{kinds}"

    def test_数据集可被加载与校验(self):
        for name in ("smoke", "full"):
            ds = get_dataset(name)
            assert ds.cases
            from eval.schemas.dataset import validate_dataset
            issues = validate_dataset(ds)
            assert not issues, f"{name} 数据集存在问题：{issues[:3]}"

    def test_指标定义可查(self):
        """
        每个指标都应能查到算法说明与局限说明。

        查不到说明报告里也没法展示，
        读报告的人只能凭猜。
        """
        from eval import thresholds

        for metric in ("faithfulness", "correctness", "relevance",
                       "completeness", "hallucination_rate"):
            assert thresholds.definition_of(metric), \
                f"{metric} 缺少算法说明"
            assert thresholds.caveat_of(metric), \
                f"{metric} 缺少局限说明"

    def test_阈值全部可查且在合理区间(self):
        from eval import thresholds
        for name, value in thresholds.all_thresholds().items():
            assert 0.0 <= value <= 1.0, f"{name}={value} 超出 [0,1]"

    def test_基线可保存与对比(self, tmp_path, monkeypatch):
        """
        基线机制应可独立验证。
        """
        from eval import baseline as bl

        p = tmp_path / "baseline.json"
        bl.save_baseline({"overall_pass_rate": 0.95}, path=str(p),
                         meta={"time": "t", "model": "m", "dataset": "smoke"})

        loaded = bl.load_baseline(str(p))
        assert loaded["metrics"]["overall_pass_rate"] == 0.95

        cmp = bl.compare(
            {"overall_pass_rate": 0.60}, loaded["metrics"],
            baseline_meta=loaded["meta"],
            current_meta={"model": "m", "dataset": "smoke"})
        assert cmp.has_regression, "从 0.95 掉到 0.60 必须判为退化"
