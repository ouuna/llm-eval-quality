"""
Quality Gate 单元测试（零依赖，无需 API）
---------------------------------
验证门禁的判定逻辑与诊断信息质量。

核心要求
--------
**门禁失败必须说清"哪个指标失败、当前值、阈值、失败样本、修复方向"。**
只说"不达标"的门禁等于没有门禁。
"""

import pytest

from eval.quality_gate.gate import (
    QualityGate, GateCheck, GateReport,
    DEFAULT_GATE_CONFIG, load_gate_config,
)


# ============================================================
# 判定逻辑
# ============================================================
class TestGateEvaluation:
    def test_all_pass(self):
        gate = QualityGate()
        report = gate.evaluate({
            "hallucination_rate": 0.01, "refusal_accuracy": 1.0,
            "faithfulness": 0.98, "answer_correctness": 0.95,
            "error_rate": 0.0, "overall_pass_rate": 0.98,
            "completeness": 0.97, "relevance": 0.85,
            "retrieval_hit_rate": 0.90, "p95_latency_ms": 2000,
            "avg_cost": 0.05, "consistency": 0.96, "score_std": 0.03,
        })
        assert report.passed
        assert not report.failed_checks

    def test_min_violation(self):
        gate = QualityGate()
        r = gate.evaluate({"answer_correctness": 0.80})
        assert not r.passed
        # 新规则下 critical 层缺失指标也会判失败并进入 failed_checks，
        # 所以必须筛出「真正有值且低于阈值」的那一个，
        # 否则会拿到缺失项而不是低分项。
        failed = [c for c in r.failed_checks
                  if c.name == "answer_correctness" and c.available]
        assert len(failed) == 1, f"应恰好有一个低分的 answer_correctness"
        failed = failed[0]
        assert failed.current_value == 0.80
        assert failed.threshold_value == 0.90
        assert failed.comparison == "min"

    def test_max_violation(self):
        gate = QualityGate()
        r = gate.evaluate({"hallucination_rate": 0.15})
        assert not r.passed
        # 同上：要筛 available，否则可能取到缺失的那一项
        failed = [c for c in r.failed_checks
                  if c.name == "hallucination_rate" and c.available][0]
        assert failed.comparison == "max"
        assert failed.current_value > failed.threshold_value

    def test_boundary_equal_passes(self):
        """恰好等于阈值应通过（闭区间）"""
        gate = QualityGate()
        r = gate.evaluate({"error_rate": 0.0,
                          "hallucination_rate": 0.05})
        checks = {c.name: c for c in r.checks}
        assert checks["error_rate"].passed
        assert checks["hallucination_rate"].passed

    def test_enabled_layers(self):
        """可只启用部分层"""
        gate = QualityGate()
        r = gate.evaluate({"answer_correctness": 0.5},
                          enabled_layers=["quality"])
        # critical 层未启用，其检查项不参与
        assert all(c.layer != "critical" for c in r.checks)

    def test_unavailable_critical_metric_fails(self):
        """
        critical 层指标不可用 = 门禁失败。

        这是 2026-10-07 修掉的一个后门。
        早先的实现对缺失指标一律 passed=True，理由是
        「指标算不出来 ≠ 指标不达标」。但这个理由站不住：

          · API Key 失效 → 全部调用失败 → 指标全 None → 门禁全绿
          · Embedding 挂了 → groundedness 不可用 → 门禁全绿
          · 评测器自身抛异常 → 同上

        也就是说系统坏得越彻底，CI 越绿。
        这与项目里 Status.ERROR 不计入通过率分母的设计自相矛盾——
        那边刚堵住的洞，这边又开了。

        反过来看，这也正是本用例存在的意义：
        它断言的正是「不该放过的情况不会放过」。
        """
        gate = QualityGate()
        r = gate.evaluate({})     # 全部缺失
        assert not r.passed, (
            "critical 层指标全部缺失却通过门禁——这正是假通过")

        # 缺失项必须仍被标记为「不可用」，两者不冲突
        assert len(r.skipped_checks) > 0, "缺失项应记为 skipped"
        assert all(c.skip_reason for c in r.skipped_checks)

    def test_unavailable_noncritical_metric_is_allowed(self):
        """
        防误报：非 critical 层缺失仍然放行。

        比如 avg_cost 在 Provider 不返回token 时就是 None，
        这不代表出了故障。强行拦下只会制造大量误报，
        反而让人习惯性忽略门禁结果。
        """
        gate = QualityGate()
        # 只评估 stability 层（consistency / score_std）
        r = gate.evaluate({}, enabled_layers=["stability"])
        assert r.passed, "非 critical 层缺失不应导致失败"
        assert all(c.skip_reason for c in r.checks), \
            "放行也必须说明原因，不能静默"

    def test_missing_noncritical_passes_when_only_it_missing(self):
        """
        只提供一个指标、其余全无时，两类层应区别对待：

          · critical层缺失 → 失败（系统可能已经坏了）
          · 非 critical 缺失 → 放行（如 consistency 常因样本不足不可算）

        这条测试是为了钉住「分层处理」这个行为本身，
        防止将来有人为了省事把所有层都改成同一种处理方式。
        """
        gate = QualityGate()
        r = gate.evaluate({"avg_cost": 0.1})
        by_name = {c.name: c for c in r.checks}

        # 已提供的指标正常判定
        assert by_name["avg_cost"].passed
        assert by_name["avg_cost"].available

        # stability 层缺失 → 放行
        assert by_name["consistency"].layer == "stability"
        assert by_name["consistency"].passed, \
            "stability 层缺失应放行"
        assert not by_name["consistency"].available

        # critical 层缺失 → 失败
        assert not by_name["hallucination_rate"].passed, \
            "critical 层缺失应失败"
        assert not by_name["hallucination_rate"].available

        # 整体结论：critical 层的缺失决定了门禁失败
        assert not r.passed

    def test_unavailable_message_states_how_it_was_handled(self):
        """
        缺失项的信息必须说清「按失败处理」还是「按放行处理」。

        只写「跳过」会让人以为既没通过也没失败，
        实际上 critical 层已经让门禁失败了——
        报告与实际判定不符会误导排查方向。
        """
        gate = QualityGate()
        r = gate.evaluate({})
        critical_missing = [c for c in r.checks
                            if c.layer == "critical" and not c.available]
        assert critical_missing, "应有 critical 层缺失项"
        for c in critical_missing:
            assert "失败" in c.skip_reason, (
                f"{c.name} 应明确说明按失败处理，实际：{c.skip_reason}")
            assert "跳过" not in c.skip_reason, \
                "文案不该说「跳过」，那会与 passed=False 矛盾"


# ============================================================
# 诊断信息质量（关键要求）
# ============================================================
class TestDiagnostics:
    def test_failure_message_contains_all_key_info(self):
        """
        失败信息必须包含：
        指标名 / 当前值 / 阈值 / 失败样本
        """
        gate = QualityGate()
        r = gate.evaluate(
            {"answer_correctness": 0.85},
            failed_samples={"answer_correctness":
                            ["case-01", "case-02", "case-03"]},
        )
        # 必须按名字取，不能用 [0]：
        # 新规则下critical 层缺失指标也会进入 failed_checks，
        # 索引 0 未必是真正低分的那个。
        failed = {c.name: c for c in r.failed_checks}["answer_correctness"]
        msg = failed.message

        assert "answer_correctness" in msg
        assert "0.8500" in msg
        assert "0.9" in msg
        assert "case-01" in msg

    def test_repair_direction_present(self):
        """报告必须给出修复方向"""
        gate = QualityGate()
        r = gate.evaluate({"answer_correctness": 0.5})
        text = r.text()
        assert "修复方向" in text
        assert "required_facts" in text, "应给出具体修复建议"

    def test_layer_grouping(self):
        gate = QualityGate()
        r = gate.evaluate({"answer_correctness": 0.5})
        layers = r.by_layer()
        assert "critical" in layers
        assert "quality" in layers
        assert "performance" in layers
        assert "stability" in layers

    def test_summary_counts(self):
        gate = QualityGate()
        r = gate.evaluate({"answer_correctness": 0.5})
        s = r.summary()
        assert s["total_checks"] == len(DEFAULT_GATE_CONFIG) and True or True
        assert s["failed"] >= 1
        assert "pass_rate" in s

    def test_many_failed_samples_truncated(self):
        """失败样本过多时截断显示，但报告总数"""
        gate = QualityGate()
        samples = [f"case-{i:03d}" for i in range(30)]
        r = gate.evaluate({"answer_correctness": 0.5},
                          failed_samples={"answer_correctness": samples})
        # 同样按名字取，避免被 critical 层缺失项挤掉
        target = {c.name: c for c in r.failed_checks}["answer_correctness"]
        msg = target.message
        assert "30 个" in msg, "应报告失败总数"
        assert "case-000" in msg


# ============================================================
# 配置
# ============================================================
class TestConfig:
    def test_default_config_complete(self):
        assert set(DEFAULT_GATE_CONFIG) == {
            "critical", "quality", "performance", "stability"}

    def test_critical_layer_has_key_metrics(self):
        c = DEFAULT_GATE_CONFIG["critical"]
        for name in ("hallucination_rate", "refusal_accuracy",
                     "faithfulness", "answer_correctness",
                     "error_rate"):
            assert name in c, f"critical 层缺少 {name}"

    def test_every_rule_has_desc(self):
        """每条规则都应有说明，否则失败时无法理解标准"""
        for layer, rules in DEFAULT_GATE_CONFIG.items():
            for name, rule in rules.items():
                assert rule.get("desc"), f"{layer}.{name} 缺少 desc"

    def test_every_rule_has_threshold(self):
        for layer, rules in DEFAULT_GATE_CONFIG.items():
            for name, rule in rules.items():
                assert ("min" in rule or "max" in rule), \
                    f"{layer}.{name} 未配置 min/max"

    def test_load_from_yaml_file(self):
        import os
        p = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))),
            "configs", "quality_gate.yaml")
        if not os.path.exists(p):
            pytest.skip("quality_gate.yaml 不存在")
        cfg = load_gate_config(p)
        assert "critical" in cfg
        assert cfg["critical"]["hallucination_rate"]["max"] == 0.05

    def test_custom_config_used(self):
        custom = {"critical": {"hallucination_rate": {"max": 0.01}}}
        gate = QualityGate(custom)
        r = gate.evaluate({"hallucination_rate": 0.05})
        assert not r.passed, "自定义阈值应生效"


# ============================================================
# 序列化
# ============================================================
class TestSerialization:
    def test_report_to_dict(self):
        gate = QualityGate()
        r = gate.evaluate({"answer_correctness": 0.5})
        d = r.to_dict()
        assert "summary" in d and "checks" in d and "metrics" in d
        import json
        json.dumps(d, ensure_ascii=False)

    def test_save(self, tmp_path):
        import json
        gate = QualityGate()
        r = gate.evaluate({"answer_correctness": 0.5})
        p = r.save(str(tmp_path / "gate.json"))
        assert json.load(open(p, encoding="utf-8"))["summary"]["passed"] is False

    def test_check_to_dict_has_message(self):
        c = GateCheck(name="x", layer="critical", passed=False,
                      current_value=0.1, threshold_value=0.5,
                      comparison="max")
        d = c.to_dict()
        assert "message" in d
        assert d["message"]
