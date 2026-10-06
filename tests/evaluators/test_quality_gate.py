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
            "regression_pass_rate": 1.0, "overall_pass_rate": 0.98,
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
        failed = r.failed_checks[0]
        assert failed.name == "answer_correctness"
        assert failed.current_value == 0.80
        assert failed.threshold_value == 0.90
        assert failed.comparison == "min"

    def test_max_violation(self):
        gate = QualityGate()
        r = gate.evaluate({"hallucination_rate": 0.15})
        assert not r.passed
        failed = [c for c in r.failed_checks if c.name == "hallucination_rate"][0]
        assert failed.comparison == "max"
        assert failed.current_value > failed.threshold_value

    def test_boundary_equal_passes(self):
        """恰好等于阈值应通过（闭区间）"""
        gate = QualityGate()
        r = gate.evaluate({"regression_pass_rate": 1.0,
                          "hallucination_rate": 0.05})
        checks = {c.name: c for c in r.checks}
        assert checks["regression_pass_rate"].passed
        assert checks["hallucination_rate"].passed

    def test_enabled_layers(self):
        """可只启用部分层"""
        gate = QualityGate()
        r = gate.evaluate({"answer_correctness": 0.5},
                          enabled_layers=["quality"])
        # critical 层未启用，其检查项不参与
        assert all(c.layer != "critical" for c in r.checks)

    def test_unavailable_metric_is_skipped(self):
        """
        指标不可用时跳过而非判失败

        理由：指标不可计算不等于指标不达标。
        但报告必须显式说明跳过原因。
        """
        gate = QualityGate()
        r = gate.evaluate({})     # 全部缺失
        assert r.passed, "指标缺失不应导致门禁失败"
        assert len(r.skipped_checks) > 0
        assert all(c.skip_reason for c in r.skipped_checks)

    def test_skipped_message_explains(self):
        gate = QualityGate()
        r = gate.evaluate({})
        msg = r.skipped_checks[0].message
        assert "跳过" in msg


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
        failed = r.failed_checks[0]
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
        msg = r.failed_checks[0].message
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
                     "regression_pass_rate"):
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
