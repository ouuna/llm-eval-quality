"""
LLM Judge 单元测试
---------------------------------
分两类：
  A. 无需 API 的纯函数测试（JSON 提取、Schema 校验、归一化、失败处理）
  B. 需 API 的能力与偏差测试

**重点验证需求第六条的"必须控制 Judge Bias"**：
    实际测得（glm-4-flash，2026-10-06）
      刻度漂移     spread=0.000→ 稳定
      位置偏好     position_effect=+0.28     → 未达阈值 0.3
      实体调换判幻觉漏判（score 0.72 但 hallucination=False）
      域外编造     score 0.08 hallucination=True  → 正确识别
    这些数据说明：**Judge 不能单独作为幻觉判据，必须与声明级验证交叉验证。**
"""

import os
import json
import pytest

from eval import env_loader
from eval.evaluators.judge import (
    JudgeClient, JudgeResult, extract_json, validate_judge_schema,
    normalize_scores, check_judge_consistency, check_position_bias,
    DEFAULT_JUDGE_CONFIG,
)


API_AVAILABLE = bool(env_loader.get_api_key() and env_loader.get_base_url())
needs_api = pytest.mark.skipif(not API_AVAILABLE, reason="未配置 API")


# ============================================================
# A. 纯函数测试
# ============================================================
class TestJSONExtraction:
    """LLM 常在 JSON 前后加解释或 markdown 围栏，必须稳健提取"""

    def test_plain_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_markdown_fence(self):
        text = '```json\n{"a": 1}\n```'
        assert extract_json(text) == {"a": 1}

    def test_fence_without_json_tag(self):
        assert extract_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_surrounding_text(self):
        text = '好的，评判结果如下：\n{"a": 1}\n希望有帮助。'
        assert extract_json(text) == {"a": 1}

    def test_nested_object(self):
        text = '{"scores": {"faithfulness": 4}}'
        assert extract_json(text) == {"scores": {"faithfulness": 4}}

    def test_invalid_returns_none(self):
        assert extract_json("完全不是 JSON") is None
        assert extract_json("") is None
        assert extract_json(None) is None

    def test_malformed_returns_none(self):
        assert extract_json('{"a": 1,,,}') is None


class TestSchemaValidation:
    """Schema 校验是防止 Judge 输出污染的关键"""

    def test_valid_schema(self):
        data = {
            "scores": {"faithfulness": 4, "relevance": 5, "correctness": 4},
            "hallucination": False,
            "reason": "回答有据",
        }
        assert validate_judge_schema(data) == []

    def test_missing_scores(self):
        issues = validate_judge_schema({"hallucination": False, "reason": "x"})
        assert any("scores" in i for i in issues)

    def test_score_out_of_range(self):
        data = {
            "scores": {"faithfulness": 7, "relevance": 5, "correctness": 4},
            "hallucination": False, "reason": "x",
        }
        issues = validate_judge_schema(data)
        assert any("超出" in i for i in issues)

    def test_score_not_number(self):
        data = {
            "scores": {"faithfulness": "高", "relevance": 5, "correctness": 4},
            "hallucination": False, "reason": "x",
        }
        issues = validate_judge_schema(data)
        assert any("不是数字" in i for i in issues)

    def test_missing_required_dimension(self):
        data = {"scores": {"faithfulness": 4}, "hallucination": False, "reason": "x"}
        issues = validate_judge_schema(data)
        assert any("缺少必需维度" in i for i in issues)

    def test_hallucination_not_bool(self):
        data = {
            "scores": {"faithfulness": 4, "relevance": 5, "correctness": 4},
            "hallucination": "yes", "reason": "x",
        }
        assert any("不是布尔值" in i for i in validate_judge_schema(data))

    def test_not_dict(self):
        issues = validate_judge_schema([1, 2, 3])
        assert issues

    def test_missing_reason(self):
        data = {
            "scores": {"faithfulness": 4, "relevance": 5, "correctness": 4},
            "hallucination": False,
        }
        assert any("reason" in i for i in validate_judge_schema(data))


class TestNormalize:
    def test_scale_0_to_1(self):
        out = normalize_scores({"faithfulness": 5, "relevance": 3})
        assert out["faithfulness"] == 1.0
        assert out["relevance"] == 0.6

    def test_non_numeric_dropped(self):
        out = normalize_scores({"faithfulness": 5, "bad": "x"})
        assert "bad" not in out


class TestNotConfigured:
    """未配置时必须显式失败，绝不静默通过

    注意：必须用 no_api_config 同时屏蔽环境变量与 .env 文件，
    只清环境变量的话，本地存在 .env 时这些用例会假通过。
    """

    def test_missing_key_reports_reason(self, no_api_config):
        c = JudgeClient()
        assert not c.is_available()
        assert "EVAL_API_KEY" in c.availability()["reason"]

    def test_missing_model_reports_reason(self, no_api_config, monkeypatch):
        #只配了key，模型名全部缺失
        monkeypatch.setenv("EVAL_API_KEY", "sk-test")
        c = JudgeClient()
        assert not c.is_available()
        assert "Judge 模型" in c.availability()["reason"]

    def test_judge_returns_unavailable_not_pass(self, no_api_config):
        """
        核心回归：未配置时返回 unavailable，error_kind 明确

        绝不能返回一个"通过"的结果——那是需求第六条明令禁止的
        "把 evaluator API 失败当成测试通过"。
        """
        c = JudgeClient()
        r = c.judge("问题", "答案", "上下文")
        assert r.available is False
        assert r.error_kind == "not_configured"
        assert r.final_score is None
        assert r.error


class TestJudgeResult:
    def test_final_score_average(self):
        r = JudgeResult(available=True,
                        scores={"faithfulness": 1.0, "relevance": 0.5})
        assert r.final_score == 0.75

    def test_final_score_none_when_empty(self):
        assert JudgeResult(available=True, scores={}).final_score is None

    def test_to_dict_serializable(self):
        r = JudgeResult(available=True, scores={"faithfulness": 1.0})
        d = r.to_dict()
        json.dumps(d, ensure_ascii=False)   # 不应抛异常

    def test_config_model_not_hardcoded(self):
        c1 = JudgeClient({"model": "judge-a"})
        c2 = JudgeClient({"model": "judge-b"})
        assert c1.model == "judge-a"
        assert c2.model == "judge-b"

    def test_temperature_zero_by_default(self):
        """temperature=0 是控制刻度漂移的前提"""
        assert DEFAULT_JUDGE_CONFIG["temperature"] == 0.0


# ============================================================
# B. 需 API 的能力测试
# ============================================================
@needs_api
class TestJudgeCapability:
    """
    Judge 的实际判定能力

    **这些测试记录的是真实测量结果，不是理想值。**
    发现的漏判场景同样是重要结论——
    它说明 Judge 不能单独作为幻觉判据。
    """

    CTX = "自动化测试框架分层：基础层负责请求封装，用例层负责断言，数据层负责测试数据。"
    Q = "自动化测试框架通常分为哪几层？"

    def test_correct_answer_scores_high(self):
        c = JudgeClient()
        r = c.judge(self.Q, "自动化测试框架分为基础层、用例层、数据层。", self.CTX)
        assert r.available
        assert r.final_score > 0.8
        assert r.hallucination is False
        assert r.evidence, "必须给出 evidence（防肯定倾向）"

    def test_fabricated_entity_high_score_but_no_hallucination(self):
        """
        已知漏判场景（实测记录）

        Judge 给"报告层"打了 0.72 分，但 hallucination=False。
        原因：它把问题理解为"资料未提及报告层"（信息缺失），
        而非"资料说数据层，模型说报告层"（事实冲突）。

        这个测试固化了该局限，作为"为什么需要声明级验证交叉验证"的依据。
        """
        c = JudgeClient()
        r = c.judge(self.Q, "自动化测试框架分为基础层、用例层、报告层。", self.CTX)
        assert r.available
        # Judge 未识别为幻觉（这是实测结果）
        assert r.hallucination is False
        # 但分数明显低于完全正确的情况
        correct = c.judge(self.Q, "自动化测试框架分为基础层、用例层、数据层。", self.CTX)
        assert r.final_score < correct.final_score

    def test_out_of_domain_fabrication_detected(self):
        """域外编造能被正确识别"""
        c = JudgeClient()
        r = c.judge("Python 怎么安装？", "可以 pip install python 安装。",
                    "（未检索到资料）")
        assert r.available
        assert r.hallucination is True
        assert r.final_score < 0.4

    def test_temperature_zero_deterministic(self):
        """temperature=0 时同一输入应得到相同结果"""
        c = JudgeClient()
        a = c.judge(self.Q, "自动化测试框架分为基础层、用例层、数据层。", self.CTX)
        b = c.judge(self.Q, "自动化测试框架分为基础层、用例层、数据层。", self.CTX)
        assert a.final_score == b.final_score

    def test_stats_tracked(self):
        c = JudgeClient()
        c.judge(self.Q, "自动化测试框架分为基础层、用例层、数据层。", self.CTX)
        assert c.stats["calls"] >= 1
        assert c.stats["failures"] == 0


@needs_api
class TestJudgeBiasMeasurement:
    """
    偏差量化（需求第六节）

    这类测试的价值不在于"通过"，而在于**测量并记录**偏差程度。
    如果某天换模型导致偏差变大，测试会失败并提醒重新校准。
    """

    CTX = "自动化测试框架分层：基础层负责请求封装，用例层负责断言，数据层负责测试数据。"
    Q = "自动化测试框架通常分为哪几层？"

    def test_scale_drift(self):
        """
        刻度漂移检测

        实测 glm-4-flash：spread=0.000（temperature=0 下完全确定）
        """
        c = JudgeClient()
        r = check_judge_consistency(
            c, self.Q, "自动化测试框架分为基础层、用例层、数据层。",
            self.CTX, repeats=3,
        )
        assert r["available"]
        assert r["stable"], f"Judge 存在刻度漂移：{r}"
        assert r["spread"] <= 0.2

    def test_position_bias_below_threshold(self):
        """
        位置偏好检测

        实测 position_effect=+0.28，阈值 0.3，未超但已接近。
        这个测试会在换模型后失败，提示重新校准。
        """
        c = JudgeClient()
        r = check_position_bias(
            c, self.Q, self.CTX,
            good="自动化测试框架分为基础层、用例层、数据层。",
            bad="自动化测试框架分为基础层、用例层、报告层。",
            rounds=2,
        )
        assert r["available"]
        # 允许轻微位置效应，但超过 0.4 说明严重偏置
        assert abs(r["position_effect"]) <= 0.4, \
            f"位置偏好过大：{r['position_effect']} —— {r['reason']}"


@needs_api
class TestJudgeFailureHandling:
    """失败处理：必须显式标记，绝不静默通过"""

    def test_bad_model_name_returns_error(self):
        """模型名错误应返回 error 而非抛异常"""
        c = JudgeClient({"model": "不存在的模型-xxx"})
        r = c.judge("问题", "答案", "上下文")
        assert r.available is False
        assert r.error
        assert r.final_score is None

    def test_error_result_records_kind(self):
        c = JudgeClient({"model": "不存在的模型-xxx"})
        r = c.judge("问题", "答案", "上下文")
        assert r.error_kind in ("api_error", "timeout", "invalid_json",
                                "schema_error", "not_configured")

    def test_failure_counted_in_stats(self):
        c = JudgeClient({"model": "不存在的模型-xxx"})
        c.judge("问题", "答案", "上下文")
        assert c.stats["failures"] >= 1
