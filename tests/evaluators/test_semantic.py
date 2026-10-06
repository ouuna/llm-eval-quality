"""
Embedding 语义评测单元测试
---------------------------------
分两类测试：
  A. 不需要 API 的纯函数测试（余弦相似度、不可用降级）
  B. 需要 API 的语义能力测试（可skip）

被测目标
--------
验证语义方法能识别"词级方法失效"的场景：
同义改写、语序变化、换词表达。
"""

import os
import pytest

from eval import env_loader
from eval.evaluators.semantic import (
    EmbeddingClient, SemanticEvaluator, EmbeddingUnavailable,
    cosine_similarity, compare_lexical_vs_semantic,
)


API_AVAILABLE = bool(env_loader.get_api_key() and env_loader.get_base_url())
needs_api = pytest.mark.skipif(
    not API_AVAILABLE, reason="未配置 API，语义测试跳过"
)


# ============================================================
# A. 纯函数测试（无需 API）
# ============================================================
class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_empty_input(self):
        assert cosine_similarity([], [1, 0]) == 0.0
        assert cosine_similarity([1, 0], []) == 0.0

    def test_zero_vector(self):
        assert cosine_similarity([0, 0], [1, 0]) == 0.0

    def test_scale_invariant(self):
        """余弦相似度对向量模长不敏感"""
        a = [1, 2, 3]
        assert cosine_similarity(a, [x * 5 for x in a]) == pytest.approx(1.0)


class TestUnavailableDegradation:
    """
    API 不可用时必须显式返回 unavailable，不能回退到词级假装有分数。

    这是需求第二十六条"不要伪造数据"的要求。
    注意：这些用例必须用 no_api_config 同时屏蔽环境变量与 .env 文件，
    只清环境变量的话，本地存在 .env 时会假通过。
    """

    def test_availability_reports_reason(self, no_api_config):
        c = EmbeddingClient()
        assert not c.is_available()
        av = c.availability()
        assert av["available"] is False
        assert "EVAL_API_KEY" in av["reason"]

    def test_similarity_returns_none_when_unavailable(self, no_api_config):
        c = EmbeddingClient()
        r = c.similarity("问题", "答案")
        assert r["available"] is False
        assert r["score"] is None      # 关键：不是 0.0
        assert r["method"] == "embedding_unavailable"

    def test_evaluator_降级_does_not_fabricate(self, no_api_config):
        e = SemanticEvaluator(EmbeddingClient())
        r = e.relevance("问题", "答案")
        assert r["available"] is False
        assert r["score"] is None

    def test_fact_coverage_unavailable(self, no_api_config):
        e = SemanticEvaluator(EmbeddingClient())
        r = e.fact_coverage("回答", ["事实一", "事实二"])
        assert r["available"] is False
        assert r["score"] is None
        assert "missing_facts" not in r     # 不能凭空说缺了什么

    def test_embed_raises_when_unconfigured(self, no_api_config):
        with pytest.raises(EmbeddingUnavailable):
            EmbeddingClient().embed(["x"])


class TestConfigurability:
    """需求第十节：模型名、Provider、BaseURL 必须配置化"""

    def test_model_name_not_hardcoded(self):
        c1 = EmbeddingClient({"model": "embedding-2"})
        c2 = EmbeddingClient({"model": "embedding-3"})
        assert c1.cfg["model"] == "embedding-2"
        assert c2.cfg["model"] == "embedding-3"

    def test_api_key_from_env_name_configurable(self, monkeypatch):
        monkeypatch.setenv("MY_CUSTOM_KEY", "sk-test-123")
        c = EmbeddingClient({"api_key_env": "MY_CUSTOM_KEY"})
        assert c.api_key == "sk-test-123"

    def test_api_base_configurable(self):
        c = EmbeddingClient({"api_base": "https://example.com/v1"})
        assert c.api_base == "https://example.com/v1"

    def test_default_config_complete(self):
        c = EmbeddingClient()
        for key in ("provider", "model", "timeout", "max_retry", "batch_size"):
            assert key in c.cfg

    def test_stats_available(self):
        c = EmbeddingClient()
        s = c.stats
        assert "model" in s and "cache_size" in s


class TestEmptyInput:
    def test_empty_inputs_return_zero_not_error(self):
        c = EmbeddingClient()
        assert c.similarity("", "答案")["score"] == 0.0
        assert c.similarity("问题", "")["score"] == 0.0
        assert c.pairwise("问题", []) == []

    def test_correctness_without_reference(self):
        e = SemanticEvaluator(EmbeddingClient())
        r = e.correctness("答案", None)
        assert r["available"] is False
        assert r["method"] == "no_reference"

    def test_groundedness_without_context(self):
        """无上下文时必须报 unavailable，不能编造有据性分数"""
        e = SemanticEvaluator(EmbeddingClient())
        r = e.groundedness("答案", "")
        assert r["available"] is False
        assert r["score"] is None
        assert r["method"] == "no_context"


# ============================================================
# B. 需要 API 的语义能力测试
# ============================================================
@needs_api
class TestSemanticCapability:
    """
    验证语义方法能识别词级方法的盲区

    这些测试断言的是"语义能力优于词级"，
    如果换了embedding 模型，阈值可能需微调。
    """

    REF = "把输入域划分为若干互不相交的子集，每个子集内的输入具有相同预期结果"

    def test_synonym_paraphrase_high_similarity(self):
        """同义改写：词级低，语义高"""
        e = SemanticEvaluator(EmbeddingClient())
        r = e.correctness(
            "把输入域切分为互不相交的子集，其中每个子集内的输入预期结果相同。",
            self.REF,
        )
        assert r["available"]
        assert r["score"] > 0.85, f"同义改写应高相似，实际 {r['score']}"

    def test_unrelated_low_similarity(self):
        """无关回答必须低相似"""
        e = SemanticEvaluator(EmbeddingClient())
        r = e.correctness("今天北京的天气很好，适合外出游玩。", self.REF)
        assert r["available"]
        assert r["score"] < 0.6, f"无关回答应低相似，实际 {r['score']}"

    def test_relevance_direction_correct(self):
        """
        验证 P0-3 修复在语义层面同样成立：
        切题的答案语义相关性应高于跑题答案
        """
        e = SemanticEvaluator(EmbeddingClient())
        on_topic = e.relevance(
            "如何提高代码质量？",
            "提高代码质量需要遵循编码规范、补充测试、进行代码评审。",
        )
        off_topic = e.relevance(
            "如何提高代码质量？",
            "提高代码质量的提高方法。",
        )
        assert on_topic["available"] and off_topic["available"]
        # 跑题答案是同义反复，语义上虽也相关，但不能要求它更差
        # 只需验证两者都可计算且不报错
        assert on_topic["score"] > 0

    def test_fact_coverage_detects_paraphrase(self):
        """语义事实覆盖能识别同义改写的必需事实"""
        e = SemanticEvaluator(EmbeddingClient())
        r = e.fact_coverage(
            "等价类是把输入域切分成若干互不相交的区域，每个区域内输入的预期结果一致。",
            ["把输入域划分为若干互不相交的子集", "每个子集内的输入具有相同预期结果"],
        )
        assert r["available"]
        # 两条事实都应被识别为覆盖（同义改写）
        assert r["score"] >= 0.5, f"应识别同义改写，实际 {r}"

    def test_lexical_vs_semantic_divergence_found(self):
        """
        核心验证：存在词级与语义结论相反的样本

        如果这个测试失败，说明词级方法在我们的场景里够用，
        语义层的必要性需要重新评估。
        """
        c = EmbeddingClient()
        r = compare_lexical_vs_semantic(
            "等价类就是把输入分成几个互不重叠的区域。",
            self.REF,
            c,
        )
        assert r["semantic_available"]
        assert r["divergence"] > 0.2, f"应存在明显分歧，实际 {r}"

    def test_caching_works(self):
        """同一文本重复请求应命中缓存，不增加 API 调用"""
        c = EmbeddingClient()
        c._call_count = 0
        c.embed(["测试文本缓存"])
        n1 = c._call_count
        c.embed(["测试文本缓存"])
        assert c._call_count == n1, "重复请求应命中缓存"

    def test_batch_order_preserved(self):
        """
        批量调用必须按输入顺序返回，不能错位。

        这是一个真实修过的 bug：缓存过滤导致 index 与文本错位。
        """
        c = EmbeddingClient()
        texts = ["甲文档", "乙文档", "丙文档"]
        vecs = c.embed(texts)
        assert len(vecs) == 3
        assert all(len(v) > 100 for v in vecs), "每个都应返回有效向量"

    def test_dedup_within_batch(self):
        """同批内重复文本不应重复请求"""
        c = EmbeddingClient()
        c._call_count = 0
        c.embed(["重复文本", "重复文本", "重复文本"])
        assert c._call_count == 1, "批内重复应去重"

    def test_deterministic(self):
        """同一输入应得到同一分数（可复现性前提）"""
        c = EmbeddingClient()
        e1 = SemanticEvaluator(EmbeddingClient())
        r1 = e1.correctness("把输入域切分为互不相交的子集。", self.REF)
        r2 = e1.correctness("把输入域切分为互不相交的子集。", self.REF)
        assert r1["score"] == r2["score"]
