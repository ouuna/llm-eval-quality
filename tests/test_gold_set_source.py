"""
Gold Set 数据源回归测试
----------------------
这个文件防的是一次真实的、且很典型的假通过。

发生了什么
----------
用户花时间把 30 条样本人工标注完了，写进
`goldset/gold_set.json`，每条都带 `reviewed=true`。

但 `validation.py` 的入口写的是：

    from eval.datasets.gold_set import SAMPLES

`SAMPLES` 是代码里另一份**自动预填的模板**，全部 `reviewed=False`。
按设计这类样本不参与统计（`is_usable` 要求 reviewed=True），
于是验证永远返回 `insufficient_data`。

更麻烦的是 `reports/` 里存着一份看起来很漂亮的准确率报告，
它是手工作业跑出来的，与代码实际行为不符——
**报告与代码不一致，比直接报错危险得多**，
因为它会让评测器看起来已经被验证过了。

所以这里要钉死两件事：
1. 入口必须读人工标注的 JSON
2. 读不到时必须明说用的是模板，不能静默降级
"""

import json
import os

import pytest

from eval.datasets.gold_set import GoldSample
from eval.evaluators.validation import (
    DEFAULT_GOLD_SET_PATH, resolve_gold_samples,
)


class TestResolveGoldSamples:

    def test_读取用户标注的json(self):
        """
        核心回归：入口必须读goldset/gold_set.json，
        不能读代码里的 SAMPLES 模板。
        """
        assert os.path.exists(DEFAULT_GOLD_SET_PATH), \
            "人工标注文件不存在，路径配置有问题"

        samples, source = resolve_gold_samples()

        assert "gold_set.json" in source, \
            f"实际数据源不是标注 JSON：{source}"

        reviewed = [s for s in samples if s.reviewed]
        assert reviewed, \
            "从JSON 读到的样本里没有任何 reviewed=True，" \
            "说明读到的还是模板"

    def test_标注数据真的被用上了(self):
        """接通的直接证据：usable 样本数不为0"""
        samples, _ = resolve_gold_samples()
        usable = [s for s in samples if s.is_usable]
        assert len(usable) > 0, (
            "可用样本为 0，说明is_usable 的条件没被满足——"
            "标注读进来了但没被认可")

    def test_复核者信息被保留(self):
        """
        复核者必须体现在来源说明里。

        这不是装饰：整个项目的可信度都建立在「谁标的」这个问题上，
        如果一份报告显示不出标注者，读者就无法判断它值不值得信。
        """
        samples, source = resolve_gold_samples()
        reviewers = {s.reviewed_by for s in samples if s.reviewed_by}
        assert reviewers, "JSON 里应记录 reviewed_by"
        for r in reviewers:
            assert r in source, \
                f"来源说明里应写明复核者 {r}，实际：{source}"

    def test_文件不存在时明说用模板(self):
        """
        降级必须显式。

        静默退回模板 → 验证返回 insufficient_data →
        用户以为「还没标注」，实际是「标注被忽略了」。
        """
        missing = os.path.join(
            os.path.dirname(DEFAULT_GOLD_SET_PATH), "no_such_file.json")
        samples, source = resolve_gold_samples(missing)

        assert samples, "降级时至少要返回模板内容"
        assert "内置模板" in source, \
            f"降级时必须明说数据来源，实际：{source}"
        assert "未复核" in source, \
            f"必须说明模板不可作为真值，实际：{source}"

    def test_文件损坏时不崩溃(self):
        """坏文件应降级+说明，而不是抛异常让CI 崩掉"""
        import tempfile

        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
                encoding="utf-8") as f:
            f.write("{ 这不是合法 JSON ")
            bad = f.name

        try:
            samples, source = resolve_gold_samples(bad)
            assert samples, "坏文件也应降级返回模板"
            assert "解析失败" in source, \
                f"必须说明解析失败，实际：{source}"
        finally:
            os.unlink(bad)


class TestGoldSampleRoundTrip:

    def test_json往返不丢字段(self):
        """
        to_dict / from_dict 必须无损往返。

        不然用户辛苦标的字段会在某次保存后静默丢失——
        标注丢失比标注错误更难发现。
        """
        original = GoldSample(
            id="T1", category="entity_swap",
            question="q", answer="a", context="c",
            label_hallucination=True, label_correct=False,
            label_complete=True, label_should_refuse=None,
            required_facts=["x"], forbidden_facts=["y"],
            note="n", difficulty="hard",
            prefill_source="human_confirmed",
            reviewed=True, reviewed_by="someone", reviewed_at="2026-10-06",
        )
        restored = GoldSample.from_dict(original.to_dict())
        assert restored == original, "往返后字段不一致"

    def test_从真实json读取的字段完整(self):
        with open(DEFAULT_GOLD_SET_PATH, encoding="utf-8") as f:
            raw = json.load(f)

        for item in raw:
            s = GoldSample.from_dict(item)
            assert s.id, "样本必须有 id"
            assert isinstance(s.reviewed, bool), \
                f"{s.id} 的 reviewed 应为布尔值"
            if s.reviewed:
                assert s.label_hallucination is not None, \
                    f"{s.id} 标记为已复核，却没给幻觉标签"