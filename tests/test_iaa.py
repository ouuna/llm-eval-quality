"""
标注一致性（IAA）测试
---------------------
覆盖任务一的产出：
- cohens_kappa 的边界行为（完全一致 / 完全无关 / 空 / 单类别 / 样本数不等）
- compare_annotations 能正确找出分歧样本
- load_gold_set 的文件不存在时返回 None（不再静默退回模板）
"""

import json
import os

import pytest

from eval.datasets.gold_set import load_gold_set
from eval.evaluators.validation import cohens_kappa, compare_annotations


class TestCohensKappa:

    def test_完全一致_返回1(self):
        labels = [True, False, True, False, True, False]
        r = cohens_kappa(labels, labels)
        assert r["kappa"] == 1.0
        assert r["agreement"] == 1.0
        assert r["reason"] == ""

    def test_完全无关_接近0(self):
        # 两个标注者完全独立：A 全 True，B 全 False（数量相等）
        a = [True] * 5 + [False] * 5
        b = [False] * 5 + [True] * 5
        r = cohens_kappa(a, b)
        assert r["kappa"] is not None
        assert r["kappa"] < 0.5, f"期望低 Kappa，实际 {r['kappa']}"

    def test_空输入_返回None而非异常(self):
        r = cohens_kappa([], [])
        assert r["kappa"] is None
        assert r["n"] == 0
        assert "空输入" in r["reason"]

    def test_单类别输入_返回None并说明原因(self):
        r = cohens_kappa([True, True, True], [True, True, True])
        assert r["kappa"] is None
        assert r["agreement"] == 1.0
        assert "单类别" in r["reason"]

    def test_样本数不等_返回None并说明原因(self):
        r = cohens_kappa([True, False], [True])
        assert r["kappa"] is None
        assert "不一致" in r["reason"] or "配对" in r["reason"]

    def test_混合标签字符串也能算(self):
        # 标签可以是任意可比较值，不止 bool
        r = cohens_kappa(["pos", "neg", "pos"], ["pos", "pos", "neg"])
        assert r["kappa"] is not None
        assert r["n"] == 3


class TestCompareAnnotations:

    def _write(self, path, samples):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        return path

    def _sample(self, sid, label, note="", reviewed=True):
        return {
            "id": sid, "category": "entity_swap",
            "question": f"问题{sid}", "answer": f"答案{sid}",
            "context": "上下文",
            "label_hallucination": label,
            "label_correct": None, "label_complete": None,
            "label_should_refuse": None,
            "required_facts": [], "forbidden_facts": [],
            "note": note, "difficulty": "medium",
            "prefill_source": "human_confirmed",
            "reviewed": reviewed, "reviewed_by": "r", "reviewed_at": "2026",
        }

    def test_能正确找出分歧样本(self, tmp_path):
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        self._write(str(a), [
            self._sample("S1", True, "A认为有幻觉"),
            self._sample("S2", False, "A认为无幻觉"),
            self._sample("S3", True, "A认为有幻觉"),
        ])
        self._write(str(b), [
            self._sample("S1", True, "B认为有幻觉"),
            self._sample("S2", True, "B认为有幻觉"),   # 分歧
            self._sample("S3", False, "B认为无幻觉"),  # 分歧
        ])

        r = compare_annotations(str(a), str(b))
        assert r["status"] == "ok"
        assert r["agreed"] == 1
        assert r["disagreed"] == 2
        ids = {d["id"] for d in r["differences"]}
        assert ids == {"S2", "S3"}
        # 分歧明细包含题目与两个标签
        for d in r["differences"]:
            assert d["question"]
            assert "label_a" in d and "label_b" in d

    def test_文件不存在时返回error(self, tmp_path):
        r = compare_annotations(
            str(tmp_path / "none_a.json"), str(tmp_path / "none_b.json"))
        assert r["status"] == "error"
        assert r["differences"] == []
        assert "读不到" in r["error"] or "解析失败" in r["error"]

    def test_双方一致时kappa为1(self, tmp_path):
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        samples = [
            self._sample("S1", True), self._sample("S2", False),
            self._sample("S3", True), self._sample("S4", False),
        ]
        self._write(str(a), samples)
        self._write(str(b), samples)
        r = compare_annotations(str(a), str(b))
        assert r["kappa"]["kappa"] == 1.0


class TestLoadGoldSetNoSilentFallback:

    def test_文件不存在时返回None(self, tmp_path):
        missing = str(tmp_path / "no_such.json")
        with pytest.warns(UserWarning):
            result = load_gold_set(missing)
        assert result is None

    def test_文件存在时正常加载(self, tmp_path):
        p = tmp_path / "gs.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump([self._sample_dict()], f, ensure_ascii=False)
        result = load_gold_set(str(p))
        assert result is not None
        assert len(result) == 1
        assert result[0].id == "S1"

    def _sample_dict(self):
        return {
            "id": "S1", "category": "c", "question": "q", "answer": "a",
            "context": "", "label_hallucination": True,
            "label_correct": None, "label_complete": None,
            "label_should_refuse": None,
            "required_facts": [], "forbidden_facts": [],
            "note": "", "difficulty": "medium",
            "prefill_source": "auto", "reviewed": False,
            "reviewed_by": "", "reviewed_at": "",
        }
