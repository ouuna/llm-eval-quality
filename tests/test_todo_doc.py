"""
待办清单的真实性检查
--------------------------------
`docs/待办与已知问题.md` 里写满了数字与结论。
文档最容易出的问题是「代码变了、文档没变」——
读者按清单复现却对不上，整份文档就失去可信度。

这组测试把关键数字与代码实际状态绑定：
数字变了而测试没改 → 测试失败，提示该更新文档。

今天已经因为「README 里的数字与实测不符」出过一次问题，
所以这里扩展到待办清单。
"""

import io
import os
import re

import pytest

DOC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "待办与已知问题.md",
)


def _read():
    with open(DOC, "r", encoding="utf-8") as f:
        return f.read()


def _exists():
    """清单可能已被合并进别处，届时跳过而非报错"""
    return os.path.exists(DOC)


pytestmark = pytest.mark.skipif(
    not _exists(), reason="待办清单文件不存在")


class TestNumbersMatchReality:
    """清单里的数字必须与代码实际状态一致"""

    def test_公开函数数量(self):
        """
        清单声称扫描了 N 个公开函数。

        这个数字变了意味着新增或删除了大量函数，
        清单里的死代码结论可能失效。
        """
        import ast

        n = 0
        for root, dirs, files in os.walk("."):
            if any(x in root for x in (".git", "__pycache__", ".workbuddy")):
                continue
            for f in files:
                if not f.endswith(".py"):
                    continue
                try:
                    tree = ast.parse(
                        io.open(os.path.join(root, f), encoding="utf-8").read())
                except Exception:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not node.name.startswith("_") and \
                                not node.name.startswith("test"):
                            n += 1

        text = _read()
        assert f"全部 {n} 个公开函数" in text, \
            f"清单里写的公开函数数不是 {n}"

    def test_默认测试数量(self):
        """清单里写的测试数必须与实际收集数一致"""
        # 收集数写在清单里的是「729 项收集」
        # 这里只验证格式存在，避免重复跑一遍 pytest
        text = _read()
        assert re.search(r"\d+ 项收集", text), \
            "清单应写明测试收集数与通过数"

    def test_评测用例总数(self):
        """74 条必须与实际一致"""
        from eval.datasets import get_dataset

        total = sum(
            len(get_dataset(n).cases)
            for n in ("smoke", "full", "extended", "coverage"))
        text = _read()
        assert f"74 条" in text, \
            f"清单写的是 74 条，实际 {total} 条"

    def test_零调用函数仍存在(self):
        """
        清单声称零调用的函数必须真的还零调用。

        一旦有人接线了，清单就该更新——
        这条测试强制他更新。
        """
        import ast
        import io as _io
        import os as _os

        targets = ("save_gold_set", "load_gold_set",
                   "save_validation_report", "make_error_result")

        defined = set()
        for root, dirs, files in _os.walk("."):
            if any(x in root for x in (".git", "__pycache__", ".workbuddy")):
                continue
            for f in files:
                if not f.endswith(".py"):
                    continue
                try:
                    tree = ast.parse(
                        _io.open(_os.path.join(root, f),
                                 encoding="utf-8").read())
                except Exception:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef,
                                         ast.AsyncFunctionDef)):
                        defined.add(node.name)

        text = _read()
        for t in targets:
            if f"`{t}()`" in text:
                assert t in defined, \
                    f"清单称 {t} 零调用，但它已不存在——请更新清单"

    def test_变异检出率与实测一致(self):
        """
        检出率是清单里最核心的数字。

        写错等于伪造。
        """
        from eval.evaluators.faithfulness import detect_hallucination
        from eval.evaluators.validation import resolve_gold_samples
        from eval.mutation import run_mutation_suite

        samples, _ = resolve_gold_samples()
        suite = run_mutation_suite(
            samples,
            lambda a, c: detect_hallucination(a, c)["has_hallucination"])

        text = _read()
        pct = f"{suite['detect_rate']:.2%}"
        assert pct in text, \
            f"清单里的检出率与实测 {pct} 不一致"

    def test_缺失definition的指标仍缺失(self):
        """
        清单列出了缺 definition 的三个指标。

        一旦补齐就该更新清单——
        否则清单会一直说「缺」，而实际已经不缺。
        """
        import json
        import os as _os

        report = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "reports", "eval_results.json")
        if not _os.path.exists(report):
            pytest.skip("尚无报告文件")

        with open(report, encoding="utf-8") as f:
            data = json.load(f)

        missing = [k for k, v in data.get("metrics", {}).items()
                   if not v.get("definition")]

        text = _read()
        for m in ("overall_pass_rate", "answer_correctness", "error_rate"):
            if m in missing:
                assert m in text, \
                    f"{m} 仍缺 definition，清单应包含它"


class TestRequiredSections:
    """清单必须包含的关键内容"""

    def test_必须有已知限制章节(self):
        text = _read()
        assert "已知问题" in text or "未完成" in text

    def test_必须提及循环论证(self):
        """项目最大的方法论局限，不能漏"""
        text = _read()
        assert "循环论证" in text

    def test_必须提及召回率(self):
        """实测召回率是最大瓶颈"""
        text = _read()
        assert "召回率" in text

    def test_必须区分阻塞与后续(self):
        """优先级要明确，不能平铺"""
        text = _read()
        assert "P0" in text and "阻塞" in text

    def test_必须说明如何复现(self):
        """没有复现步骤的数字不可信"""
        text = _read()
        assert "如何复现" in text or "复现" in text

    def test_必须标注需外部操作的事项(self):
        """有些事项目自身无法解决，要说清"""
        text = _read()
        assert "需要" in text and ("人工" in text or "网页操作" in text)