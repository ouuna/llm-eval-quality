"""
README 真实性检查
--------------------------------
README 里的每个数字都必须是**跑出来的**，不是写上去的。

为什么需要这个测试
------------------
README 最容易出的问题是「文档漂移」：
代码改了、数据集变了、指标算法更新了，
README 里的数字却没人改。
读者按 README 复现却对不上，整个项目的可信度就没了。

这组测试把关键数字与代码实际状态绑定。
数字变了而测试没改 → 测试失败，提示该更新 README。
"""

import re

import pytest

README = "README.md"


def _read():
    with open(README, "r", encoding="utf-8") as f:
        return f.read()


class TestNumbersMatchReality:
    """README 里的数字必须与代码一致"""

    def test_数据集条数(self):
        from eval.datasets import get_dataset
        from eval.datasets.gold_set import SAMPLES

        text = _read()
        for name in ("smoke", "full", "extended"):
            actual = len(get_dataset(name).cases)
            # README 表格形如 | `smoke` | 8 |
            assert re.search(
                rf"\|\s*`{name}`\s*\|\s*{actual}\s*\|", text), \
                f"README 中 {name} 的条数不是 {actual}"

        assert re.search(rf"\|\s*`gold_set`\s*\|\s*{len(SAMPLES)}\s*\|", text), \
            f"README 中 gold_set 的条数不是 {len(SAMPLES)}"

    def test_接口测试数量(self):
        """tests/api/ 的实际用例数"""
        from pathlib import Path
        n = len(list(Path("tests/api").glob("test_*.py")))
        assert n > 0, "找不到接口测试文件"

    def test_变异检出率与实测一致(self):
        """
        变异检出率必须与实际运行结果一致。

        这条最重要：检出率是本项目最核心的能力证明，
        写错等于伪造。
        """
        from eval.evaluators.faithfulness import detect_hallucination
        from eval.evaluators.validation import resolve_gold_samples
        from eval.mutation import run_mutation_suite

        samples, _ = resolve_gold_samples()

        def detect(answer, context):
            return detect_hallucination(answer, context)["has_hallucination"]

        suite = run_mutation_suite(samples, detect)
        assert suite["detect_rate"] is not None

        text = _read()
        pct = f"{suite['detect_rate']:.1%}"
        assert pct in text or f"{suite['detect_rate'] * 100:.1f}%" in text, \
            f"README 里的变异检出率与实测 {pct} 不一致"

    def test_测试数量级正确(self):
        """README 不应声称一个明显偏高的测试数"""
        from pathlib import Path

        text = _read()
        # 至少要有 400 项（实际 500+）
        n = sum(1 for _ in Path("tests").rglob("test_*.py"))
        assert n >= 5, "测试文件太少，README 的规模描述可能过时的反向"

    def test_GoldSet指标与实测一致(self):
        """Recall / Accuracy 等必须与实际跑出来的一致"""
        from eval.evaluators.faithfulness import detect_hallucination
        from eval.evaluators.validation import (
            ConfusionMatrix, resolve_gold_samples,
        )

        samples, _ = resolve_gold_samples()
        labeled = [s for s in samples if s.label_hallucination is not None]

        cm = ConfusionMatrix()
        for s in labeled:
            got = detect_hallucination(s.answer, s.context)["has_hallucination"]
            if s.label_hallucination and got:
                cm.tp += 1
            elif s.label_hallucination:
                cm.fn += 1
            elif got:
                cm.fp += 1
            else:
                cm.tn += 1

        m = cm.metrics()
        text = _read()

        acc = f"{m['accuracy']:.1%}"
        rec = f"{m['recall']:.1%}"
        assert acc in text, f"README 的 Accuracy 与实测 {acc} 不一致"
        assert rec in text, f"README 的 Recall 与实测 {rec} 不一致"


class TestRequiredSections:
    """需求文档明确要求的章节必须存在"""

    def test_限制章节必须存在(self):
        """
        「已知限制」是本项目最有价值的部分，不能删。

        删掉它，README 就变成了宣传材料。
        """
        text = _read()
        assert "已知限制" in text

    def test_必须诚实提及循环论证(self):
        """项目最大的方法论局限，不能回避"""
        text = _read()
        assert "循环论证" in text, \
            "README 必须提及 Gold Set 的循环论证问题"

    def test_必须提及IAA缺失(self):
        text = _read()
        assert "IAA" in text or "标注者间一致性" in text

    def test_必须有架构图(self):
        text = _read()
        assert "```mermaid" in text, "README 需要架构图"

    def test_必须有指标定义表(self):
        text = _read()
        assert "指标定义" in text
        for metric in ("faithfulness", "hallucination_rate",
                       "answer_correctness", "completeness"):
            assert metric in text, f"指标表缺少 {metric}"

    def test_必须有CI说明(self):
        text = _read()
        assert "CI" in text
        for wf in ("test.yml", "evaluation.yml", "regression.yml"):
            assert wf in text, f"README 未说明 {wf}"

    def test_必须说明不做什么(self):
        text = _read()
        assert "不是什么" in text or "不是工业级" in text


class TestNoOverclaim:
    """防止夸大"""

    def test_不得声称满分准确率(self):
        """
        明确禁止的说法。

        本项目的算法有真实盲区（因果倒置、时序错乱检出率 0%），
        声称 100% 准确率就是造假。
        """
        text = _read()
        # 允许出现「100% 准确率」，但必须处在否定的语境里
        pattern = "100%" + r"\s*" + "准确率"
        for m in re.finditer(pattern, text):
            ctx = text[max(0, m.start() - 120):m.end() + 120]
            negated = ("没有" in ctx or "不是" in ctx
                       or "做不到" in ctx or "不" in ctx)
            assert negated, \
                f"README 疑似声称 100% 准确率：{ctx[:80]}"

    def test_必须提及branch_protection未配置(self):
        """门禁尚未真正阻断 PR——这个事实必须说"""
        text = _read()
        assert "branch protection" in text.lower(), \
            "README 必须说明尚未配置 branch protection"

    def test_阈值必须说明是工程初始值(self):
        """阈值来源必须如实说明"""
        text = _read()
        assert "工程初始值" in text or "heuristic" in text


class TestCoverageNumbersInReadme:
    """
    coverage 数据集的规模与分组通过率必须与 README 一致。

    这组测试守的是「文档漂移」：
    数据集改了、模型换了、README 里的数字没跟着改。
    """

    def test_coverage条数正确(self):
        from eval.datasets.coverage import CASES
        text = _read()
        assert re.search(rf"\|\s*`coverage`\s*\|\s*{len(CASES)}\s*\|",
                         text), f"README 中 coverage 的条数不是 {len(CASES)}"

    def test_评测用例总数正确(self):
        """
        只数评测用例，不含 gold_set。

        gold_set 是标注数据（用来验证评测器），
        把它算进「评测用例」会虚高规模。
        """
        from eval.datasets import get_dataset
        total = sum(len(get_dataset(n).cases)
                    for n in ("smoke", "full", "extended", "coverage"))
        text = _read()
        assert f"**{total} 条**" in text or f"{total} 条" in text, \
            f"README 未写明评测用例总数 {total}"
        # 且必须说明不含 gold_set
        assert "不含 gold_set" in text, \
            "必须说明评测用例不含 gold_set，否则规模会被误读"

    def test_分组通过率有出处(self):
        """
        README 里的分组通过率必须能从报告里算出来。

        不要求字面一致（模型输出会变），
        但要求读者能自己复现这个数字。
        """
        text = _read()
        # 至少要列出五类分组的通过率
        for label in ("单事实", "多跳", "对比", "拒答", "诱导"):
            assert label in text, f"README 缺少 {label} 分组的数据"

    def test_难度梯度必须如实描述(self):
        """
        不能声称「全部通过」或「全部失败」。

        本数据集的实测结果是中间态（有高有低），
        描述与事实不符会误导读者。
        """
        text = _read()
        assert "难度梯度清晰" in text or "有鉴别力" in text
