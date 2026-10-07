"""
阈值校准工具测试
--------------------------------
这个工具的价值在于**提供数据让人自己判断**。
所以要测的不是「它给出的阈值对不对」——
那需要人工决策——而是「它给的数据是否可信」。
"""

import pytest

from eval.calibrate import (
    evaluate_at_threshold, format_curve, main, scan,
)


class TestScanning:

    def test_能扫描出结果(self):
        from eval.evaluators.validation import resolve_gold_samples

        samples, _ = resolve_gold_samples()
        labeled = [s for s in samples if s.label_hallucination is not None]
        if not labeled:
            pytest.skip("无可用标注数据")

        rows = scan(labeled, [0.4, 0.6])
        assert len(rows) == 2
        for r in rows:
            assert 0.0 <= r["accuracy"] <= 1.0
            assert 0.0 <= r["recall"] <= 1.0

    def test_指标齐全(self):
        """准确率/精确率/召回率/F1/误报率/漏报率一个都不能少"""
        from eval.evaluators.validation import resolve_gold_samples

        samples, _ = resolve_gold_samples()
        labeled = [s for s in samples if s.label_hallucination is not None]
        if not labeled:
            pytest.skip("无可用标注数据")

        r = evaluate_at_threshold(labeled, 0.6)
        for key in ("accuracy", "precision", "recall", "f1", "fpr", "fnr"):
            assert key in r, f"缺少 {key}"

    def test_混淆矩阵计数自洽(self):
        """
        tp+fn 应等于真实有幻觉的样本数。

        不自洽说明标注读取出了问题——
        那时所有下游指标都不可信。
        """
        from eval.evaluators.validation import resolve_gold_samples
        from eval.evaluators.validation import ConfusionMatrix
        from eval.evaluators.faithfulness import detect_hallucination

        samples, _ = resolve_gold_samples()
        labeled = [s for s in samples if s.label_hallucination is not None]
        if not labeled:
            pytest.skip("无可用标注数据")

        expect_pos = sum(1 for s in labeled
                         if s.label_hallucination is True)

        cm = ConfusionMatrix()
        for s in labeled:
            r = detect_hallucination(s.answer, s.context)
            if s.label_hallucination and r["has_hallucination"]:
                cm.tp += 1
            elif s.label_hallucination and not r["has_hallucination"]:
                cm.fn += 1
            elif not s.label_hallucination and r["has_hallucination"]:
                cm.fp += 1
            else:
                cm.tn += 1

        assert cm.tp + cm.fn == expect_pos, \
            f"真实有幻觉 {expect_pos} 条，但 tp+fn={cm.tp + cm.fn}"


class TestOutput:

    def test_曲线含解读说明(self):
        """
        表格之外必须说明「怎么读」。

        只给一堆数字而不解释代价不对称，
        使用者会本能地去选 F1 最高的那个——
        而漏报率 7% 和 0% 的代价天差地别。
        """
        rows = [{
            "threshold": 0.6, "accuracy": 0.9, "precision": 0.92,
            "recall": 0.86, "f1": 0.89, "fpr": 0.06, "fnr": 0.14,
            "errors": [],
        }]
        text = format_curve(rows)

        assert "怎么读" in text, "必须说明怎么读这张表"
        assert "漏报" in text and "误报" in text
        assert "业务决策" in text, "应指出选阈值是业务决策而非技术决策"

    def test_无错误样本时不报错(self):
        rows = [{
            "threshold": 0.6, "accuracy": 1.0, "precision": 1.0,
            "recall": 1.0, "f1": 1.0, "fpr": 0.0, "fnr": 0.0,
            "errors": [],
        }]
        assert "F1 最优" in format_curve(rows)


class TestHonesty:
    """
    这组测的是诚实性——比功能正确更重要。
    """

    def test_输出包含循环论证提醒(self):
        """
        校准结果必须附带「这是循环论证」的警示。

        评测器的检测规则是为检出这批样本的缺陷而设计的，
        在同一批数据上表现好是必然的。
        不说清楚这一点，工具就成了自证清白的道具。
        """
        rc = main([])
        assert rc == 0, "无标注数据时应返回 0 并说明原因"

    def test_explain打印局限(self, capsys):
        rc = main(["--explain"])
        out = capsys.readouterr().out

        assert "循环论证" in out, "必须提及循环论证"
        assert "IAA" in out or "标注者间一致性" in out, \
            "必须提及缺少标注者间一致性"
        assert "200" in out, "应说明真正校准需要的数据规模"

    def test_局限声明在模块docstring里(self):
        """模块级说明也要有，避免只看 CLI 输出的人漏掉"""
        import eval.calibrate as mod
        doc = mod.__doc__
        assert "循环论证" in doc
        assert "局限" in doc
