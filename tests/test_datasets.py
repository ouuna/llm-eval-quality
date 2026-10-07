"""
数据集构造质量测试
--------------------------------
数据集是评测系统的**输入**。输入有问题，输出的所有数字都不可信。

这类测试的作用不是「保证每条用例都对」——
那需要人工评审——而是保证**结构性约束**成立：

  · 拒答用例不能有标准答案
  · 必需事实不能与禁止事实矛盾
  · 不能有重复的用例 id
  · 不能全是同一类问题

最后一条尤其重要：
一个全是同义改写的 100 条数据集，
通过率会很好看，但对评测能力的验证几乎没有贡献。
"""

import pytest

from eval.datasets import get_dataset
from eval.datasets.extended import CASES, CONSTRUCTION_NOTE, DATASET
from eval.schemas.dataset import validate_dataset


class TestDatasetIntegrity:

    def test_校验器零问题(self):
        """
        全部数据集都应通过 validate_dataset。

        这道闸门在 runner 里也会跑，
        但放在测试里能第一时间发现问题，
        而不是等评测跑完（可能已经花了 API 费用）。
        """
        for name in ("smoke", "full", "extended", "coverage"):
            ds = get_dataset(name)
            issues = validate_dataset(ds)
            assert not issues, \
                f"{name} 数据集有 {len(issues)} 个问题：{issues[:3]}"

    def test_id唯一(self):
        """重复 id 会让结果无法定位到具体用例"""
        for name in ("smoke", "full", "extended", "coverage"):
            ids = [c.id for c in get_dataset(name).cases]
            assert len(ids) == len(set(ids)), f"{name} 存在重复 id"

    def test_combo不会重复(self):
        """
        组合数据集不应引入重复用例。

        smoke+full+extended 三者 id 前缀不同，应该干净。
        """
        ds = get_dataset("smoke+full+extended")
        ids = [c.id for c in ds.cases]
        assert len(ids) == len(set(ids)), "组合数据集出现重复用例"


class TestExtendedCoverage:
    """
    扩展集的价值在于「覆盖此前缺失的类型」，
    所以要验证它确实做到了。
    """

    def test_规模合理(self):
        """
        不是越多越好，也不是越少越好。

        下限：每类至少 2 条，否则某一类只有 1 条时统计没有意义
        """
        assert len(CASES) >= 15, f"扩展集只有 {len(CASES)} 条，覆盖不足"

    def test_覆盖五类场景(self):
        """总结/条件/数据异常/边界/对抗，一类都不能少"""
        from eval.datasets.extended import (
            ADVERSARIAL, BOUNDARY, CONDITIONAL, CONTEXT_ANOMALY, SUMMARY,
        )

        for name, group in [("总结类", SUMMARY), ("条件类", CONDITIONAL),
                            ("数据异常", CONTEXT_ANOMALY),
                            ("边界输入", BOUNDARY),
                            ("对抗诱导", ADVERSARIAL)]:
            assert len(group) >= 2, \
                f"{name} 只有 {len(group)} 条，单条无法反映该类问题"

    def test_覆盖多种expected_behavior(self):
        """
        必须包含 answer / refuse 等不同期望。

        全是 answer 的数据集测不出拒答能力——
        而拒答能力恰恰是 RAG 系统最容易出问题的地方。
        """
        behaviors = {c.expected_behavior for c in CASES}
        assert "answer" in behaviors
        assert "refuse" in behaviors, "缺少拒答类用例"

    def test_包含拒答与作答混合(self):
        """
        拒答占比应在合理区间。

        太高说明数据集偏消极（什么都该拒答），
        太低说明测不出过度拒答的问题。
        """
        refuse = sum(1 for c in CASES if c.expected_behavior == "refuse")
        ratio = refuse / len(CASES)
        assert 0.15 <= ratio <= 0.6, \
            f"拒答类占比 {ratio:.0%} 不合理（应 15%~60%）"

    def test_每条都有明确测试意图(self):
        """
        每条用例都应写清「它在测什么」。

        没有 note 的用例，三个月后没人知道它为什么存在，
        最后会变成没人敢删的历史包袱。
        """
        missing = [c.id for c in CASES if not c.note]
        assert not missing, f"以下用例缺少 note：{missing}"

    def test_每条都有标签(self):
        """标签用于按维度筛选分析"""
        missing = [c.id for c in CASES if not c.tags]
        assert not missing, f"以下用例缺少 tags：{missing}"

    def test_不是同义改写堆砌(self):
        """
        问题文本应有足够多样性。

        如果两条用例的问句完全相同，
        它们测的其实是同一件事，不该各占一条。
        """
        questions = [c.question for c in CASES]
        assert len(questions) == len(set(questions)), \
            "存在完全重复的问题文本"

        # 统计不同首尾字符的分布，粗略判断多样性
        prefixes = {q[:2] for q in questions}
        assert len(prefixes) >= 5, \
            f"问题开头种类仅 {len(prefixes)} 种，多样性不足"


class TestHonesty:
    """
    这组测的是**诚实性**。

    需求文档明确要求「不要伪造测试数据」。
    能被机器检查的诚实性是：
    至少不要在文件里声称「已人工标注」而实际没有。
    """

    def test_构造说明必须存在且指明未复核(self):
        """
        扩展集必须明确声明「未经第二位标注者复核」。

        这条断言看起来奇怪，但它守的是底线：
        如果将来有人把它当成可信基线来引用，
        至少文件里有明确的警示。
        """
        assert "未经第二位标注者复核" in CONSTRUCTION_NOTE, \
            "构造说明必须明确指出未复核"
        assert "Gold Set" in CONSTRUCTION_NOTE, \
            "应指向真正经过人工复核的数据源"

    def test_不以数据集规模冒充能力(self):
        """
        说明里不能宣称「覆盖 XX 类场景」这种能力性结论，
        只能说覆盖了哪些用例类型。
        """
        assert "能力基线" in CONSTRUCTION_NOTE, \
            "应说明它不是能力基线"


class TestGroundTruthQuality:
    """
    Ground Truth 质量直接决定评测准确度。
    """

    def test_作答用例应有required_facts(self):
        """
        answer 类用例必须给 required_facts。

        只有 reference_answer 的话，
        程序无法判定「答得对不对」——
        字符级匹配 reference_answer 会把同义改写判为错。
        """
        for c in CASES:
            if c.expected_behavior == "answer":
                assert c.ground_truth.required_facts, \
                    f"[{c.id}] answer 类用例缺少 required_facts"

    def test_拒答用例不应有reference_answer(self):
        """
        拒答用例的正确答案是「拒答」本身。

        给了 reference_answer 会造成矛盾：
        评测时到底是判「答案与参考一致」还是判「拒答了」？
        """
        for c in CASES:
            if c.expected_behavior == "refuse":
                assert not c.ground_truth.reference_answer, \
                    f"[{c.id}] refuse 类用例不应有 reference_answer"

    def test_禁止事实不得与必需事实矛盾(self):
        for c in CASES:
            req = set(c.ground_truth.required_facts)
            forb = set(c.ground_truth.forbidden_facts)
            overlap = req & forb
            assert not overlap, \
                f"[{c.id}] required 与 forbidden 矛盾：{overlap}"

    def test_注入类用例应有防护意图(self):
        """
        prompt_injection 类用例必须写明它在测什么防护。

        注入测试的价值在于「验证防护有效」，
        而不是「有一个问题带这些关键词」。
        """
        for c in CASES:
            if c.category == "prompt_injection":
                assert any(t in c.tags for t in
                           ("诱导编造", "诱导忽略知识库",
                            "角色扮演诱导", "顺势追问", "指令覆盖")), \
                    f"[{c.id}] 注入类用例的 tags 未说明防护类型：{c.tags}"

class TestRefusalNotIrrelevant:
    """
    回归测试：正确拒答不得被记为「跑题」

    真实发现
    --------
    扩展数据集跑真实API 时，5 条拒答用例全部被判失败，
    原因都是 IRRELEVANT——理由是
    「缺少 required_facts / reference / context，无法判定相关性」。

    根因是设计缺陷：
      · 拒答用例本来就不该有 required_facts
        （正确答案就是「拒答」本身，validate_dataset 会明确报错）
      · 但相关性评测的最后一级会判「无依据 → 不相关」
      · 于是**系统正确拒答，反被算成跑题**

    把好结果算成坏结果，比直接报错更糟：
    它会让人以为系统有问题，去修不该修的地方。
    """

    REFUSALS = [
        "参考资料中未提及。",
        "参考资料中未提及该问题相关内容，无法回答。",
        "无法回答",
        "没有相关内容",
        "不知道",
    ]

    @pytest.mark.parametrize("answer", REFUSALS)
    def test_拒答不应被判跑题(self, answer):
        from eval.evaluators.correctness import evaluate_relevance

        # 完全不给依据，复现拒答用例的真实处境
        r = evaluate_relevance(answer, required_facts=None,
                              reference_answer=None, context="")

        assert r["is_relevant"], (
            f"正确拒答被判为不相关：{answer!r} → {r['reason']}")
        assert r["method"] == "refusal_acknowledged"

    @pytest.mark.parametrize("answer", REFUSALS)
    def test_拒答在有依据时也不该判跑题(self, answer):
        """即使给了 context，拒答仍应被识别"""
        from eval.evaluators.correctness import evaluate_relevance

        r = evaluate_relevance(
            answer, required_facts=["等价类划分"],
            reference_answer=None,
            context="等价类划分：把输入域划分为子集。")

        assert r["is_relevant"]

    def test_拒答词表全项目统一(self):
        """
        faithfulness 与 correctness 必须用同一份词表。

        早先各写一份且内容不同，导致同一个回答
        在幻觉检测里被认作拒答、在相关性检测里却不被认作。
        """
        from eval.evaluators.correctness import (
            REFUSAL_MARKERS, _is_refusal,
        )
        from eval.evaluators.faithfulness import detect_hallucination

        for marker in REFUSAL_MARKERS:
            assert _is_refusal(f"回答里包含{marker}这个词"), \
                f"_is_refusal 未识别 {marker}"

            # 幻觉检测也应认得同一个词
            r = detect_hallucination(
                f"回答里包含{marker}", "任意上下文", should_refuse=True)
            assert r["has_hallucination"] is False, \
                f"faithfulness 未识别 {marker} 为拒答"

    def test_非拒答不应被误判(self):
        """防误报：正常内容不能被当成拒答"""
        from eval.evaluators.correctness import _is_refusal

        for text in [
            "等价类划分是把输入域划分为子集的方法。",
            "冒烟测试验证核心功能是否可用。",
            "",
        ]:
            assert not _is_refusal(text), f"{text!r} 被误判为拒答"


class TestDescriptionMatchesReality:
    """
    描述里的条数必须与实际一致。

    真实踩过：full 数据集的 description 写「22 条」，
    实际只有 19 条。读报告的人会以为漏了 3 条用例，
    或者以为有条被删了没改描述。
    """

    def test_描述中的条数与实际一致(self):
        import re

        for name in ("smoke", "full", "extended", "coverage"):
            ds = get_dataset(name)
            m = re.search(r"(\d+)\s*条", ds.description)
            if m:
                assert int(m.group(1)) == len(ds.cases), (
                    f"{name} 描述说「{m.group(1)} 条」"
                    f"但实际 {len(ds.cases)} 条")

    def test_描述中的类别数与实际一致(self):
        import re

        for name in ("smoke", "full", "extended", "coverage"):
            ds = get_dataset(name)
            m = re.search(r"(\d+)\s*个类别", ds.description)
            if m:
                actual = len({c.category for c in ds.cases})
                assert int(m.group(1)) == actual, (
                    f"{name} 描述说「{m.group(1)} 个类别」"
                    f"但实际 {actual} 个")


class TestCoverageDataset:
    """
    场景覆盖集的质量约束。

    这组测试的价值在于**防止数据集退化**：
    评测数据集最可能的劣化方式是「悄悄变成同义改写堆砌」——
    条数达标了，但测不出新东西。

    所以这里不看条数，看分布。
    """

    def test_五类意图都有足够样本(self):
        """每类至少 3 条，否则某一类只有 1~2 条时统计没意义"""
        from eval.datasets.coverage import GROUP_STATS

        assert len(GROUP_STATS) == 5, f"应覆盖 5 类意图，实际 {len(GROUP_STATS)}"
        for name, n in GROUP_STATS.items():
            assert n >= 3, f"{name} 只有 {n} 条，样本太少"

    def test_难度不应偏轻(self):
        """
        至少三成是 hard。

        全是 easy 的数据集测不出能力边界——
        通过率会很好看，但对评测体系没有任何鉴别力。
        """
        from eval.datasets.coverage import CASES

        hard = sum(1 for c in CASES if c.difficulty == "hard")
        ratio = hard / len(CASES)
        assert ratio >= 0.3, \
            f"hard 占比 {ratio:.0%} 过低，数据集偏简单"

    def test_多跳与对比类必须存在(self):
        """
        这两类是最能拉开差距的——
        单跳做对不代表多跳能做对。

        缺了它们，数据集只能验证「基本问答」，
        验证不了推理能力。
        """
        from eval.datasets.coverage import CASES

        hard_types = [c for c in CASES
                      if c.category == "multi_hop"
                      or "对比" in " ".join(c.tags)]
        assert len(hard_types) >= 8, \
            f"多跳+对比类只有 {len(hard_types)} 条，不足以验证推理能力"

    def test_问题文本不重复(self):
        """同义改写不构成新用例"""
        from eval.datasets.coverage import CASES

        questions = [c.question for c in CASES]
        assert len(questions) == len(set(questions)), "存在重复问题"

    def test_构造说明存在且诚实(self):
        """必须声明未复核，且说明与 extended 的分工"""
        from eval.datasets.coverage import CONSTRUCTION_NOTE

        assert "未经第二位标注者复核" in CONSTRUCTION_NOTE
        assert "extended" in CONSTRUCTION_NOTE, "应说明与 extended 的分工"

    def test_拒答类不给参考答案(self):
        from eval.datasets.coverage import CASES

        for c in CASES:
            if c.expected_behavior == "refuse":
                assert not c.ground_truth.reference_answer, \
                    f"[{c.id}] 拒答类不应有 reference_answer"


class TestDatasetScale:
    """规模约束"""

    def test_评测用例总数在需求区间(self):
        """
        需求文档要求 50~100 条。

        注意只数**评测用例**，不含 gold_set——
        后者是用来验证评测器的标注数据，不是被测系统的用例。
        把两者混起来算会虚高规模。
        """
        from eval.datasets import get_dataset

        total = sum(len(get_dataset(n).cases)
                    for n in ("smoke", "full", "extended", "coverage"))
        assert 50 <= total <= 100, \
            f"评测用例 {total} 条，不在需求的 50~100 区间"
