"""
变异测试
--------------------------------
回答一个别人不会问、但必须问的问题：

    **「如果评测器是坏的，它会告诉我吗？」**

普通测试只能证明「正确输入得到正确输出」。
变异测试反过来：**故意把输入改坏，看评测器能不能发现**。

类比：单元测试测「乘法算得对不对」，
变异测试测「如果我把乘法写错了，测试会发现吗」。

为什么对 LLM 评测特别重要
------------------------
LLM 评测器本身就是一段代码，它也会写错。
而且它的错误有个恶劣特性：
**评测器坏了 → 分数虚高 → 缺陷漏到生产环境**。
这种错误不会以任何形式暴露，
除非有人专门去破坏输入试试。

怎么读检出率
------------
检出率 = 被发现的坏输入 / 应被发现的总坏输入

  · 高（>90%）：评测器比较可靠
  · 中（70~90%）：有已知盲区，需记录
  · 低（<70%）：**不要相信这个评测器的结论**

**低检出率不等于评测器没用**，它只是说明
「它能发现这类问题」而不是「它能发现所有问题」。
所以本项目的目标不是刷高数字，而是**如实暴露盲区**。
"""

import json

import pytest

from eval.datasets.gold_set import GoldSample
from eval.evaluators.faithfulness import detect_hallucination
from eval.mutation import ALL_OPERATORS, generate_mutations, run_mutation_test

pytestmark = pytest.mark.offline


# ============================================================
# 判定函数：把真实评测器包装成变异测试能用的形式
# ============================================================
def make_detector(forbidden_facts=None, required_facts=None):
    """
    构造判定函数：给定（答案, 上下文）返回是否检出幻觉。

    刻意直接调用 detect_hallucination 而不是走整套 runner：
    变异测试要测的是**幻觉检测这一环**，
    混进其它评测器会让失败原因难定位——
    到底是没检出幻觉，还是完整性判定太宽松？
    """
    def detect(answer: str, context: str) -> bool:
        r = detect_hallucination(
            answer, context,
            forbidden_facts=forbidden_facts,
            required_facts=required_facts,
        )
        return r["has_hallucination"]

    return detect


# ============================================================
# 一、变异算子自身
# ============================================================
class TestMutationOperators:

    def test_算子列表非空(self):
        """M1~M10 十类算子不应为空"""
        assert len(ALL_OPERATORS) >= 8, \
            f"变异算子只有 {len(ALL_OPERATORS)} 类，覆盖不足"

    def test_算子有唯一标识(self):
        """每类算子应有唯一 name，便于统计"""
        names = [op.get("name") or op.get("kind")
                 for op in ALL_OPERATORS] if isinstance(
                     ALL_OPERATORS[0], dict) else [
            getattr(op, "__name__", str(op)) for op in ALL_OPERATORS]

        assert len(names) == len(set(names)), \
            f"算子标识有重复：{names}"

    def test_能生成多种变异(self):
        """
        答案与上下文足够丰富时应产出多类变异。

        只有一类变异时，检出率的分母太小——
        「10 条里发现 10 条」的检出率毫无意义。
        """
        answer = ("自动化测试框架分为基础层、用例层、数据层、报告层，"
                  "共包含 4 个部分。")
        context = "自动化测试框架分层：基础层、用例层、数据层。"

        muts = generate_mutations(answer, context, max_per_kind=2)
        kinds = {m.name for m in muts}

        assert len(muts) >= 4, f"只生成了 {len(muts)} 条变异"
        assert len(kinds) >= 3, f"只覆盖 {len(kinds)} 类变异：{kinds}"

    def test_变异确实改变了内容(self):
        """变异后必须与原文不同，否则不算变异"""
        answer = "自动化测试框架分为基础层、用例层、数据层。"
        context = "自动化测试框架分层：基础层、用例层、数据层。"

        muts = generate_mutations(answer, context, max_per_kind=2)
        changed = [m for m in muts if m.mutated != answer]

        assert changed, "所有变异都没改变内容"

    def test_空答案变异产出空串(self):
        """
        「空答案」变异必须真的产出空串。

        它验证的是「该答却没答」这类缺陷能否被识别，
        不是「生成一个别的错误答案」。
        """
        answer = "自动化测试框架分为基础层、用例层、数据层。"
        context = "自动化测试框架分层：基础层。"

        muts = generate_mutations(answer, context, max_per_kind=2)
        empty = [m for m in muts if m.name == "空答案"]

        assert empty, "应生成空答案变异"
        assert all(m.mutated == "" for m in empty)

    def test_无变异结构时不报错(self):
        """
        内容贫瘠时可能生成不出变异。

        这时应返回空列表而不是抛异常——
        「没有可变异的东西」是正常情况，不是错误。
        """
        muts = generate_mutations("好", "", max_per_kind=2)
        assert isinstance(muts, list)

    def test_拒答样本不做内容变异(self):
        """
        拒答类素材上追加伪事实是**无效变异**。

        「参考资料中未提及。该方法由 Kent Beck 提出」——
        幻觉检测器看到拒答标记就短路返回了，必然检不出。

        如果把它算作漏报，会：
          · 让检出率虚低
          · 更糟：报告里出现一个根本不存在的「已知盲区」

        这是变异素材选错，不是评测器能力问题。
        """
        refusal = "参考资料中未提及。"
        muts = generate_mutations(refusal, "", max_per_kind=2)

        assert muts == [], \
            "拒答 + 空上下文不应产生任何变异"
        for m in muts:
            assert m.name != "追加伪事实", \
                "在拒答后面追加伪事实是无效变异"

    def test_拒答样本有上下文时可做拒答倒置(self):
        """
        拒答样本在**有上下文**时可以测「拒答倒置」。

        因为此时「本该有答案却拒答」是真实缺陷，
        值得验证——尽管它的 should_detect 是 False
        （幻觉检测器不该管过度拒答，那是拒答评测器的职责）。
        """
        muts = generate_mutations(
            "参考资料中未提及。", "自动化测试框架分为基础层。",
            max_per_kind=2)

        assert muts, "有上下文时拒答样本仍应能产生变异"
        assert all(m.name == "拒答倒置" for m in muts), \
            f"应只生成拒答倒置，实际：{[m.name for m in muts]}"

    def test_拒答倒置不算幻觉检出率分母(self):
        """
        拒答倒置的 should_detect 必须为 False。

        它验证的是「过度拒答」，属于拒答评测器的职责。
        若设成 True，幻觉检出率会被凭空拖低——
        而幻觉检测器本来就不该检出它。
        """
        muts = generate_mutations(
            "自动化测试框架分为基础层。",
            "自动化测试框架分为基础层、用例层、数据层。",
            max_per_kind=2)

        for m in muts:
            if m.name == "拒答倒置":
                assert m.should_detect is False, \
                    "拒答倒置不应计入幻觉检出率的分母"


# ============================================================
# 二、检出率
# ============================================================
class TestDetectionRate:

    def test_实体调换应被检出(self):
        """
        具体缺陷逐个验证，而不是只看总检出率。

        总检出率高不代表每类缺陷都能发现——
        某类检出率为 0 可能被其他类拉高了平均。
        """
        context = "自动化测试框架分层：基础层、用例层、数据层。"

        muts = generate_mutations(
            "自动化测试框架分为基础层、用例层、数据层。",
            context, max_per_kind=2)

        detect = make_detector(forbidden_facts=["报告层"])
        entity_swaps = [m for m in muts if m.name == "实体调换"]

        for m in entity_swaps:
            got = detect(m.mutated, context)
            assert got, (
                f"实体调换未被检出：{m.mutated[:40]}"
                f"（这是最典型的幻觉类型，检不出等于评测器基本无效）")

    def test_数字篡改应被检出(self):
        context = "自动化测试框架分为基础层、用例层、数据层。"
        muts = generate_mutations(
            "自动化测试框架包含 3 个部分。", context, max_per_kind=2)

        detect = make_detector()
        numeric = [m for m in muts if m.name == "数字篡改"]

        for m in numeric:
            assert detect(m.mutated, context), \
                f"数字篡改未被检出：{m.mutated[:40]}"

    def test_跑题应被检出(self):
        """完全跑题的答案最容易被检出——这算基线水平"""
        context = "自动化测试框架分层：基础层、用例层、数据层。"
        muts = generate_mutations(
            "自动化测试框架分为基础层、用例层、数据层。",
            context, max_per_kind=2)

        detect = make_detector()
        irrelevant = [m for m in muts if m.name == "答案替换"]

        for m in irrelevant:
            assert detect(m.mutated, context), "跑题未被检出"

    def test_检出率必须可计算且在区间内(self):
        """
        单条用例的变异检出率应在 [0, 1]。

        如果算出 1.5 或负数，说明统计逻辑有 bug。
        """
        case = GoldSample(
            id="MUT_01", category="entity_swap",
            question="自动化测试框架分为哪几层？",
            answer="自动化测试框架分为基础层、用例层、数据层。",
            context="自动化测试框架分层：基础层、用例层、数据层。",
            label_hallucination=False,
        )

        r = run_mutation_test(case, make_detector())

        assert 0.0 <= r["detect_rate"] <= 1.0, \
            f"检出率越界：{r['detect_rate']}"
        # 自洽性：检出 + 漏报 = 应检出的总数。
        # 注意不是 total —— total 含「不适用」的变异（空答案、上下文污染），
        # 它们本来就不该检出，计入会凭空扣分。
        assert r["should_detect"] == r["detected"] + r["missed"], \
            "计数不自洽：should_detect != detected + missed"
        assert r["total"] == r["should_detect"] + r["not_applicable"]

    def test_无可变异时明确说明(self):
        """
        答案里没有可变异结构时，应返回 None 而非 0.0。

        返回 0.0 会被统计成「一条都没检出」，
        拉低整体检出率——而实际是「没东西可测」。
        这两种情况必须区分。

        用「答案与上下文完全一致」构造无可变异的情况：
        变异算子要产生的是「与原文不同」的内容，
        原文本身就是上下文时，没有任何算子适用。
        """
        case = GoldSample(
            id="MUT_EMPTY", category="x",
            question="q", answer="自动化测试", context="自动化测试",
            label_hallucination=False,
        )

        r = run_mutation_test(case, make_detector())
        if r["total"] == 0:
            # 真的没有变异 → 检出率必须是无定义
            assert r["detect_rate"] is None, \
                "无可变异时检出率应为 None，不是 0.0"
        else:
            # 有变异但其中「应检出」的为 0 → 同样无定义
            assert r["should_detect"] > 0 or r["detect_rate"] is None, \
                "没有应检出的变异时，检出率不应为具体数值"
            assert 0.0 <= r["detect_rate"] <= 1.0

    def test_检出率分母只算应检出的(self):
        """
        非幻觉类变异（空答案、上下文污染）不应计入分母。

        它们的 should_detect=False，验证的是「另一类缺陷能否识别」，
        混进幻觉检出率的分母会凭空扣分，
        让报出来的数字比真实能力差。
        """
        case = GoldSample(
            id="MUT_DENOM", category="entity_swap",
            question="自动化测试框架分为哪几层？",
            answer="自动化测试框架分为基础层、用例层、数据层。",
            context="自动化测试框架分层：基础层、用例层、数据层。",
            label_hallucination=False,
        )

        r = run_mutation_test(case, make_detector())

        assert r["total"] == r["should_detect"] + r["not_applicable"], \
            "total 应等于应检出数 + 不适用数"
        assert r["should_detect"] == r["detected"] + r["missed"], \
            "应检出的变异应等于已检出 + 漏报"
        assert r["detect_rate"] == round(
            r["detected"] / r["should_detect"], 4) \
            if r["should_detect"] else None


# ============================================================
# 三、整体评估
# ============================================================
class TestOverallCapability:

    """
    整体检出能力评估。

    这一组的价值不在于「数字好看」，
    而在于**如实记录能力边界**——
    面试时能说出「我的评测器在哪类缺陷上会漏报」，
    比说「检出率 100%」可信得多。
    """

    @pytest.fixture(scope="class")
    def capability_report(self):
        """用内置 Gold Set 跑一次完整变异测试"""
        from eval.datasets.gold_set import SAMPLES

        detect = make_detector()
        per_case = []
        for s in SAMPLES:
            r = run_mutation_test(s, detect)
            if r["total"] > 0:
                per_case.append(r)

        total = sum(r["total"] for r in per_case)
        detected = sum(r["detected"] for r in per_case)
        missed = sum(r["missed"] for r in per_case)

        return {
            "cases": len(per_case),
            "total": total,
            "detected": detected,
            "missed": missed,
            "rate": (detected / total) if total else None,
            "details": per_case,
        }

    def test_应产出变异样本(self, capability_report):
        """
        变异测试本身要能跑出东西。

        跑不出样本说明算子或数据有问题，
        那后面的检出率数字都没有意义。
        """
        assert capability_report["total"] > 0, \
            "变异测试未产出任何样本"

    def test_检出率应在合理下限之上(self, capability_report):
        """
        底线断言：检出率不能低于 50%。

        低于这个数说明评测器基本无效——
        连实体调换、数字篡改这种最典型的幻觉都抓不住。
        真实值应该高得多，这里只设底线防回归。
        """
        rate = capability_report["rate"]
        assert rate is not None
        assert rate >= 0.5, \
            f"检出率 {rate:.1%} 低于底线 50%，评测器可靠性存疑"

    def test_检出率不得为完美(self, capability_report):
        """
        **刻意的反向断言**：检出率不该是 100%。

        理由
        ----
        字符级覆盖率算法有先天盲区：
        「系统不会崩溃」vs「系统不会挂掉」这类
        「换一种说法说同一件事」的伪装，它抓不到。

        如果哪天这个断言失败了，说明两件事之一：
          · 实现退化了（原来能抓的现在抓不到）
          · 盲区被修好了（这是好事，需要更新文档）

        无论哪种，都需要人来确认——所以它必须失败。
        这比写一个「100% 很好」的测试有价值得多。
        """
        rate = capability_report["rate"]
        assert rate < 1.0, (
            "检出率为 100%，这很可能是数据或实现有问题。"
            "字符级算法必然存在盲区，"
            "如实记录漏报类型比刷高数字有价值")

    def test_漏报的变异应可列举(self, capability_report):
        """
        漏报必须能说出是哪些，不能只有一个总数。

        「漏报率 15%」无法指导改进；
        「漏报集中在 X 类」才知道该修哪里。
        """
        missed_items = []
        for case in capability_report["details"]:
            for m in case.get("results", []):
                if m.get("correct") is False:
                    missed_items.append({
                        "name": m.get("name"),
                        "case_id": case.get("case_id"),
                    })

        # 允许漏报，但必须能被列举出来
        assert isinstance(missed_items, list)
        if missed_items:
            kinds = {m["name"] for m in missed_items}
            assert all(kinds), "漏报项必须有类型名"


# ============================================================
# 四、CI 入口
# ============================================================
class TestMutationCli:

    def test_可作为脚本运行(self):
        """
        `python -m eval.mutation` 应能产出报告。

        变异测试必须能独立运行——
        藏在某个 pytest 用例里，就没人会主动去看结果。
        """
        import subprocess
        import sys
        import os

        root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        r = subprocess.run(
            [sys.executable, "-m", "eval.mutation"],
            cwd=root, capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONPATH": root},
        )

        assert r.returncode == 0, f"运行失败：{r.stderr[:300]}"
        assert "变异" in r.stdout, "输出应说明这是变异测试"
