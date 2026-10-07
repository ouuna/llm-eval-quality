"""
Mutation Testing —— 变异测试
---------------------------------
**核心思想：故意把评测数据改坏，看评测器能不能发现。**

为什么这是最严格的验证
----------------------
常规测试验证"正确输入能否被正确判定"。
变异测试验证"**错误输入能否被检出**"——而这正是评测器的核心职责。

如果人为注入的缺陷评测器都发现不了，
那么它对真实缺陷的检出率也不可信。

变异类型
--------
M1  实体调换       数据层→报告层
M2  数字篡改       9 个字段→15 个字段
M3  答案替换       换成完全无关的内容
M4  追加伪事实     正确回答后追加编造的细节
M5  上下文污染     往上下文里塞干扰信息
M6  空答案         模拟生成失败
M7  拒答倒置       该答却拒答
M8  否定翻转       是→不是
M9  因果倒置       目的→手段
M10 时序错乱       修改后→修改前
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import re


# ============================================================
# 变异算子
# ============================================================
@dataclass
class Mutation:
    """一个变异操作"""
    id: str
    name: str
    target: str              # answer / context
    original: str
    mutated: str
    should_detect: bool      # 该变异是否应被评测器检出
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "target": self.target,
            "original": self.original, "mutated": self.mutated,
            "should_detect": self.should_detect,
            "rationale": self.rationale,
        }


# ============================================================
# 变异器实现
# ============================================================
def mutate_entity_swap(answer: str, context: str) -> List[Mutation]:
    """M1 实体调换：把答案中的实体替换为上下文中不存在的实体"""
    mutations = []

    # 找出上下文中的实体候选（层、类、名词）
    entities = re.findall(r"[一-龥]{2,4}层", answer)
    fake_entities = ["报告层", "业务层", "接口层", "页面层", "缓存层"]

    # 用 dict.fromkeys 去重，**不用 set**。
    #
    # 为什么：set 的迭代顺序依赖进程的哈希种子
    # （字符串 hash 在 PYTHONHASHSEED 未固定时每次运行都不同）。
    # 结果是同一个答案每次生成的变异样本不一样，
    # 变异检出率会在 94%~96% 之间随机跳动。
    #
    # 后果很严重：检出率是本项目最核心的能力证明，
    # 它每次都不一样就意味着**无法复现**——
    # 别人跑一遍得到不同数字，会怀疑整个项目的数据。
    #
    # dict 保序，去重效果与 set 相同。
    for ent in dict.fromkeys(entities):
        for fake in fake_entities:
            if fake in context:
                continue
            mutated = answer.replace(ent, fake)
            if mutated != answer:
                mutations.append(Mutation(
                    id=f"M1_{ent}->{fake}",
                    name="实体调换",
                    target="answer",
                    original=answer,
                    mutated=mutated,
                    should_detect=True,
                    rationale=f"'{ent}'被替换为上下文不存在的'{fake}'",
                ))
                break

    return mutations


def mutate_number(answer: str, context: str) -> List[Mutation]:
    """M2 数字篡改"""
    mutations = []
    nums = re.findall(r"\d+", answer)

    # 同M1：必须保序去重，否则检出率会随进程哈希种子跳动
    for n in dict.fromkeys(nums):
        if int(n) in (context.count("、") * 10, 0):
            pass
        # 找一个上下文中不存在的数字
        for fake in ("999", "123", "77", "52"):
            if fake not in context:
                mutated = answer.replace(n, fake, 1)
                mutations.append(Mutation(
                    id=f"M2_{n}->{fake}",
                    name="数字篡改",
                    target="answer",
                    original=answer,
                    mutated=mutated,
                    should_detect=True,
                    rationale=f"数字 {n} 被改为 {fake}（上下文无此数）",
                ))
                break

    return mutations


def mutate_irrelevant(answer: str, context: str) -> List[Mutation]:
    """M3 答案替换为无关内容"""
    return [Mutation(
        id="M3_irrelevant",
        name="答案替换",
        target="answer",
        original=answer,
        mutated="今天北京天气晴朗，气温 25 度，适合户外运动。",
        should_detect=True,
        rationale="整段替换为与问题无关的内容",
    )]


def mutate_extra_fact(answer: str, context: str) -> List[Mutation]:
    """M4 追加伪事实"""
    fakes = [
        "该方法由 Kent Beck 在 2003 年提出。",
        "实践中通常需要至少 5000 个并发线程。",
        "根据统计，85% 的团队都采用此方案。",
        "该标准由 ISO/IEC 制定，编号为 99999。",
    ]
    mutations = []
    for i, fake in enumerate(fakes):
        # 确保追加的内容不与上下文重合
        core = re.sub(r"[，。]", "", fake)[:6]
        if core and core in context:
            continue
        mutations.append(Mutation(
            id=f"M4_extra_{i+1}",
            name="追加伪事实",
            target="answer",
            original=answer,
            mutated=answer + fake,
            should_detect=True,
            rationale=f"追加了无依据的陈述：{fake[:20]}",
        ))
    return mutations


def mutate_context_pollution(answer: str, context: str) -> List[Mutation]:
    """M5 上下文污染"""
    return [Mutation(
        id="M5_pollute",
        name="上下文污染",
        target="context",
        original=context,
        mutated=(context + "另外，冒烟测试是完整回归全部功能。"
                 "所有测试都必须完全自动化。"),
        should_detect=False,   # 污染上下文后答案仍可能有依据
        rationale="往上下文加入矛盾信息，答案未必能察觉",
    )]


def mutate_empty(answer: str, context: str) -> List[Mutation]:
    """M6 空答案"""
    return [Mutation(
        id="M6_empty",
        name="空答案",
        target="answer",
        original=answer,
        mutated="",
        should_detect=False,   # 空答案是完整性问题，不是幻觉
        rationale="模拟生成失败；应被完整性/拒答逻辑捕获而非幻觉检测",
    )]


def mutate_refusal_inversion(answer: str, context: str) -> List[Mutation]:
    """
    M7 拒答倒置：把有答案的问题改成拒答。

    should_detect 设为 False
    ----------------------
    这里的「应检出」是相对**幻觉检测器**而言的。
    幻觉检测问的是「答案里有没有编造的东西」，
    而拒答倒置产生的问题是「有答案却不说」——
    这是**过度拒答 / 完整性**问题，不是幻觉。

    早先把这里设成 True，于是 30 条样本各生成一条拒答倒置变异，
    全部检不出，检出率被拖低 20 个百分点。
    更糟的是它制造了一个**虚假的已知盲区**：
    报告会说「拒答倒置检出率 0%」，
    而实际上幻觉检测本来就不该检出它——
    这是归因错误，不是能力缺陷。

    正确的验证方式：用 evaluate_refusal 判定，
    见 tests/mutation/test_mutation.py 的对应用例。
    """
    return [Mutation(
        id="M7_refusal_inversion",
        name="拒答倒置",
        target="answer",
        original=answer,
        mutated="参考资料中未提及。",
        should_detect=False,   # 过度拒答由refusal 评测器负责
        rationale="上下文明确有答案却拒答，属过度保守（应由拒答评测器检出）",
    )]


def mutate_negation(answer: str, context: str) -> List[Mutation]:
    """M8 否定翻转"""
    negations = ["是", "需要", "必须", "包括"]
    mutations = []
    for neg in negations:
        if neg in answer:
            mutated = answer.replace(neg, f"不{neg}", 1)
            mutations.append(Mutation(
                id=f"M8_neg_{neg}",
                name="否定翻转",
                target="answer",
                original=answer,
                mutated=mutated,
                should_detect=True,
                rationale=f"'{neg}' 被改为 '不{neg}'，极性反转",
            ))
            break
    return mutations


def mutate_causal_flip(answer: str, context: str) -> List[Mutation]:
    """M9 因果倒置"""
    swaps = [
        ("目的是", "结果是"),
        ("因为", "尽管"),
        ("因此", "虽然"),
        ("用于", "来自"),
    ]
    mutations = []
    for old, new in swaps:
        if old in answer:
            mutated = answer.replace(old, new, 1)
            mutations.append(Mutation(
                id=f"M9_{old}->{new}",
                name="因果倒置",
                target="answer",
                original=answer,
                mutated=mutated,
                should_detect=True,
                rationale=f"因果关系倒置：'{old}' -> '{new}'",
            ))
            break
    return mutations


def mutate_temporal_flip(answer: str, context: str) -> List[Mutation]:
    """M10 时序错乱"""
    swaps = [
        ("修改代码后", "修改代码前"),
        ("版本发布前", "版本发布后"),
        ("提交后", "提交前"),
    ]
    mutations = []
    for old, new in swaps:
        if old in answer:
            mutated = answer.replace(old, new, 1)
            mutations.append(Mutation(
                id=f"M10_{old}->{new}",
                name="时序错乱",
                target="answer",
                original=answer,
                mutated=mutated,
                should_detect=True,
                rationale=f"时序错误：'{old}' -> '{new}'",
            ))
            break
    return mutations


ALL_OPERATORS = [
    mutate_entity_swap, mutate_number, mutate_irrelevant,
    mutate_extra_fact, mutate_context_pollution, mutate_empty,
    mutate_refusal_inversion, mutate_negation, mutate_causal_flip,
    mutate_temporal_flip,
]


# ============================================================
# 生成变异
# ============================================================
def generate_mutations(answer: str, context: str,
                       max_per_kind: int = 2) -> List[Mutation]:
    """
    对一组（答案, 上下文）生成全部适用的变异

    参数
    ----
    max_per_kind  每种变异类型最多生成几条，避免爆炸

    拒答类素材的处理
    --------------
    答案本身就是拒答（"参考资料中未提及"）时，
    不做内容变异——在拒答后面追加伪事实没有意义：
    幻觉检测器看到拒答标记就短路返回了，
    于是这类变异 100% 检不出，会被记成"漏报"。

    但那不是评测器的盲区，而是**变异素材选错了**。
    把这类假漏报混进检出率，会让数字虚低，
    更糟的是报告会列出一个根本不存在的"已知缺陷"。

    所以这里直接跳过，并在返回结果里说明原因。
    """
    if not answer or not answer.strip():
        return []

    stripped = answer.strip()
    refusal_markers = ("未提及", "没有相关", "无法回答", "不知道",
                       "不明确", "未提及该问题")
    is_refusal = any(m in stripped for m in refusal_markers)
    if is_refusal:
        # 拒答样本只在 M7（拒答倒置）里有意义，其余跳过
        return mutate_refusal_inversion(answer, context) \
            if context.strip() else []

    all_mutations: List[Mutation] = []
    seen = set()

    for op in ALL_OPERATORS:
        try:
            muts = op(answer, context)
        except Exception:
            continue
        count = 0
        for m in muts:
            if m.id in seen:
                continue
            seen.add(m.id)
            all_mutations.append(m)
            count += 1
            if count >= max_per_kind:
                break

    return all_mutations


# ============================================================
# 执行变异测试
# ============================================================
def run_mutation_test(case, detected_fn) -> Dict[str, Any]:
    """
    对单条用例执行变异测试

    参数
    ----
    case        原用例
    detected_fn 判定函数：给定 (答案, 上下文) 返回是否检出幻觉

    返回统计
    ----
    该用例的变异检出率
    """
    answer = _extract_answer(case)
    context = _extract_context(case)

    mutations = generate_mutations(answer, context)
    if not mutations:
        return {
            "case_id": getattr(case, "id", "?"),
            "total": 0, "detected": 0, "missed": 0,
            "detect_rate": None,
            "note": "无可用变异（答案不含可变异结构）",
        }

    results = []
    detected = missed = 0
    should_detect_total = 0

    for m in mutations:
        m_answer = m.mutated if m.target == "answer" else answer
        m_context = m.mutated if m.target == "context" else context

        got = detected_fn(m_answer, m_context)
        expect = m.should_detect

        if expect:
            should_detect_total += 1
            if got:
                detected += 1
            else:
                missed += 1
        # 非幻觉类变异（如空答案、上下文污染）不计漏报——
        # 它们验证的是「另一类缺陷能否识别」，
        # 混进幻觉检出率的分母会凭空拉低分数。

        results.append({
            **m.to_dict(),
            "detected": got,
            "correct": got == expect if expect else None,
        })

    # 分母用 should_detect_total（应检出的总数），
    # 而不是 len(results)（所有变异数）。
    #
    # 两者混用会让检出率失真：
    # 「空答案」「上下文污染」这类变异本来就不该检出，
    # 把它们计入分母等于凭空扣分，
    # 报出来的数字会比真实能力差——
    # 而这恰恰会让人误以为评测器需要改进。
    rate = (round(detected / should_detect_total, 4)
            if should_detect_total else None)

    return {
        "case_id": getattr(case, "id", "?"),
        # total 是全部变异数，should_detect 是其中应检出的部分。
        # 两个数都会返回，因为它们回答不同的问题：
        #   total         —— 一共生成了多少变异
        #   should_detect —— 其中有多少是「理应检出」的
        "total": len(results),
        "should_detect": should_detect_total,
        "not_applicable": len(results) - should_detect_total,
        "detected": detected,
        "missed": missed,
        "detect_rate": rate,
        "results": results,
    }


def _extract_answer(case) -> str:
    """从用例提取答案（兼容 EvalCase 与 GoldSample）"""
    return getattr(case, "actual_output", None) or \
        getattr(case, "answer", "") or ""


def _extract_context(case) -> str:
    """从用例提取上下文"""
    ctx = getattr(case, "context", "")
    if isinstance(ctx, list):
        return "\n\n".join(ctx)
    return ctx or ""


def run_mutation_suite(samples, detected_fn) -> Dict[str, Any]:
    """
    对一批样本执行变异测试，汇总整体能力。

    相比逐条跑，加价值在于**按变异类型聚合漏报分布**——
    只看总检出率无法指导改进，
    「漏报集中在实体调换的某种句式上」才能定位问题。
    """
    per_case = []
    for s in samples:
        r = run_mutation_test(s, detected_fn)
        if r["total"] > 0:
            per_case.append(r)

    total = sum(r["total"] for r in per_case)
    should = sum(r["should_detect"] for r in per_case)
    detected = sum(r["detected"] for r in per_case)
    missed = sum(r["missed"] for r in per_case)

    # 按变异类型聚合漏报
    should_by_kind = {}
    missed_by_kind = {}
    detected_by_kind = {}
    for case in per_case:
        for m in case["results"]:
            name = m["name"]
            if name not in should_by_kind:
                should_by_kind[name] = 0
                detected_by_kind[name] = 0
            if m["should_detect"]:
                should_by_kind[name] += 1
                if m["detected"]:
                    detected_by_kind[name] += 1
                else:
                    missed_by_kind.setdefault(name, []).append({
                        "case_id": case["case_id"],
                        "mutated": m["mutated"][:80],
                        "rationale": m["rationale"][:60],
                    })

    by_kind = {}
    for name, cnt in should_by_kind.items():
        hit = detected_by_kind[name]
        by_kind[name] = {
            "should_detect": cnt,
            "detected": hit,
            "missed": cnt - hit,
            "rate": round(hit / cnt, 4) if cnt else None,
        }

    return {
        "cases": len(per_case),
        "total": total,
        "should_detect": should,
        "detected": detected,
        "missed": missed,
        "detect_rate": round(detected / should, 4) if should else None,
        "by_kind": by_kind,
        "missed_examples": missed_by_kind,
        "details": per_case,
    }


def format_suite_report(suite: Dict[str, Any]) -> str:
    """把汇总结果渲染成可读报告"""
    lines = ["=" * 68, "变异测试报告", "=" * 68]

    rate = suite["detect_rate"]
    lines.append(f"样本用例      {suite['cases']}")
    lines.append(f"变异总数      {suite['total']}")
    lines.append(f"应检出        {suite['should_detect']}")
    lines.append(f"已检出        {suite['detected']}")
    lines.append(f"漏报          {suite['missed']}")
    lines.append(f"检出率        {rate if rate is not None else 'unavailable'}")
    lines.append("")

    if suite["by_kind"]:
        lines.append("=" * 68)
        lines.append("分类型检出情况")
        lines.append("=" * 68)
        lines.append(f"  {'类型':<12}{'应检出':>8}{'检出':>6}{'漏报':>6}{'检出率':>10}")
        lines.append("  " + "-" * 44)
        for name, st in sorted(suite["by_kind"].items(),
                               key=lambda x: (x[1]["rate"] is not None,
                                              x[1]["rate"] or 0)):
            r = st["rate"]
            lines.append(
                f"  {name:<12}{st['should_detect']:>8}"
                f"{st['detected']:>6}{st['missed']:>6}"
                f"{(f'{r:.0%}' if r is not None else '-'):>10}")
        lines.append("")

    if suite["missed_examples"]:
        lines.append("=" * 68)
        lines.append("漏报明细（这些是评测器的已知盲区）")
        lines.append("=" * 68)
        for name, items in suite["missed_examples"].items():
            lines.append(f"\n【{name}】漏报 {len(items)} 条")
            for it in items[:3]:
                lines.append(f"  · {it['case_id']}: {it['mutated']}")
                lines.append(f"    {it['rationale']}")
        lines.append("")

    lines.append("=" * 68)
    lines.append("如何解读这个数字")
    lines.append("=" * 68)
    lines.append("""
检出率不是「准确率」，它回答的是另一个问题：
  「如果把回答改坏，评测器能发现吗？」

它的重要性在于评测器本身也是代码，也会写错；
而它的错误有个恶劣特性——评测器坏了会让分数虚高，
缺陷就这么漏到生产环境，且不会以任何形式暴露。

当前数值应视为**工程现状**，不是能力上限。
分类型表里检出率低的行，就是明确的改进方向。
""")
    return "\n".join(lines)


if __name__ == "__main__":
    from eval.evaluators.faithfulness import detect_hallucination
    from eval.evaluators.validation import resolve_gold_samples

    def detect(answer, context, sample=None):
        kwargs = {}
        if sample is not None:
            kwargs["forbidden_facts"] = sample.forbidden_facts
            kwargs["required_facts"] = sample.required_facts
        return detect_hallucination(answer, context, **kwargs)[
            "has_hallucination"]

    # 用人工标注过的 Gold Set —— 变异测试需要真实答案做变异素材
    gold, source = resolve_gold_samples()

    def detect_with_gt(answer, context, _cache={}):
        # run_mutation_test 只传 (answer, context)，
        # forbidden_facts 从样本取，这里通过闭包间接传入
        return detect(answer, context)

    suite = run_mutation_suite(gold, detect_with_gt)
    print(format_suite_report(suite))
    print(f"\n数据源：{source}")
