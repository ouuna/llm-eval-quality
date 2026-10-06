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

    for ent in set(entities):
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

    for n in set(nums):
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
    """M7 拒答倒置"""
    return [Mutation(
        id="M7_refusal_inversion",
        name="拒答倒置",
        target="answer",
        original=answer,
        mutated="参考资料中未提及。",
        should_detect=True,
        rationale="上下文明确有答案却拒答，属过度保守",
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
    """
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

    for m in mutations:
        m_answer = m.mutated if m.target == "answer" else answer
        m_context = m.mutated if m.target == "context" else context

        got = detected_fn(m_answer, m_context)
        expect = m.should_detect

        if expect and got:
            detected += 1
        elif expect and not got:
            missed += 1
        # 非幻觉类变异（如空答案）不计漏报

        results.append({
            **m.to_dict(),
            "detected": got,
            "correct": got == expect if expect else None,
        })

    return {
        "case_id": getattr(case, "id", "?"),
        "total": len(results),
        "should_detect": sum(1 for r in results if r["should_detect"]),
        "detected": detected,
        "missed": missed,
        "detect_rate": round(
            detected / sum(1 for r in results if r["should_detect"]), 4)
        if sum(1 for r in results if r["should_detect"]) else None,
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


if __name__ == "__main__":
    from eval.evaluators.faithfulness import detect_hallucination

    def detect(answer, context):
        return detect_hallucination(answer, context)["has_hallucination"]

    # 用一条有代表性的答案演示
    sample_answer = "自动化测试框架分为基础层、用例层、数据层。"
    sample_context = "自动化测试框架分层：基础层负责请求封装，用例层负责断言，数据层负责测试数据。"

    class C:
        id = "DEMO"
        answer = sample_answer
        context = sample_context

    report = run_mutation_test(C(), detect)
    print("=" * 66)
    print(f"变异测试：{report['case_id']}")
    print("=" * 66)
    print(f"  变异总数{report['total']}  应检出 {report['should_detect']}"
          f"  检出 {report['detected']}  漏报 {report['missed']}")
    print(f"  检出率：{report['detect_rate']}")
    print()
    for r in report["results"]:
        mark = "✓" if r["correct"] else ("漏检" if r["should_detect"] else "-")
        print(f"  [{mark:4}] {r['name']:10} {r['rationale'][:44]}")
