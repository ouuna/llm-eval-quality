"""
正确性与相关性指标
---------------------------------
修复审计发现的两个致命缺陷：

P0-2  expected_output 从不参与判定
    旧实现：passed = no_halluc and is_complete
    缺陷：域内用例只要"用词在上下文出现过"就算通过，
         域内准确率报100% 是假的。

P0-3  相关性指标方向反了
    旧实现：|question_chars ∩ answer_chars| / |question_chars|
    问题：这测的是"答案有没有抄问题里的字"，不是"答案是否切题"。
    反例：问题「如何提高代码质量？」
          答「代码质量的提高方法」——实词全覆盖得 1.0，但这是废话
          答「提高代码质量需要遵循规范」得 0.4，但这才是正确答案
         完全无关的回答也能得高分，只要复述了问题词。

本模块的修法
------------
- correctness：基于 required_facts / acceptable_answers 判定，
              不再依赖 reference_answer 的字符串匹配
- relevance：算 answer ↔ reference 的相似度（方向正确），
             并以 required_facts 覆盖作为辅助判据
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import re

from eval.evaluators.faithfulness import (
    content_chars, split_sentences, split_claims, _STOPWORDS,
)


# ============================================================
# 正确性：正确性
# ============================================================
@dataclass
class CorrectnessResult:
    """正确性判定结果"""
    score: float                                  # 0-1
    is_correct: bool
    matched_facts: List[str] = field(default_factory=list)
    missing_facts: List[str] = field(default_factory=list)
    wrong_facts: List[str] = field(default_factory=list)
    reason: str = ""
    method: str = "required_facts"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "is_correct": self.is_correct,
            "matched_facts": self.matched_facts,
            "missing_facts": self.missing_facts,
            "wrong_facts": self.wrong_facts,
            "reason": self.reason,
            "method": self.method,
        }


def evaluate_correctness(answer: str, required_facts: List[str],
                        forbidden_facts: List[str] = None,
                        acceptable_answers: List[str] = None,
                        reference_answer: str = None,
                        threshold: float = 0.7) -> CorrectnessResult:
    """
    判定答案是否正确

    判定优先级
    ----------
    1. forbidden_facts 命中    → 直接判错（最高优先级）
    2. required_facts 覆盖率    → 达到阈值即正确
    3. acceptable_answers 匹配  → 语义等价表述视为正确
    4. 无required_facts 时退回 reference 相似度（明确标注方法降级）

    与旧实现的区别
    --------------
    旧实现完全不看expected_output，所以"答案错误但用词在上下文里"
    这种情况会通过。本实现会因缺失 required_facts 而判错。
    """
    answer = (answer or "").strip()
    forbidden_facts = forbidden_facts or []
    required_facts = required_facts or []
    acceptable_answers = acceptable_answers or []

    if not answer:
        return CorrectnessResult(
            score=0.0, is_correct=False,
            reason="答案为空，无法判定正确性",
        )

    # ---- Step 1: forbidden 命中即判错 ----
    wrong = []
    for ff in forbidden_facts:
        if ff in answer:
            wrong.append(ff)
            continue
        ff_chars = content_chars(ff)
        a_chars = content_chars(answer)
        if ff_chars and len(ff_chars & a_chars) / len(ff_chars) >= 0.8:
            wrong.append(ff)

    if wrong:
        return CorrectnessResult(
            score=0.0, is_correct=False,
            wrong_facts=wrong,
            reason=f"答案包含禁止内容：{wrong}",
        )

    # ---- Step 2: required_facts 覆盖率 ----
    if required_facts:
        a_chars = content_chars(answer)
        matched, missing = [], []
        for rf in required_facts:
            rf_chars = content_chars(rf)
            if not rf_chars:
                matched.append(rf)
                continue
            if len(rf_chars & a_chars) / len(rf_chars) >= 0.5:
                matched.append(rf)
            else:
                missing.append(rf)

        score = len(matched) / len(required_facts)
        is_correct = score >= threshold

        return CorrectnessResult(
            score=round(score, 3),
            is_correct=is_correct,
            matched_facts=matched,
            missing_facts=missing,
            reason=(
                f"必需事实覆盖 {len(matched)}/{len(required_facts)}"
                f"（阈值 {threshold}）"
            ),
            method="required_facts",
        )

    # ---- Step 3: acceptable_answers 匹配 ----
    if acceptable_answers:
        a_chars = content_chars(answer)
        best = 0.0
        for aa in acceptable_answers:
            aa_chars = content_chars(aa)
            if not aa_chars:
                continue
            best = max(best, len(aa_chars & a_chars) / len(aa_chars))

        is_correct = best >= threshold
        return CorrectnessResult(
            score=round(best, 3),
            is_correct=is_correct,
            reason=f"与可接受答案的最佳相似度 {best:.2f}（阈值 {threshold}）",
            method="acceptable_answers",
        )

    # ---- Step 4: 降级到 reference 相似度 ----
    # 明确标注 method，让报告能说明"此用例正确性判定依据较弱"
    if reference_answer:
        a_chars = content_chars(answer)
        r_chars = content_chars(reference_answer)
        if r_chars:
            score = len(a_chars & r_chars) / len(r_chars)
            return CorrectnessResult(
                score=round(score, 3),
                is_correct=score >= threshold,
                reason=(
                    f"仅按参考答案相似度判定（{score:.2f}），"
                    f"未提供 required_facts，判定依据较弱"
                ),
                method="reference_similarity_degraded",
            )

    return CorrectnessResult(
        score=0.0, is_correct=False,
        reason="无可用判定依据（无 required_facts / acceptable_answers / reference）",
        method="insufficient_ground_truth",
    )


# ============================================================
# 完整性：Completeness
# ============================================================
def evaluate_completeness(answer: str, required_facts: List[str],
                        context: str = None,
                        threshold: float = 0.7) -> Dict[str, Any]:
    """
    判定回答是否完整（是否覆盖全部必需事实）

    与正确性的区别
    --------------
    正确性：说得对不对（有没有 forbidden、required 是否覆盖）
    完整性：说全了没有（required 的覆盖率是否达到阈值）

    典型反例：对比型问题只答了一半
        问题："A 和 B 的区别"
        答：只讲了 A
        → 相关性高（有问就有答）
        → 正确性无法判定（只说了一个）
        → 完整性低（缺B 的要点）
    """
    answer = (answer or "").strip()
    required_facts = required_facts or []

    if not answer:
        return {
            "score": 0.0, "is_complete": False,
            "missing_facts": list(required_facts),
            "reason": "答案为空",
        }

    if not required_facts:
        return {
            "score": 1.0, "is_complete": True,
            "missing_facts": [],
            "reason": "未指定必需事实，跳过完整性判定",
        }

    a_chars = content_chars(answer)
    covered, missing = [], []
    for rf in required_facts:
        rf_chars = content_chars(rf)
        if not rf_chars:
            covered.append(rf)
            continue
        if len(rf_chars & a_chars) / len(rf_chars) >= 0.5:
            covered.append(rf)
        else:
            missing.append(rf)

    score = len(covered) / len(required_facts)
    return {
        "score": round(score, 3),
        "is_complete": score >= threshold,
        "covered_facts": covered,
        "missing_facts": missing,
        "reason": f"覆盖 {len(covered)}/{len(required_facts)} 个必需事实",
    }


# ============================================================
# 相关性：Relevance
# ============================================================
def evaluate_relevance(answer: str, reference_answer: str = None,
                       required_facts: List[str] = None,
                       question: str = None,
                       context: str = None) -> Dict[str, Any]:
    """
    判定回答是否切题

    修复 P0-3 的关键：改变计算方向。

    旧实现（方向错误）
    ------------------
    score = |question_chars ∩ answer_chars| / |question_chars|
    → 测"答案有没有抄问题里的字"
    → 完全无关的回答只要复述问题词就能得高分

    新实现（三级递降）
    ------------------
    1. 有 required_facts → 用事实覆盖率作为切题依据（最可靠）
       因为"切题"的本质是"回答了被问的东西"，
       而 required_facts 就是"被问的东西"的显式声明。
    2. 无 required_facts 但有 reference → 算 answer↔reference 相似度
    3. 都没有 → 退化为 answer↔context 覆盖率，并显式标注降级

    三级都不满足时，明确返回"依据不足"而非编造分数。
    """
    answer = (answer or "").strip()

    if not answer:
        return {
            "score": 0.0, "is_relevant": False,
            "method": "empty",
            "reason": "答案为空",
        }

    a_chars = content_chars(answer)

    # ---- Level 1: required_facts ----
    if required_facts:
        covered = 0
        for rf in required_facts:
            rf_chars = content_chars(rf)
            if not rf_chars:
                covered += 1
                continue
            if len(rf_chars & a_chars) / len(rf_chars) >= 0.5:
                covered += 1
        score = covered / len(required_facts)
        return {
            "score": round(score, 3),
            "is_relevant": score >= 0.5,
            "method": "required_facts_coverage",
            "reason": f"命中 {covered}/{len(required_facts)} 个被问要点",
        }

    # ---- Level 2: answer ↔ reference ----
    if reference_answer:
        r_chars = content_chars(reference_answer)
        if r_chars:
            score = len(a_chars & r_chars) / len(r_chars)
            return {
                "score": round(score, 3),
                "is_relevant": score >= 0.5,
                "method": "answer_reference_similarity",
                "reason": f"与参考答案实词相似度 {score:.2f}",
            }

    # ---- Level 3: answer ↔ context（降级）----
    if context:
        c_chars = content_chars(context)
        if c_chars:
            score = len(a_chars & c_chars) / len(a_chars)
            return {
                "score": round(score, 3),
                "is_relevant": score >= 0.4,
                "method": "answer_context_similarity_degraded",
                "reason": (
                    f"仅按与上下文相似度判定（{score:.2f}），"
                    f"未提供 required_facts 或 reference"
                ),
            }

    return {
        "score": 0.0, "is_relevant": False,
        "method": "insufficient_ground_truth",
        "reason": "缺少 required_facts / reference / context，无法判定相关性",
    }


# ============================================================
# 拒答：Refusal
# ============================================================
def evaluate_refusal(answer: str, should_refuse: bool,
                     forbidden_facts: List[str] = None,
                     markers: List[str] = None) -> Dict[str, Any]:
    """
    判定拒答行为是否正确

    两种失败模式
    ------------
    1. 该拒答却作答（false refusal）→ 严重，编造内容
    2. 不该拒答却拒答（over-refusal）→ 保守，能力不足

    区分这两种失败很重要：错误方向不同，修复方式也不同。
    """
    answer = (answer or "").strip()
    markers = markers or ["未提及", "没有相关", "无法回答", "不知道", "不明确",
                          "不存在于", "参考资料中未"]

    refused = (not answer) or any(m in answer for m in markers)

    if should_refuse:
        if refused:
            return {"is_correct": True, "refused": True,
                    "reason": "正确拒答", "mode": "correct_refusal"}
        # 未拒答：是否命中 forbidden 决定严重程度
        hits = []
        for ff in forbidden_facts or []:
            if ff in answer:
                hits.append(ff)
        return {
            "is_correct": False, "refused": False,
            "forbidden_hits": hits,
            "reason": (
                f"应拒答却作答，且命中禁止内容：{hits}"
                if hits else "应拒答却作答"
            ),
            "mode": "false_answer",
        }

    # 不该拒答
    if refused:
        return {
            "is_correct": False, "refused": True,
            "reason": "不应拒答却拒答（过度保守）",
            "mode": "over_refusal",
        }
    return {"is_correct": True, "refused": False,
            "reason": "正常作答", "mode": "correct_answer"}


# ============================================================
# 上下文冲突：Context Conflict
# ============================================================
def detect_context_conflict(context: str, answer: str = None) -> Dict[str, Any]:
    """
    检测上下文是否自相矛盾

    场景：
        Context A: 冒烟测试：验证核心功能是否可用
        Context B: 冒烟测试：完整回归全部功能
        → 同一主体（冒烟测试）的描述互相矛盾

    实现方式
    --------
    1. 抽取每个片段的"主体"（冒烟测试）与"谓述描述"（其余部分）
    2. 同一主体出现多个不同谓述 → 冲突

    局限
    ----
    依赖"主体在前、谓语在后"的汉语 SVO 语序。
    对"北京是首都 / 上海是首都"这类主谓倒装或并列主体的场景
    需要更复杂的语义分析，当前实现会漏检。
    """
    sentences = split_sentences(context, min_len=4)
    if len(sentences) < 2:
        return {"has_conflict": False, "reason": "上下文片段不足", "conflicts": []}

    # 抽取每句的"主体"：冒号前部分，或句首的名词性成分
    subjects = {}
    for sent in sentences:
        # 以冒号为界： "冒烟测试：验证核心功能" -> ("冒烟测试", "验证核心功能")
        if "：" in sent:
            subj, pred = sent.split("：", 1)
        elif ":" in sent:
            subj, pred = sent.split(":", 1)
        else:
            # 无冒号时，尝试按"是/为"切分主体与谓语
            for cop in ("是", "为"):
                if cop in sent:
                    subj, pred = sent.split(cop, 1)
                    break
            else:
                # 兜底：整句作为主体（此时不会检出冲突）
                subj, pred = sent, ""

        subj = subj.strip()
        pred_chars = content_chars(pred)
        if not subj:
            continue
        subjects.setdefault(subj, []).append(pred_chars)

    # 同一主体有多个差异显著的谓述 → 冲突
    conflicts = []
    for subj, preds in subjects.items():
        if len(preds) < 2:
            continue
        # 两两比较谓述实词集合的 Jaccard，低于 0.5 视为矛盾
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                a, b = preds[i], preds[j]
                if not a or not b:
                    continue
                sim = len(a & b) / len(a | b)
                if sim < 0.5:
                    conflicts.append({
                        "subject": subj,
                        "variants": [
                            "".join(sorted(a)),
                            "".join(sorted(b)),
                        ],
                        "similarity": round(sim, 3),
                    })

    has_conflict = len(conflicts) > 0
    result = {
        "has_conflict": has_conflict,
        "conflicts": conflicts,
        "reason": (f"检出 {len(conflicts)} 处冲突" if has_conflict
                   else "未检出冲突"),
    }

    if has_conflict and answer:
        conflict_words = ["冲突", "矛盾", "不一致", "两个", "两种", "不同"]
        acknowledged = any(w in answer for w in conflict_words)
        result["acknowledged"] = acknowledged
        result["reason"] += f"；模型{'已' if acknowledged else '未'}识别冲突"

    return result
