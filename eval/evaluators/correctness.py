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

from eval.evaluators.base import EvalResult


# ============================================================
# 共享常量
# ============================================================

# 拒答标记词表（全项目唯一定义）
#
# 早先 faithfulness.py 与 correctness.py 各写一份，
# 且内容不同——后果是同一个回答在幻觉检测里被认作拒答，
# 在相关性检测里却不被认作，两个指标自相矛盾。
# 现在统一放这里，两个模块都从这里取。
REFUSAL_MARKERS = (
    "未提及",
    "没有相关",
    "无法回答",
    "不知道",
    "不明确",
    # 以下是 SUT 实测会用到的表述，
    # 漏掉它们会让「正确拒答」被当成「跑题」
    "未提及该问题",
    "未提及相关内容",
    "无法作答",
    "无相关",
)

from eval.evaluators.faithfulness import (
    content_chars, split_sentences, split_claims, _STOPWORDS,
)


# ============================================================
# 正确性：正确性
# ============================================================
class CorrectnessResult(EvalResult):
    """
    正确性判定结果（统一结构 + 兼容旧字段）

    改造说明
    --------
    早先这是一个独立的 dataclass，字段为
    `score / is_correct / matched_facts / ...`，
    与其它评测器的 dict 返回值不兼容。

    现在继承 EvalResult 拿到统一的
    `score / passed / reason / evaluator / details`，
    同时保留 `is_correct`、`matched_facts` 等旧字段，
    令现有调用方与测试无需改动。

    这样做比重写所有调用方风险低：
    逐个迁移，随时可停。
    """

    # 兼容别名：让 r.is_correct 继续可用
    @property
    def is_correct(self) -> bool:
        return bool(self.passed)

    @property
    def matched_facts(self) -> List[str]:
        return self.details.get("matched_facts", [])

    @property
    def missing_facts(self) -> List[str]:
        return self.details.get("missing_facts", [])

    @property
    def wrong_facts(self) -> List[str]:
        return self.details.get("wrong_facts", [])

    @property
    def method(self) -> str:
        return self.details.get("method", "required_facts")

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        # 保留旧字段名，兼容既有 JSON 消费方
        d["is_correct"] = self.is_correct
        d["matched_facts"] = self.matched_facts
        d["missing_facts"] = self.missing_facts
        d["wrong_facts"] = self.wrong_facts
        d["method"] = self.method
        return d


def _correctness(score, is_correct, reason, **details) -> CorrectnessResult:
    """构造统一结构的正确性结果"""
    return CorrectnessResult(
        score=score,
        passed=is_correct,
        reason=reason,
        evaluator="correctness",
        details=details,
    )


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
        return _correctness(0.0, False, "答案为空，无法判定正确性")

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
        return _correctness(0.0, False, f"答案包含禁止内容：{wrong}",
                           wrong_facts=wrong)

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

        return _correctness(
            round(score, 3), is_correct,
            f"必需事实覆盖 {len(matched)}/{len(required_facts)}"
            f"（阈值 {threshold}）",
            matched_facts=matched, missing_facts=missing,
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
        return _correctness(
            round(best, 3), is_correct,
            f"与可接受答案的最佳相似度 {best:.2f}（阈值 {threshold}）",
            method="acceptable_answers",
        )

    # ---- Step 4: 降级到 reference 相似度 ----
    # 明确标注 method，让报告能说明"此用例正确性判定依据较弱"
    if reference_answer:
        a_chars = content_chars(answer)
        r_chars = content_chars(reference_answer)
        if r_chars:
            score = len(a_chars & r_chars) / len(r_chars)
            return _correctness(
                round(score, 3), score >= threshold,
                f"仅按参考答案相似度判定（{score:.2f}），"
                f"未提供 required_facts，判定依据较弱",
                method="reference_similarity_degraded",
            )

    return _correctness(
        0.0, False,
        "无可用判定依据（无 required_facts / acceptable_answers / reference）",
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
def _is_refusal(answer: str) -> bool:
    """
    判断回答是否属于「明确表示无依据」。

    词表来源
    --------
    与 faithfulness.py 的拒答词表合并到此处，
    由 REFUSAL_MARKERS 统一维护。

    为什么要统一
    ------------
    早先两个文件各写一份，且内容不同
    （faithfulness 用 5 个词，correctness 用另外 7 个）。
    后果：同一个回答在幻觉检测里被认作拒答，
    在相关性检测里却不被认作——两个指标自相矛盾。

    词表条目
    --------
    前 5 个是基础词，覆盖「未提及 / 不知道」这类标准说法；
    后面几个是 SUT 实测会用到的表述
    （如「未提及该问题相关内容，无法回答」），
    漏掉它们会让正确拒答被当成跑题。
    """
    if not answer:
        return False
    return any(m in answer for m in REFUSAL_MARKERS)


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

    # ---- 拒答场景的特判 ----
    #
    # 拒答用例本来就没有 required_facts / reference_answer
    # （正确答案就是「拒答」本身，给了标准答案反而矛盾，
    #   validate_dataset 会明确报错）。
    #
    # 但相关性评测走到最后会落到
    #   「缺少 required_facts / reference / context → 判不相关」
    # 于是**正确拒答反被记为 IRRELEVANT**。
    #
    # 后果很严重：系统的拒答能力明明是对的，
    # 指标却显示「跑题」，把好结果算成坏结果。
    # 真实案例：扩展集的 5 条拒答用例全部因此被判失败。
    #
    # 所以这里先识别拒答，直接放行。
    if _is_refusal(answer):
        return {
            "score": 1.0, "is_relevant": True,
            "method": "refusal_acknowledged",
            "reason": "系统明确表示无依据，属正确拒答，不按跑题处理",
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
# 否定词与程度副词。出现在句中时，其前为「主张」，其后为「修饰」。
# 同一主张被「完全不关注」修饰，与被「关注」修饰，是直接矛盾。
_NEGATION_PARTICLES = ("完全不", "毫不", "绝不", "不", "无", "非", "未")

# 常见的谓语动词。用于「无否定词」时切出共同主体。
#
# 为什么需要它
# ------------
# 「接口测试关注状态码」与「接口测试完全不关注状态码」这一对里，
# 只有第二句含否定词。只按否定词切，第一句会整句成为主体，
# 于是两句主体不同、永远比不出矛盾——而它们恰恰是最典型的矛盾。
#
# 所以对不含否定词的句子，改按谓语动词切：
#     「接口测试 | 关注状态码」
#     「接口测试 | 完全不关注状态码」
# 主体一致，谓语一正一反，冲突立刻可见。
_PREDICATE_VERBS = ("关注", "涉及", "包括", "包含", "需要", "必须",
                    "支持", "使用", "采用", "负责", "用于", "检查",
                    "验证", "覆盖", "关注点", "重新执行", "执行",
                    "确认", "发现", "说明", "定义", "描述", "关注点")


def _pick_splitter(sentences):
    """
    为一组句子选定统一的切分策略。

    思路
    ----
    三级候选规则（否定词 / 系动词 / 谓语动词）逐个试，
    选「能成功切出主体的句子数」最多的那条。

    为什么必须统一
    --------------
    若每句各自挑第一条能用的规则，切分粒度会不一致：
        「框架分为基础层」  切不出 → 整句当主体
        「框架不为基础层」  切出「框架」
    主体一个是整句、一个是「框架」，就永远比不出矛盾。
    """
    candidates = (
        _split_by_negation,
        _split_by_copula,
        _split_by_verb,
    )

    best_fn, best_score = None, 0
    for fn in candidates:
        score = 0
        for s in sentences:
            subj, _ = fn(s)
            if subj:
                score += 1
        if score > best_score:
            best_fn, best_score = fn, score

    if best_fn is None:
        return lambda s: (None, None)
    return best_fn


def _split_by_copula(sent):
    """
    按系动词切分主谓。切不出返回 (None, None)。

    只认「是」，以及不属于动词一部分的「为」
    ------------------------------
    早先用 `if cop in sent` 匹配「为」，
    于是「自动化框架**分**为**基础层」被切成「框架分」+「基础层」——
    这个「为」其实是「分为」的一部分，不是系动词。

    主体被切错之后，它与其它句子就无法对齐到同一主体，
    冲突检测随之失效。现在要求「为」前一个字不是动词字，
    「分为/称为/作为/成为」这类都不算系动词。
    """
    if "是" in sent:
        return sent.split("是", 1)

    for i, ch in enumerate(sent):
        if ch == "为" and i > 0:
            if sent[i - 1] in "分称作变":
                continue
            return sent[:i], sent[i + 1:]

    return None, None


def _split_by_negation(sent):
    """
    按否定词切分「主张 + 修饰」。

    例：
        「接口测试完全不关注状态码」  → 主张「接口测试」，修饰「完全不关注状态码」

    切不出返回 (None, None)。
    """
    best_pos = -1
    for neg in _NEGATION_PARTICLES:
        pos = sent.find(neg, 1)     # 从1开始：避免句首就是否定词
        if pos > 0 and (best_pos == -1 or pos < best_pos):
            best_pos = pos

    if best_pos <= 0:
        return None, None

    subject = sent[:best_pos].strip()
    predicate = sent[best_pos:].strip()
    if not subject or not predicate:
        return None, None
    return subject, predicate


def _split_by_verb(sent):
    """
    按谓语动词切分「主体 + 谓语」。

    用于不含否定词、也无系动词的句子：
        「接口测试关注状态码」→「接口测试」+「关注状态码」
        「回归测试重新执行用例」→「回归测试」+「重新执行用例」

    这样才能与含否定词的兄弟句对齐到同一主体。
    """
    best_pos = -1
    for verb in _PREDICATE_VERBS:
        pos = sent.find(verb)
        if pos > 0 and (best_pos == -1 or pos < best_pos):
            best_pos = pos

    if best_pos <= 0:
        return None, None

    subject = sent[:best_pos].strip()
    predicate = sent[best_pos:].strip()
    if not subject or len(subject) < 2:
        # 主体太短（单字）说明切错了位置
        return None, None
    return subject, predicate


def _polarity_differs(subject, pred_a, pred_b):
    """
    判断两个谓语的极性是否相反。

    为什么要单独检测
    ----------------
    「接口测试关注状态码」与「接口测试完全不关注状态码」
    的实词Jaccard 是 **0.625**——因为「关注状态码」四个字完全重合，
    只多了「完全不」三个字。

    而冲突判定阈值是 0.5，0.625 > 0.5，于是判为「不矛盾」。

    这是纯字面相似度的根本局限：
    **否定词增加的字符数远少于它所反转的含义**。
    「是」vs「不是」、「可以」vs「完全不可以」都是同一类问题。

    所以当字面相似度高、但极性相反时，必须单独报冲突。

    判据：两个谓语中，恰有一个含否定词。
    含否定词的那句与另一句在语义上直接对立。
    """
    a_neg = _has_negation(pred_a)
    b_neg = _has_negation(pred_b)
    if a_neg == b_neg:
        return False        # 极性相同，不算这类冲突

    # 极性相反时进一步确认：
    # 去掉否定词后，两句应有较高的字面重合——
    # 否则它们可能谈的是完全不同的两件事，只是恰好一句有一句没有。
    a_stripped = _strip_negation(pred_a)
    b_stripped = _strip_negation(pred_b)
    ca, cb = content_chars(a_stripped), content_chars(b_stripped)
    if not ca or not cb:
        return False
    sim = len(ca & cb) / len(ca | cb)
    return sim >= 0.5


def _has_negation(text):
    """谓语中是否含否定词"""
    return any(neg in text for neg in _NEGATION_PARTICLES)


def _strip_negation(text):
    """去掉否定词，用于比较「剥掉否定后」的实质内容是否一致"""
    for neg in _NEGATION_PARTICLES:
        text = text.replace(neg, "")
    return text


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

    主体抽取的三级回退
    ------------------
    1. 有冒号 → 冒号前是主体（「冒烟测试：验证核心功能」）
    2. 有「是/为」→ 之前是主体（「自动化框架分为基础层」）
    3. **都没有 → 按否定词切分**
       「接口测试关注状态码」vs「接口测试完全不关注状态码」
       这两句没有冒号也没有「是/为」，早先的实现会把**整句**当成主体，
       于是两个主体不同，永远比不出冲突。
       但它们的差异只是「关注」与「完全不关注」——恰恰是最典型的矛盾。

    局限
    ----
    依赖「主体在前、谓语在后」的汉语 SVO 语序。
    对「北京是首都 / 上海是首都」这类主谓倒装或并列主体的场景
    仍会漏检，因为切不出共同主体。
    """
    sentences = split_sentences(context, min_len=4)
    if len(sentences) < 2:
        return {"has_conflict": False, "reason": "上下文片段不足", "conflicts": []}

    # 抽取每句的"主体"：冒号前部分，或句首的名词性成分
    subjects = {}
    # 谓语原文，保留用于极性冲突检测
    preds_raw = {}

    # ---- 选定全句共享的切分策略 ----
    #
    # 候选规则按「能切出主体的句子数」排序，选覆盖最多的一条。
    # 覆盖少说明该规则不适用于这批句子，
    # 用它会导致部分句子整句作主体、粒度与其它句子不一致。
    no_colon = [s for s in sentences
                if "：" not in s and ":" not in s]

    splitter = _pick_splitter(no_colon) if no_colon else None

    for sent in sentences:
        # 以冒号为界： "冒烟测试：验证核心功能" -> ("冒烟测试", "验证核心功能")
        if "：" in sent:
            subj, pred = sent.split("：", 1)
        elif ":" in sent:
            subj, pred = sent.split(":", 1)
        else:
            # 无冒号：用**全句共享**的切分策略。
            #
            # 为什么必须共享
            # --------------
            # 早先的实现让每句各自按三级回退挑第一条能切出主体的规则，
            # 结果切分粒度不一致：
            #     「框架分为基础层」     切不出 → 整句作主体
            #     「框架不为基础层」     切出「框架」
            # 两者主体一个是整句、一个是「框架」，永远比不出冲突。
            #
            # 共享策略：先在**所有句子**上探测哪种规则能切出结果，
            # 选定后所有句子用同一条规则切，主体粒度自然对齐。
            subj, pred = splitter(sent)
            if subj is None:
                subj, pred = sent, ""

        subj = subj.strip()
        pred_chars = content_chars(pred)
        if not subj:
            continue
        subjects.setdefault(subj, []).append(pred_chars)
        # 同时保留谓语原文，供极性冲突检测使用
        preds_raw.setdefault(subj, []).append(pred)

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
                        "kind": "different_predicate",
                    })
                elif _polarity_differs(subj, preds_raw[subj][i],
                                       preds_raw[subj][j]):
                    # 字面高度相似但极性相反 —— 这类冲突 Jaccard 看不见
                    conflicts.append({
                        "subject": subj,
                        "variants": [preds_raw[subj][i],
                                     preds_raw[subj][j]],
                        "similarity": round(sim, 3),
                        "kind": "polarity_conflict",
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
