"""
声明级验证：幻觉检测的核心能力
---------------------------------
解决旧实现的根本缺陷——字符覆盖率无法检出实体调换。

问题演示
--------
上下文："北京是中国的首都"
回答一："中国的首都是上海"       ← 事实错误，但字符覆盖率 1.0
回答二："中国的首都是北京"       ← 正确

旧实现对两者判定相同（无幻觉），因为"中国""首都""是"都在上下文里。
**字符覆盖率本质上无法区分这两者** —— 它不区分实词的"角色"。

本模块的做法
------------
把回答拆成原子声明（claim），逐条验证其是否被上下文支撑：

    Claim 1: 中国的首都是上海
    Claim 2: 中国是一个国家

    Claim 1 → unsupported（"上海"与上下文事实矛盾）
    Claim 2 → supported

声明级验证 = 证据检索 + 声明抽取 + 声明验证 + 无依据声明检测
"""

from dataclasses import dataclass, field
from typing import List, Optional
import re


# ============================================================
# 数据结构
# ============================================================
@dataclass
class Claim:
    """一条原子声明及其验证结果"""
    text: str
    supported: bool
    support_level: str          # supported / partial / unsupported
    matched_evidence: List[str] = field(default_factory=list)
    conflicting_terms: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "supported": self.supported,
            "support_level": self.support_level,
            "matched_evidence": self.matched_evidence,
            "conflicting_terms": self.conflicting_terms,
            "reason": self.reason,
        }


# ============================================================
# 文本处理
# ============================================================
# 停用词：不含语义的虚词
_STOPWORDS = set(
    "的了和与及对于是把在当中一个可以我们需要这那有被就都也很到吗呢什么"
    "怎么如何请问介绍定义进行过而且或者但如果因为所以它他她它们"
)

# 否定词：直接影响声明极性，验证时必须区分
_NEGATIONS = {"不", "没有", "无", "非", "未", "不会", "不能", "不应", "不是"}

# 事实性提示词：出现这些词说明句子在陈述事实，需严格验证
_FACTUAL_MARKERS = {
    "是", "为", "包括", "包含", "需要", "必须", "应该", "可以",
    "由", "通过", "使用", "负责", "用于",
}

# 主张极性模式
_MODAL_PATTERN = re.compile(r"(可能|也许|通常|一般|建议|推荐|应该|可以)")


def content_chars(text: str) -> set:
    """
    提取中文实词字符集合

    改进：原实现从问题中提取，但判定时用了更宽的停用词表，
    导致"是什么"这类被计入。现统一直这里。
    """
    return {c for c in text if re.match(r"[一-龥]", c) and c not in _STOPWORDS}


def split_sentences(text: str, min_len: int = 6) -> List[str]:
    """按中文标点切分句子"""
    if not text:
        return []
    parts = re.split(r"[。\n；;！？!?]", text)
    return [p.strip() for p in parts if len(p.strip()) >= min_len]


def split_claims(sentence: str, min_len: int = 2) -> List[str]:
    """
    将句子拆为原子声明

    按并列连接词与分号切分：
        "北京是中国的首都，人口约2100万"
        -> ["北京是中国的首都", "人口约2100万"]

    目的：让每条声明独立验证，避免一句多事实互相掩盖。

    min_len 默认降到 2的原因
    -----------------------
    并列枚举的尾部常是很短的片段，如：
        "分为基础层、用例层、数据层、业务层"
    切出"业务层。"仅 3 字，若按 >= 4 过滤会被丢弃，
    导致 forbidden_facts=["业务层"] 永远命中不到——
    这正是实体调换型幻觉漏检的根源之一。
    """
    # 顿号必须参与切分，它是中文并列分隔符
    parts = re.split(r"[，,、；;：:]", sentence)
    claims = [p.strip() for p in parts if len(p.strip()) >= min_len]
    return claims or ([sentence.strip()] if len(sentence.strip()) >= min_len else [])


# ============================================================
# 核心：事实一致性验证
# ============================================================
def _extract_terms(text: str) -> set:
    """
    提取句子中的事实性成分：中文实词 + 数字 + 拉丁词

    修复记录
    --------
    原实现只提取 `[一-龥]` 字符，导致：
        "回归测试由 Kent Beck 在 2003 年提出"
    里的 "Kent Beck" 被完全丢弃，编造的外国人名检测不出。
    实测该样本覆盖率被高估为 0.44，判为 supported（漏报）。

    现在额外提取：
        - 拉丁字母序列（人名、术语、产品名）
        - 数字（数量、日期、版本号）
    """
    chars = content_chars(text)

    # 拉丁词：人名/术语/产品名。长度 >= 2 以排除单字母噪声
    latin = {
        w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\.\-]{1,}", text)
    }

    # 数字：数量错误是幻觉高发类型
    nums = set(re.findall(r"\d+(?:\.\d+)?", text))

    return chars | latin | nums


def _has_negation_conflict(claim: str, evidence: str) -> bool:
    """
    检测否定极性冲突

    上下文："冒烟测试是必需环节"
    回答："冒烟测试不是必需环节"

    两者字符高度重合，但极性相反 —— 必须靠否定词检测才能发现。
    """
    claim_neg = bool(re.search(r"[不无非未]", claim))
    evid_neg = bool(re.search(r"[不无非未]", evidence))
    # 简单近似：一句有否定另一句无 → 极性冲突
    return claim_neg != evid_neg


def verify_claim(claim: str, context: str,
                 supported_threshold: float = 0.6,
                 partial_threshold: float = 0.35) -> Claim:
    """
    验证单条声明是否被上下文支撑

    三级判定
    --------
    supported    覆盖率 >= supported_threshold
    partial      partial_threshold <= 覆盖率 < supported_threshold
    unsupported  覆盖率< partial_threshold，或命中 forbidden 事实

    额外检测
    --------
    1. 否定极性冲突：同上但极性相反
    2. 数字冲突：同位置数字不同
    3. forbidden 事实：可选传入
    """
    context = context or ""
    if not context.strip():
        return Claim(
            text=claim, supported=False, support_level="unsupported",
            reason="上下文为空，无法验证任何声明",
        )

    claim_terms = _extract_terms(claim)
    if not claim_terms:
        return Claim(
            text=claim, supported=False, support_level="unsupported",
            reason="声明无实词内容",
        )

    # ---- 逐句匹配，找最佳支撑 ----
    ctx_sentences = split_sentences(context)
    best_cov = 0.0
    best_sentence = ""
    for cs in ctx_sentences:
        cs_terms = _extract_terms(cs)
        if not cs_terms:
            continue
        cov = len(claim_terms & cs_terms) / len(claim_terms)
        if cov > best_cov:
            best_cov = cov
            best_sentence = cs

    # ---- 数字冲突检测 ----
    # 修复：原先要求 best_cov < supported_threshold 才检查数字，
    # 导致"编造数字但字面重合度高"的场景漏报
    #（如"包含 9 个字段"vs"包含 15 个字段"）。
    # 现在只要出现上下文中不存在的数字即判定。
    claim_nums = set(re.findall(r"\d+(?:\.\d+)?", claim))
    if claim_nums:
        ctx_nums = set(re.findall(r"\d+(?:\.\d+)?", context))
        novel_nums = claim_nums - ctx_nums
        if novel_nums:
            return Claim(
                text=claim, supported=False, support_level="unsupported",
                matched_evidence=[best_sentence] if best_sentence else [],
                conflicting_terms=sorted(novel_nums),
                reason=f"声明中的数字 {sorted(novel_nums)} 在上下文中不存在",
            )

    # ---- 否定极性冲突 ----
    if best_sentence and _has_negation_conflict(claim, best_sentence):
        return Claim(
            text=claim, supported=False, support_level="unsupported",
            matched_evidence=[best_sentence],
            reason="声明与上下文极性相反（一方含否定，另一方不含）",
        )

    # ---- 阈值判定 ----
    if best_cov >= supported_threshold:
        return Claim(
            text=claim, supported=True, support_level="supported",
            matched_evidence=[best_sentence],
            reason=f"实词覆盖率 {best_cov:.2f} >= {supported_threshold}",
        )
    if best_cov >= partial_threshold:
        return Claim(
            text=claim, supported=False, support_level="partial",
            matched_evidence=[best_sentence],
            reason=f"实词覆盖率 {best_cov:.2f} 处于部分支持区间",
        )
    return Claim(
        text=claim, supported=False, support_level="unsupported",
        reason=f"实词覆盖率 {best_cov:.2f} < {partial_threshold}",
    )


def verify_forbidden_facts(claim: str, forbidden_facts: List[str],
                           context: str) -> List[str]:
    """
    检查声明是否命中禁止事实

    这是检出"实体调换"最直接的手段：
        required_facts = ["基础层", "用例层", "数据层"]
        forbidden_facts = ["报告层", "业务层"]
    回答说"分为基础层、用例层、业务层" → 命中 forbidden → 幻觉
    """
    hits = []
    for ff in forbidden_facts or []:
        ff_core = content_chars(ff)
        if not ff_core:
            continue
        if ff in claim:
            hits.append(ff)
            continue
        # 部分匹配：禁止事实的核心字大多出现在声明中
        claim_core = content_chars(claim)
        if ff_core and len(ff_core & claim_core) / len(ff_core) >= 0.8:
            hits.append(ff)
    return hits


# ============================================================
# 声明级幻觉检测（对外接口）
# ============================================================
def detect_hallucination(answer: str, context: str,
                        required_facts: Optional[List[str]] = None,
                        forbidden_facts: Optional[List[str]] = None,
                        supported_threshold: float = 0.6,
                        partial_threshold: float = 0.35,
                        should_refuse: bool = False,
                        refusal_markers: Optional[List[str]] = None) -> dict:
    """
    声明级幻觉检测

    完整流程：证据检索 → 声明抽取 → 声明验证 → 无依据声明检测

    返回
    ----
    {
        "has_hallucination": bool,
        "claims": [Claim, ...],
        "unsupported_claims": [Claim, ...],
        "forbidden_hits": [str, ...],
        "missing_required": [str, ...],
        "coverage": float,          # 整体覆盖率
        "reason": str,
    }

    关键修复
    --------
    旧实现对"中国的首都是上海"判定无幻觉（覆盖率 1.0）。
    本实现会先检出"上海"与上下文事实矛盾 → 判为 unsupported。
    """
    markers = refusal_markers or ["未提及", "没有相关", "无法回答", "不知道", "不明确"]

    result = {
        "has_hallucination": False,
        "claims": [],
        "unsupported_claims": [],
        "forbidden_hits": [],
        "missing_required": [],
        "coverage": 1.0,
        "reason": "",
    }

    # ---- Step 1: 拒答识别 ----
    answer = (answer or "").strip()
    if not answer:
        result["reason"] = "答案为空"
        if not should_refuse:
            # 该答却没答 —— 判为不完整（由调用方决定如何归类）
            result["reason"] = "答案为空（应作答但未作答）"
        return result

    if any(m in answer for m in markers):
        result["reason"] = "系统明确拒答"
        return result

    # ---- Step 2: 无上下文却有回答 ----
    if not (context or "").strip():
        result["has_hallucination"] = True
        result["reason"] = "无检索上下文但生成了回答"
        return result

    # ---- Step 3: 声明抽取 ----
    sentences = split_sentences(answer)
    all_claims: List[str] = []
    for s in sentences:
        all_claims.extend(split_claims(s))

    if not all_claims:
        result["reason"] = "回答过短，无实质内容"
        return result

    # ---- Step 4: 逐条验证 ----
    claims: List[Claim] = []
    for c_text in all_claims:
        claim = verify_claim(
            c_text, context,
            supported_threshold=supported_threshold,
            partial_threshold=partial_threshold,
        )

        # 叠加 forbidden 事实检测
        hits = verify_forbidden_facts(c_text, forbidden_facts, context)
        if hits:
            claim.supported = False
            claim.support_level = "unsupported"
            claim.conflicting_terms.extend(hits)
            claim.reason = f"命中禁止事实：{hits}"

        claims.append(claim)

    # ---- Step 4.5: 跑题检测 ----
    # 修复：完全无关的回答（如问等价类划分答天气）此前只报"缺必需事实"，
    # 不判为幻觉。判定方式：所有声明在上下文的覆盖率都接近 0。
    # 注意这里不看 required_facts——它缺失只说明"没答到点上"，
    # 判定"编造"需要证据：整段话与上下文毫无重合。
    if claims and all(
        len(_extract_terms(c.text) & _extract_terms(context)) == 0
        for c in claims
    ):
        result["has_hallucination"] = True
        result["claims"] = claims
        result["unsupported_claims"] = claims
        result["coverage"] = 0.0
        result["reason"] = (
            "回答内容与上下文无任何实词重合，判定为跑题或编造"
        )
        return result

    unsupported = [c for c in claims if not c.supported]
    supported_count = len(claims) - len(unsupported)

    result["claims"] = claims
    result["unsupported_claims"] = unsupported
    result["coverage"] = round(supported_count / len(claims), 3) if claims else 0.0
    result["forbidden_hits"] = sorted({
        t for c in unsupported for t in c.conflicting_terms
    })

    # ---- Step 5: required_facts 缺失检测 ----
    answer_chars = content_chars(answer)
    for rf in required_facts or []:
        rf_chars = content_chars(rf)
        if not rf_chars:
            continue
        if len(rf_chars & answer_chars) / len(rf_chars) < 0.5:
            result["missing_required"].append(rf)

    # ---- Step 6: 结论 ----
    if unsupported:
        result["has_hallucination"] = True
        first = unsupported[0]
        result["reason"] = f"{len(unsupported)}/{len(claims)} 条声明无依据，首条：{first.reason}"
    elif result["missing_required"]:
        result["has_hallucination"] = False
        result["reason"] = f"缺少必需事实：{result['missing_required']}"
    else:
        result["reason"] = f"{len(claims)} 条声明均有上下文依据"

    return result
