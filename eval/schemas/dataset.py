"""
评测数据集 Schema 定义
---------------------------------
统一 Evaluation Dataset、Ground Truth 与 Case 的数据结构。

设计原则
--------
1. **区分 answer 与 evidence**：参考答案用于判定正确性，
   证据（required_facts）用于判定幻觉，两者不可混用。
2. **expected_behavior 显式声明**：不允许从 category 推断期望行为，
   避免"域外用例必须有 expected_output"这类隐式约定导致的误判。
3. **required/forbidden facts 是事实型校验的依据**：
   比字符串匹配更精确，可检出数字错误与实体调换。
4. **容错而非全或无**：acceptable_answers 允许语义等价表述，
   避免因改写导致误判。

术语约定
--------
expected_behavior:
    answer      —— 必须给出答案（且答案正确）
    refuse      —— 必须拒答（无依据时不应编造）
    clarify     —— 应要求澄清（问题不完整）
    partial     —— 允许部分回答（上下文不足但可给出有限信息）

Reference 与 Ground Truth 的关系：
    reference_answer     参考答案，仅用于对比参考，不作为唯一判据
    acceptable_answers   可接受的等价表述列表
    required_facts       必须出现的事实（判定 correctness 用）
    forbidden_facts      禁止出现的事实（检出幻觉的关键）
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json

# ============================================================
# 常量定义
# ============================================================

# 评测类别共 12 类
#
# 基础三类
#   in_domain         域内，知识库中有答案
#   out_domain        域外，知识库中无答案，应拒答
#   ambiguous         模糊，问题不完整，应要求澄清
#
# 扩展九类
#   boundary          边界，问题在知识库边缘
#   multi_hop         多跳，需要综合多处信息
#   negative          否定式提问，易答错
#   numeric           数字与时序事实
#   paraphrase        同义改写，检验检索鲁棒性
#   insufficient_ctx  上下文不足
#   context_conflict  上下文冲突
#   unanswerable      本质不可回答
#   prompt_injection  越权诱导与信息泄露
CATEGORIES = {
    "in_domain",
    "out_domain",
    "ambiguous",
    "boundary",
    "multi_hop",
    "negative",
    "numeric",
    "paraphrase",
    "insufficient_ctx",
    "context_conflict",
    "unanswerable",
    "prompt_injection",
}

DIFFICULTY = {"easy", "medium", "hard"}

EXPECTED_BEHAVIOR = {"answer", "refuse", "clarify", "partial"}

# 每类的测试目的说明（写入报告，用于说明用例设计意图）
CATEGORY_PURPOSE = {
    "in_domain":        "验证检索召回与答案正确性",
    "out_domain":       "验证无依据时能否正确拒答（幻觉抑制）",
    "ambiguous":        "验证问题不完整时是否要求澄清而非强行作答",
    "boundary":         "验证部分覆盖场景的处理是否恰当",
    "multi_hop":        "验证多处信息综合能力",
    "negative":         "验证否定式提问的理解（易答反）",
    "numeric":          "验证数字与时序事实的准确性",
    "paraphrase":       "验证同义表述下的检索召回能力",
    "insufficient_ctx": "验证上下文不足时是否会部分作答或说明局限",
    "context_conflict": "验证检索结果自相矛盾时能否识别冲突",
    "unanswerable":     "验证本质不可回答问题的识别",
    "prompt_injection": "验证抵抗诱导偏离与信息泄露的能力",
}


# ============================================================
# 数据结构
# ============================================================
@dataclass
class GroundTruth:
    """
    Ground Truth：事实型校验依据

    与 reference_answer 的区别：
      - reference_answer 用于人工参考和报告展示
      - required_facts / forbidden_facts 用于程序化判定
        原因：字符串匹配会因改写、同义、数字格式差异导致误判
    """
    reference_answer: Optional[str] = None
    acceptable_answers: List[str] = field(default_factory=list)
    required_facts: List[str] = field(default_factory=list)
    forbidden_facts: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalCase:
    """
    单条评测用例

    字段设计原则：
      - expected_behavior 显式声明，不从 category 推断
      - context 为空表示"该问题本就不该有上下文"（如不可回答）
      - difficulty 用于分层抽样与报告分组
    """
    id: str
    category: str
    question: str
    expected_behavior: str
    context: List[str] = field(default_factory=list)
    ground_truth: GroundTruth = field(default_factory=GroundTruth)
    difficulty: str = "medium"
    tags: List[str] = field(default_factory=list)
    # 缺陷注入标记：用于 mutation testing
    injected_defect: Optional[str] = None
    note: str = ""

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"未知 category: {self.category}")
        if self.expected_behavior not in EXPECTED_BEHAVIOR:
            raise ValueError(
                f"未知 expected_behavior: {self.expected_behavior}"
            )
        if self.difficulty not in DIFFICULTY:
            raise ValueError(f"未知 difficulty: {self.difficulty}")
        if not self.id:
            raise ValueError("用例必须有 id")

    @property
    def should_refuse(self) -> bool:
        return self.expected_behavior == "refuse"

    @property
    def purpose(self) -> str:
        return CATEGORY_PURPOSE.get(self.category, "")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ground_truth"] = self.ground_truth.to_dict()
        d["purpose"] = self.purpose
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalCase":
        gt = d.get("ground_truth") or {}
        if isinstance(gt, str):          # 兼容旧格式
            gt = {"reference_answer": gt}
        return cls(
            id=d["id"],
            category=d["category"],
            question=d["question"],
            expected_behavior=d.get("expected_behavior", "answer"),
            context=d.get("context") or [],
            ground_truth=GroundTruth(
                reference_answer=gt.get("reference_answer"),
                acceptable_answers=gt.get("acceptable_answers") or [],
                required_facts=gt.get("required_facts") or [],
                forbidden_facts=gt.get("forbidden_facts") or [],
                evidence=gt.get("evidence") or [],
            ),
            difficulty=d.get("difficulty", "medium"),
            tags=d.get("tags") or [],
            injected_defect=d.get("injected_defect"),
            note=d.get("note", ""),
        )


@dataclass
class Dataset:
    """评测数据集：多个层级的集合"""
    name: str
    tier: str                  # smoke / regression / full
    description: str = ""
    cases: List[EvalCase] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def by_category(self) -> Dict[str, List[EvalCase]]:
        out: Dict[str, List[EvalCase]] = {}
        for c in self.cases:
            out.setdefault(c.category, []).append(c)
        return out

    def get(self, case_id: str) -> Optional[EvalCase]:
        for c in self.cases:
            if c.id == case_id:
                return c
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "description": self.description,
            "case_count": len(self.cases),
            "category_distribution": {
                k: len(v) for k, v in self.by_category().items()
            },
            "cases": [c.to_dict() for c in self.cases],
        }

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Dataset":
        return cls(
            name=d["name"],
            tier=d["tier"],
            description=d.get("description", ""),
            cases=[EvalCase.from_dict(c) for c in d.get("cases", [])],
        )


# ============================================================
# 数据集校验
# ============================================================
def validate_dataset(ds: Dataset) -> List[str]:
    """
    校验数据集合法性与内部一致性

    返回问题列表（空表示合法）
    """
    issues: List[str] = []

    # --- ID 唯一性 ---
    ids = [c.id for c in ds.cases]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        issues.append(f"用例 id 重复：{sorted(dup)}")

    for c in ds.cases:
        # --- expected_behavior 与 category 的一致性 ---
        if c.category == "out_domain" and c.expected_behavior != "refuse":
            issues.append(
                f"[{c.id}] category=out_domain 但 expected_behavior="
                f"{c.expected_behavior}，域外场景应声明为 refuse"
            )
        if c.category == "unanswerable" and c.expected_behavior != "refuse":
            issues.append(
                f"[{c.id}] category=unanswerable 但 expected_behavior="
                f"{c.expected_behavior}，不可回答问题应为 refuse"
            )
        if c.category == "ambiguous" and c.expected_behavior not in ("clarify", "refuse"):
            issues.append(
                f"[{c.id}] category=ambiguous 建议 expected_behavior=clarify"
            )

        # --- 域内用例必须有参考答案 ---
        if c.expected_behavior == "answer":
            if not c.ground_truth.reference_answer:
                issues.append(f"[{c.id}] expected_behavior=answer 但缺少 reference_answer")
            if not c.ground_truth.required_facts and not c.ground_truth.acceptable_answers:
                issues.append(
                    f"[{c.id}] 仅有 reference_answer 不足以程序化判定正确性，"
                    f"建议补充 required_facts 或 acceptable_answers"
                )

        # --- 拒答用例不应有参考答案 ---
        if c.expected_behavior == "refuse" and c.ground_truth.reference_answer:
            issues.append(
                f"[{c.id}] expected_behavior=refuse 但提供了 reference_answer，"
                f"拒答场景不应有标准答案"
            )

        # --- required 与 forbidden 不得矛盾 ---
        overlap = set(c.ground_truth.required_facts) & set(c.ground_truth.forbidden_facts)
        if overlap:
            issues.append(
                f"[{c.id}] required_facts 与 forbidden_facts 矛盾：{overlap}"
            )

    return issues


if __name__ == "__main__":
    import sys
    print(__doc__)
