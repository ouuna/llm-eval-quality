"""
Regression Dataset —— 历史缺陷回归集
---------------------------------
**本模块的价值不是"多一份数据"，而是把项目开发过程中的每一次
缺陷发现固化为可执行的资产。**

闭环
----
    Bug 发现 → 复现 → 固化用例 → 修复 → 回归测试 → CI 保护

收录范围
--------
Phase 0  项目审计发现的 4 个 P0 缺陷
Phase 3  重构时修复的 3 个方法论盲区
Phase 5  发现的 Judge 已知局限
Phase 6  验证阶段发现的 3 个评测器自身 bug
工程过程中遇到的 4 个环境问题

分类原则
--------
按缺陷归属分层，便于定位责任：
    evaluator_bug   评测器自身缺陷（最严重——评测器错了，指标全错）
    method_limit     方法论局限（已知但未修，需在报告标注）
    env_issue        运行环境问题（编码/换行/字符）
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


# ============================================================
# 缺陷定义
# ============================================================
@dataclass
class RegressionCase:
    """
    一条历史缺陷的回归用例

    与普通 EvalCase 的区别：
        - expected_verdict 关注"评测器是否正确判定"，而非"被测系统质量"
        - 额外记录 discovered_in（发现阶段）与 root_cause（根因）
    """
    id: str
    phase: str
    layer: str                       # evaluator_bug / method_limit / env_issue
    severity: str                    # P0 / P1 / P2

    # 复现所需的输入
    question: str
    answer: str
    context: str = ""
    required_facts: List[str] = field(default_factory=list)
    forbidden_facts: List[str] = field(default_factory=list)

    # 期望的判定结果
    should_detect_hallucination: bool = True

    # 由哪个指标验证该缺陷
    #   faithfulness  声明级验证（幻觉检测）
    #   correctness    正确性/完整性检测
    #   relevance      相关性检测
    #   mechanism      机制专项测试（如CRLF 切分、min_len 过滤）
    #   manual         需人工核验（如 Judge 自我偏好）
    verify_by: str = "faithfulness"

    # 缺陷档案
    discovered_in: str = ""
    root_cause: str = ""
    fix_summary: str = ""

    @property
    def is_fixed(self) -> bool:
        return not self.fix_summary.startswith("未修复")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "phase": self.phase, "layer": self.layer,
            "severity": self.severity,
            "question": self.question, "answer": self.answer,
            "context": self.context,
            "required_facts": self.required_facts,
            "forbidden_facts": self.forbidden_facts,
            "should_detect_hallucination": self.should_detect_hallucination,
            "verify_by": self.verify_by,
            "discovered_in": self.discovered_in,
            "root_cause": self.root_cause,
            "fix_summary": self.fix_summary,
            "is_fixed": self.is_fixed,
        }


CASES: List[RegressionCase] = [
    # ==========================================================
    # Phase 0 —— 审计发现的 4 个 P0
    # ==========================================================
    RegressionCase(
        id="REG-P0-001", phase="Phase 0 审计", layer="evaluator_bug",
        severity="P0",
        question="（空回答场景）", answer="",
        context="任意上下文内容",
        should_detect_hallucination=False,
        discovered_in="Phase 0 / 旧 evaluator.py:118",
        root_cause="check_hallucination 首行 `if not output.strip(): return True`，"
                   "把空输出判为'无幻觉'。API 失败 → 空串 → 误判通过。",
        fix_summary="引入 status 三态（passed/failed/error），error 不计入分母；"
                    "空输出且 should_refuse=False 时报'应作答但未作答'。",
    ),
    RegressionCase(
        id="REG-P0-002", phase="Phase 0 审计", layer="evaluator_bug",
        severity="P0", verify_by="correctness",
        question="自动化测试框架通常分为哪几层？",
        answer="自动化测试框架分为基础层、用例层。",
        context="自动化测试框架分层：基础层负责请求封装，用例层负责断言，数据层负责测试数据。",
        required_facts=["基础层", "用例层", "数据层"],
        discovered_in="Phase 0 / 旧 evaluator.py:322",
        root_cause="passed = no_halluc and is_complete，expected_output 从不参与判定。"
                   "只要用词在上下文出现过就算通过，'域内准确率 100%' 是假的。",
        fix_summary="新增 evaluate_correctness，以 required_facts 覆盖率判定正确性。",
    ),
    RegressionCase(
        id="REG-P0-003", phase="Phase 0 审计", layer="evaluator_bug",
        severity="P0", verify_by="relevance",
        question="如何提高代码质量？",
        answer="代码质量的提高方法。",
        context="提高代码质量需要遵循编码规范。",
        discovered_in="Phase 0 / 旧 evaluator.py:212",
        root_cause="relevance_score 算 |question∩answer| / |question|，"
                   "即'答案有没有抄问题里的字'。同义反复可得满分。",
        fix_summary="改为三级递降：required_facts → answer↔reference → answer↔context。",
    ),
    RegressionCase(
        id="REG-P0-004", phase="Phase 0 审计", layer="evaluator_bug",
        severity="P0",
        question="中国的首都是哪里？",
        answer="中国的首都是上海。",
        context="北京是中国的首都。",
        discovered_in="Phase 0",
        root_cause="字符覆盖率无法检出实体调换——'中国的首都是上海' 的实词"
                   "全部包含在 '北京是中国的首都' 中，覆盖率 1.0。",
        fix_summary="引入声明级验证：拆 Claim → 逐条验证；"
                    "支持 forbidden_facts 直接定位调换实体。",
    ),

    # ==========================================================
    # Phase 3 —— 重构时修复的方法论盲区
    # ==========================================================
    RegressionCase(
        id="REG-M01", phase="Phase 3 重构", layer="method_limit",
        severity="P1",
        question="自动化测试框架通常分为哪几层？",
        answer="自动化测试框架分为基础层、用例层、数据层、业务层。",
        context="自动化测试框架分层：基础层负责请求封装，用例层负责断言，数据层负责测试数据。",
        forbidden_facts=["业务层", "接口层", "页面层"],
        discovered_in="Phase 3 测试暴露",
        root_cause="split_claims 的 min_len=4 过滤掉了'业务层。'（仅 3 字），"
                   "并列枚举尾部短片段被丢弃 → forbidden_facts 永远命中不到。",
        fix_summary="min_len 降至 2；顿号纳入切分字符。",
    ),
    RegressionCase(
        id="REG-M02", phase="Phase 3 重构", layer="method_limit",
        severity="P1", verify_by="correctness",
        question="冒烟测试的目的是什么？",
        answer="冒烟测试是完整回归全部功能。",
        context="冒烟测试：验证核心功能是否可用。冒烟测试：完整回归全部功能。",
        forbidden_facts=["完整回归全部功能"],
        discovered_in="Phase 3 测试暴露",
        root_cause="冲突检测用'去掉实词后的骨架'比对，"
                   "而两条矛盾句骨架完全不同，检测不到。",
        fix_summary="改为'主体 + 谓述'分离：同主体出现 Jaccard<0.5 的谓述即判冲突。",
    ),
    RegressionCase(
        id="REG-M03", phase="Phase 3 重构", layer="method_limit",
        severity="P1",
        question="冒烟测试是必需环节吗？",
        answer="冒烟测试不是必需环节。",
        context="冒烟测试是必需环节，验证核心功能是否可用。",
        discovered_in="Phase 3 设计",
        root_cause="否定翻转：字面高度重合但极性相反，字符覆盖率无法区分。",
        fix_summary="新增 _has_negation_conflict 极性检测。",
    ),

    # ==========================================================
    # Phase 5 —— Judge 的已知局限
    # ==========================================================
    RegressionCase(
        id="REG-J01", phase="Phase 5", layer="method_limit",
        severity="P1", verify_by="manual",
        question="自动化测试框架通常分为哪几层？",
        answer="自动化测试框架分为基础层、用例层、报告层。",
        context="自动化测试框架分层：基础层负责请求封装，用例层负责断言，数据层负责测试数据。",
        forbidden_facts=["报告层"],
        discovered_in="Phase 5 实测",
        root_cause="LLM Judge 把'资料说数据层 vs 模型说报告层'理解为"
                   "'资料未提及报告层'（信息缺失）而非事实冲突。"
                   "实测 score=0.72 但 hallucination=False。",
        fix_summary="未修复——Judge 的语义边界。已写入测试固化，"
                    "作为'必须交叉验证'的依据。",
    ),
    RegressionCase(
        id="REG-J02", phase="Phase 5", layer="method_limit",
        severity="P2", verify_by="manual",
        question="（任意问题）", answer="（任意回答）",
        context="",
        should_detect_hallucination=False,
        discovered_in="Phase 5 设计",
        root_cause="Judge 与被测系统使用同一模型（glm-4-flash），"
                   "存在自我偏好风险，未做交叉验证。",
        fix_summary="未修复。Judge 模型已可配置化（JUDGE_MODEL_NAME），"
                    "预留交叉验证能力但未实际验证。",
    ),

    # ==========================================================
    # Phase 6 —— 验证阶段发现的 3 个评测器 bug
    # ==========================================================
    RegressionCase(
        id="REG-P6-001", phase="Phase 6 验证", layer="evaluator_bug",
        severity="P0",
        question="回归测试是谁提出的？",
        answer="回归测试由 Kent Beck 在 2003 年提出。",
        context="回归测试：修改代码后重新执行相关测试用例，确认没有引入新缺陷。",
        discovered_in="Phase 6 / Gold Set GOLD_H15",
        root_cause="_extract_terms 只提取 [一-龥]，'Kent Beck' 被完全丢弃，"
                   "覆盖率虚高至 0.44 → 判为 supported（漏报）。",
        fix_summary="额外提取拉丁词（人名/术语/产品名）与数字。",
    ),
    RegressionCase(
        id="REG-P6-002", phase="Phase 6 验证", layer="evaluator_bug",
        severity="P0",
        question="测试用例通常包含多少个字段？",
        answer="测试用例通常包含 15 个字段。",
        context="测试用例：用例编号、所属模块、用例标题、前置条件、操作步骤、"
                "预期结果、实际结果、执行结果、优先级。",
        forbidden_facts=["15"],
        discovered_in="Phase 6 / Gold Set GOLD_H07",
        root_cause="数字冲突检测要求 best_cov < supported_threshold 才触发。"
                   "此样本字面重合度高，跳过了数字检查。",
        fix_summary="改为只要出现上下文中不存在的数字即判定。",
    ),
    RegressionCase(
        id="REG-P6-003", phase="Phase 6 验证", layer="evaluator_bug",
        severity="P0",
        question="什么是等价类划分？",
        answer="今天北京天气晴朗，气温 25 度，适合户外运动。",
        context="等价类划分：把输入域划分为若干互不相交的子集。",
        discovered_in="Phase 6 / Gold Set GOLD_H14",
        root_cause="跑题的回答只报'缺必需事实'，不判幻觉。"
                   "required_facts 缺失只说明'没答到点上'，不等于'编造'。",
        fix_summary="新增跑题检测：所有声明与上下文零实词重合时判为幻觉。",
    ),

    # ==========================================================
    # 工程环境问题
    # ==========================================================
    RegressionCase(
        id="REG-E01", phase="开发过程", layer="env_issue",
        verify_by="mechanism", severity="P2",
        question="（CRLF 切分）", answer="任意答案",
        context="等价类划分：...\r\n\r\n边界值分析：...",
        discovered_in="项目初期",
        root_cause="load_knowledge 按 '\\n\\n' 切分，但 Windows 记事本保存为 CRLF，"
                   "切点对不上 → 整份文件被当成 1 条 → 检索恒为空。",
        fix_summary="切分前统一 replace('\\r\\n','\\n')；"
                    "并加 .gitattributes 强制 eol=lf。",
    ),
    RegressionCase(
        id="REG-E02", phase="开发过程", layer="env_issue",
        verify_by="mechanism", severity="P2",
        question="（零宽字符）", answer="任意答案",
        context="任意上下文",
        should_detect_hallucination=False,
        discovered_in="项目初期",
        root_cause="从聊天复制代码时混入零宽字符，if __name__=='__main__' 恒为假，"
                   "脚本无任何输出。",
        fix_summary="改用 ast.parse 定位；后续避免从聊天直接粘贴代码。",
    ),
    RegressionCase(
        id="REG-E03", phase="开发过程", layer="env_issue",
        verify_by="mechanism", severity="P1",
        question="什么是等价类划分？",
        answer="把输入域划分为若干互不相交的子集。",
        context="等价类划分：把输入域划分为若干互不相交的子集。",
        should_detect_hallucination=False,
        discovered_in="项目初期",
        root_cause="检索按单字匹配，'什么是等价类划分' 拆成 什/么/是/等/价/类…，"
                   "这些字在文档中不存在 → 所有文档 score=0 → 检索恒空。",
        fix_summary="改为语义词组切分 + 同义词扩展 + 词长加权，召回率 0 → 68%。",
    ),
    RegressionCase(
        id="REG-E04", phase="开发过程", layer="env_issue",
        verify_by="mechanism", severity="P1",
        question="（Prompt 占位符）", answer="任意答案",
        context="任意上下文",
        should_detect_hallucination=False,
        discovered_in="Phase 1",
        root_cause="config.yaml 的 Prompt 模板含 {domain} 占位符，"
                   "但代码只传 context/question → KeyError → pytest 全FAILED。",
        fix_summary="从模板移除该占位符。",
    ),
]


# ============================================================
# 统计
# ============================================================
def statistics() -> Dict[str, Any]:
    by_layer: Dict[str, int] = {}
    by_phase: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}

    for c in CASES:
        by_layer[c.layer] = by_layer.get(c.layer, 0) + 1
        by_phase[c.phase] = by_phase.get(c.phase, 0) + 1
        by_severity[c.severity] = by_severity.get(c.severity, 0) + 1

    return {
        "total": len(CASES),
        "by_layer": by_layer,
        "by_phase": by_phase,
        "by_severity": by_severity,
        "p0_count": sum(1 for c in CASES if c.severity == "P0"),
        "unfixed": sum(1 for c in CASES if not c.is_fixed),
    }


if __name__ == "__main__":
    st = statistics()
    print("=" * 66)
    print("Regression Dataset 统计")
    print("=" * 66)
    print(f"  总数       {st['total']}")
    print(f"  P0级       {st['p0_count']}")
    print(f"  未修复     {st['unfixed']}")
    print()
    print("  按缺陷层：")
    for k, v in sorted(st["by_layer"].items(), key=lambda x: -x[1]):
        print(f"    {k:20} {v}")
    print()
    print("  全部用例：")
    for c in CASES:
        flag = "已修复" if c.is_fixed else "未修复"
        print(f"    {c.id:14} {c.severity:4} {c.layer:15} {flag}")
