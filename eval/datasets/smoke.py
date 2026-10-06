"""
Smoke Dataset —— 快速回归集
---------------------------------
用途：CI 默认执行。规模小、速度快、覆盖关键失败模式。

设计原则
--------
每条用例必须代表一类**不可妥协的质量底线**，而不是数量堆积。
若某类用例需要真实 API 才能执行，应移至 regression 或 full 层级。

覆盖的 6 类关键失败模式
------------------------
1. 域内正确作答          —— 基本能力
2. 域外正确拒答          —— 幻觉抑制底线
3. 数字事实准确          —— 幻觉高发类型
4. 实体正确              —— 实体调换是幻觉中最隐蔽的类型
5. 上下文不足时的部分作答 —— 过度生成检测
6. 模糊问题要求澄清       —— 避免强行作答
"""

from eval.schemas.dataset import Dataset, EvalCase, GroundTruth

CASES = [
    # ---------- 1. 域内正确作答 ----------
    EvalCase(
        id="smoke_in_01",
        category="in_domain",
        question="什么是等价类划分？",
        expected_behavior="answer",
        difficulty="easy",
        tags=["基础", "检索"],
        ground_truth=GroundTruth(
            reference_answer="把输入域划分为若干互不相交的子集，每个子集内的输入具有相同预期结果。",
            required_facts=["子集", "预期结果"],
        ),
    ),
    EvalCase(
        id="smoke_in_02",
        category="in_domain",
        question="冒烟测试的目的是什么？",
        expected_behavior="answer",
        difficulty="easy",
        tags=["基础"],
        ground_truth=GroundTruth(
            reference_answer="验证核心功能是否可用，判断版本是否具备进入详细测试的条件。",
            required_facts=["核心功能", "详细测试"],
        ),
    ),

    # ---------- 2. 数字事实准确（幻觉高发） ----------
    EvalCase(
        id="smoke_num_01",
        category="numeric",
        question="边界值分析需要关注哪几个边界值？",
        expected_behavior="answer",
        difficulty="medium",
        tags=["数字", "边界值"],
        ground_truth=GroundTruth(
            reference_answer="最小值、最小值加一、最大值减一、最大值。",
            #数字型事实必须逐项校验：回答含"四个"但写错具体值仍算幻觉
            required_facts=["最小值", "最大值", "减一", "加一"],
            forbidden_facts=["中间值", "平均值"],
        ),
    ),

    # ---------- 3. 实体正确（实体调换检测） ----------
    EvalCase(
        id="smoke_ent_01",
        category="in_domain",
        question="自动化测试框架通常分为哪几层？",
        expected_behavior="answer",
        difficulty="medium",
        tags=["实体", "分层"],
        ground_truth=GroundTruth(
            reference_answer="通常分为基础层、用例层、数据层。",
            required_facts=["基础层", "用例层", "数据层"],
            # 实体调换检测：出现"接口层"说明模型编造了不存在的一层
            forbidden_facts=["接口层", "业务层", "页面层"],
        ),
    ),

    # ---------- 4. 域外正确拒答（幻觉抑制底线） ----------
    EvalCase(
        id="smoke_out_01",
        category="out_domain",
        question="Python 怎么安装？",
        expected_behavior="refuse",
        difficulty="easy",
        tags=["拒答", "域外"],
        ground_truth=GroundTruth(
            reference_answer=None,
            forbidden_facts=["pip install", "官网下载", "Anaconda"],
        ),
    ),
    EvalCase(
        id="smoke_out_02",
        category="out_domain",
        question="北京有哪些好吃的火锅店？",
        expected_behavior="refuse",
        difficulty="hard",
        tags=["拒答", "域外", "诱导"],
        # 这类问题模型极易"热情推荐"，是最典型的幻觉场景
        ground_truth=GroundTruth(
            reference_answer=None,
            forbidden_facts=["推荐", "海底捞", "呷哺"],
        ),
    ),

    # ---------- 5. 模糊问题应澄清 ----------
    EvalCase(
        id="smoke_amb_01",
        category="ambiguous",
        question="怎么测试？",
        expected_behavior="clarify",
        difficulty="medium",
        tags=["澄清", "模糊"],
        ground_truth=GroundTruth(
            reference_answer=None,
            forbidden_facts=["等价类", "边界值", "冒烟"],
        ),
    ),

    # ---------- 6. 不可回答问题 ----------
    EvalCase(
        id="smoke_una_01",
        category="unanswerable",
        question="你觉得 2026 年最好的编程语言是什么，为什么？",
        expected_behavior="refuse",
        difficulty="hard",
        tags=["拒答", "主观题"],
        ground_truth=GroundTruth(
            reference_answer=None,
            forbidden_facts=["Python 是", "Java 是", "最好的语言是"],
        ),
    ),
]


DATASET = Dataset(
    name="smoke",
    tier="smoke",
    description="CI 快速回归集：6 类关键失败模式，8 条用例",
    cases=CASES,
)
