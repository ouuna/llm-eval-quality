"""
Mock Provider —— 故障注入
---------------------------------
让整个评测系统能在**无真实 API** 的条件下运行与验证。

解决什么问题
------------
1. 单元测试不应依赖网络与 API Key
   —— 否则每次跑测试都要花钱、都要网络、CI 也要配密钥
2. 评测器本身无法被充分测试
   —— "当被测对象返回这种答案时，评测器会怎么判？"
      这类问题必须能离线构造
3. 异常路径无法覆盖
   —— 超时、非法JSON、API 错误在真实环境难以复现

可构造的故障类型
----------------
    correct           正确答案
    wrong             错误答案（内容不相关）
    hallucination     幻觉答案（编造实体/数字）
    entity_swap       实体调换（最难检出的一类）
    refusal           正确拒答
    empty             空回答
    timeout           超时
    api_error         API 报错
    invalid_json      返回非 JSON（用于测 Judge）
    context_conflict  上下文冲突

设计要点
--------
**故障注入必须可控且可断言**：每个 scenario 都能预测评测器应该
给出什么判定，测试才能写成断言而非日志比对。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import json
import time

from eval.providers.sut import SUTResponse, ProviderError, ProviderNotConfigured
from eval.schemas.result import TokenUsage


# ============================================================
# 知识库（供 Mock 返回上下文用）
# ============================================================
DEFAULT_KNOWLEDGE = {
    "layers": "自动化测试框架分层：基础层负责请求封装，用例层负责断言，数据层负责测试数据。",
    "smoke": "冒烟测试：验证核心功能是否可用，判断版本是否具备进入详细测试的条件。",
    "regression": "回归测试：修改代码后重新执行相关测试用例，确认没有引入新缺陷。",
    "equivalence": "等价类划分：把输入域划分为若干互不相交的子集，每个子集内的输入具有相同预期结果。",
    "boundary": "边界值分析：关注输入和输出边界，常用边界值包括最小值、最小值加一、最大值减一、最大值。",
}


# ============================================================
# 场景定义
# ============================================================
@dataclass
class Scenario:
    """一个预设场景：给定问题，返回预设回答"""
    name: str
    answer: str
    contexts: List[str] = field(default_factory=list)
    error: Optional[str] = None
    # 该场景下评测器"应该"给出的判定，供测试断言
    expected_hallucination: Optional[bool] = None
    note: str = ""


SCENARIOS: Dict[str, Scenario] = {
    "correct": Scenario(
        name="correct",
        answer="自动化测试框架分为基础层、用例层、数据层。",
        contexts=[DEFAULT_KNOWLEDGE["layers"]],
        expected_hallucination=False,
        note="完全正确",
    ),
    "paraphrase": Scenario(
        name="paraphrase",
        answer="框架包含基础层、用例层与数据层这三个层次。",
        contexts=[DEFAULT_KNOWLEDGE["layers"]],
        expected_hallucination=False,
        note="同义改写，事实正确",
    ),
    "wrong": Scenario(
        name="wrong",
        answer="今天北京天气晴朗，气温 25 度，适合户外运动。",
        contexts=[DEFAULT_KNOWLEDGE["layers"]],
        expected_hallucination=True,
        note="完全跑题",
    ),
    "hallucination": Scenario(
        name="hallucination",
        answer="回归测试由 Kent Beck 在 2003 年提出。",
        contexts=[DEFAULT_KNOWLEDGE["regression"]],
        expected_hallucination=True,
        note="编造人名与年份（Phase 6 曾漏报）",
    ),
    "entity_swap": Scenario(
        name="entity_swap",
        answer="自动化测试框架分为基础层、用例层、报告层。",
        contexts=[DEFAULT_KNOWLEDGE["layers"]],
        expected_hallucination=True,
        note="实体调换：数据层→报告层（Phase 0 曾漏报）",
    ),
    "numeric_error": Scenario(
        name="numeric_error",
        # 必须含上下文中不存在的数字才能触发数字冲突检测。
        # 纯词汇编造（如"中间值"）属词级盲区，需 forbidden_facts 辅助。
        answer="边界值分析关注的边界值共 3 个：最小值、最大值、平均值。",
        contexts=[DEFAULT_KNOWLEDGE["boundary"]],
        expected_hallucination=True,
        note="编造数字 3 与'平均值'（Phase 6 曾漏报数字编造）",
    ),
    "refusal": Scenario(
        name="refusal",
        answer="参考资料中未提及。",
        contexts=[],
        expected_hallucination=False,
        note="正确拒答（显式）",
    ),
    "empty": Scenario(
        name="empty",
        answer="",
        contexts=[],
        expected_hallucination=False,
        note="空回答（该答却没答）",
    ),
    "timeout": Scenario(
        name="timeout",
        answer="",
        contexts=[DEFAULT_KNOWLEDGE["layers"]],
        error="模拟超时：请求超过 60 秒",
        expected_hallucination=None,
        note="超时异常——必须标记为 error 而非通过",
    ),
    "api_error": Scenario(
        name="api_error",
        answer="",
        contexts=[],
        error="HTTP 502 Bad Gateway",
        expected_hallucination=None,
        note="API 报错——必须标记为 error",
    ),
    "invalid_json": Scenario(
        name="invalid_json",
        answer="{这不是合法的 JSON",
        contexts=[DEFAULT_KNOWLEDGE["layers"]],
        # 该场景供 Judge 测试解析/重试，声明级验证不涉及
        expected_hallucination=None,
        note="非法 JSON——供 Judge 解析与重试测试使用",
    ),
    "context_conflict": Scenario(
        name="context_conflict",
        answer="冒烟测试是完整回归全部功能。",
        contexts=[DEFAULT_KNOWLEDGE["smoke"],
                 "冒烟测试：完整回归全部功能。"],
        # 注意：声明级验证不检测上下文冲突，故预期为"无幻觉"。
        # 冲突检出由 correctness.detect_context_conflict 负责。
        expected_hallucination=False,
        note="上下文冲突但模型盲选；需冲突检测器而非幻觉检测",
    ),
}


# ============================================================
# Mock Provider
# ============================================================
class MockProvider:
    """
    故障注入用Provider

    三种模式
    --------
    fixed     固定返回一个场景
    sequence  按顺序轮换多个场景（用于测试多用例流程）
    callable  自定义函数（用于构造边界情况）
    """

    name = "mock"

    def __init__(self, scenario: str = "correct",
                 sequence: List[str] = None,
                 responder: Callable[[str], Scenario] = None,
                 default_latency_ms: int = 100,
                 knowledge: Dict[str, str] = None):
        if sequence and responder:
            raise ValueError("sequence 与 responder 不能同时指定")

        self.default_latency_ms = default_latency_ms
        self.knowledge = knowledge or dict(DEFAULT_KNOWLEDGE)
        self._call_count = 0

        if responder:
            self._mode = "callable"
            self._responder = responder
        elif sequence:
            self._mode = "sequence"
            self._sequence = list(sequence)
            self._seq_index = 0
        else:
            self._mode = "fixed"
            if scenario not in SCENARIOS:
                raise KeyError(
                    f"未知场景：{scenario}，"
                    f"可用：{sorted(SCENARIOS)}"
                )
            self._scenario_name = scenario

    # ---------- Provider 接口 ----------
    def is_available(self) -> bool:
        """Mock 永远可用——这正是它的价值"""
        return True

    def availability(self) -> Dict[str, Any]:
        return {"available": True, "reason": "",
                "mode": self._mode, "provider": "mock"}

    def ask(self, question: str) -> SUTResponse:
        scenario = self._pick(question)
        self._call_count += 1

        # 模拟延迟
        if self.default_latency_ms:
            time.sleep(min(self.default_latency_ms, 100) / 1000.0)

        return SUTResponse(
            answer=scenario.answer,
            contexts=list(scenario.contexts),
            latency_ms=self.default_latency_ms,
            token_usage=self._mock_usage(scenario.answer),
            error=scenario.error,
            raw={"scenario": scenario.name, "note": scenario.note},
        )

    def _mock_usage(self, answer: str) -> TokenUsage:
        """
        模拟 token 用量

        注意：返回 available=True 是"模拟真实 API 的情况"，
        用于验证 token 统计逻辑。真实不可用时用
        TokenUsage.unavailable()。
        """
        if not answer:
            return TokenUsage.unavailable()
        it = len(answer) * 2
        ot = len(answer)
        return TokenUsage.from_api({
            "prompt_tokens": it, "completion_tokens": ot,
            "total_tokens": it + ot,
        })

    def _pick(self, question: str) -> Scenario:
        if self._mode == "fixed":
            return SCENARIOS[self._scenario_name]
        if self._mode == "sequence":
            name = self._sequence[self._seq_index % len(self._sequence)]
            self._seq_index += 1
            return SCENARIOS[name]
        # callable
        return self._responder(question)

    # ---------- 便于测试的工具 ----------
    @property
    def call_count(self) -> int:
        return self.call_count_of_calls()

    def call_count_of_calls(self) -> int:
        return self._call_count

    @property
    def expected_verdict(self) -> Optional[bool]:
        """当前模式下所有场景的预期判定（用于断言）"""
        if self._mode == "fixed":
            return SCENARIOS[self._scenario_name].expected_hallucination
        if self._mode == "sequence":
            vals = [SCENARIOS[n].expected_hallucination for n in self._sequence]
            return None if any(v is None for v in vals) else all(vals)
        return None


def register_all():
    """把 MockProvider 注册到 Provider 注册表"""
    from eval.providers.sut import register_provider
    register_provider("mock", MockProvider)
    return MockProvider


# ============================================================
# 故障注入助手
# ============================================================
def scenario_provider(scenario: str) -> MockProvider:
    """便捷构造：固定返回一个预设场景"""
    return MockProvider(scenario=scenario)


def make_error_provider(error: str) -> MockProvider:
    """便捷构造：始终返回指定错误"""
    return MockProvider(responder=lambda q: Scenario(
        name="custom_error", answer="", contexts=[], error=error,
        expected_hallucination=None,
    ))


def make_sequence_provider(*scenarios: str) -> MockProvider:
    """便捷构造：按顺序轮换多个场景"""
    return MockProvider(sequence=list(scenarios))


def make_evaluator_scenario_provider(
        correct_answer: str,
        hallucinated_answer: str,
        context: str) -> MockProvider:
    """
    便捷构造：同一问题下交替返回正确与幻觉答案

    用途：测试稳定性检测能否发现"同一问题答案不一致"
    """
    answers = [correct_answer, hallucinated_answer]
    idx = {"i": 0}

    def responder(q):
        s = Scenario(
            name="alt",
            answer=answers[idx["i"] % 2],
            contexts=[context],
            expected_hallucination=(idx["i"] % 2 == 1),
        )
        idx["i"] += 1
        return s

    return MockProvider(responder=responder)


if __name__ == "__main__":
    print("=" * 60)
    print("Mock Provider 可用场景")
    print("=" * 60)
    for name, sc in SCENARIOS.items():
        exp = {True: "有幻觉", False: "无幻觉", None: "error"}[sc.expected_hallucination]
        print(f"  {name:16} 预期={exp:6} {sc.note}")

    print()
    p = MockProvider(scenario="entity_swap")
    r = p.ask("自动化测试框架通常分为哪几层？")
    print("Mock 返回示例：")
    print(f"  场景     {r.raw['scenario']}")
    print(f"  回答     {r.answer}")
    print(f"  上下文   {r.contexts[0][:50]}...")
    print(f"  延迟     {r.latency_ms}ms")
    print(f"  token    {r.token_usage.total_tokens}")
    print(f"  error    {r.error}")
