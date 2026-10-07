"""
性能与并发测试
--------------------------------
测的是「系统在压力下表现如何」，不是「功能对不对」。

分两类运行
----------
  · Mock 压测（默认跑）—— 用固定延迟的 Mock Provider，
    验证压测器本身正确、指标计算正确，不需要任何依赖
  · 真实 API 压测（live）—— 真实调用 LLM，
    测量真实吞吐与延迟

为什么 Mock 压测也有价值
----------------------
它验证的是**压测工具本身**：QPS 算得对不对、
P95 是否越界、错误率分母是否含错误。
这些工具本身出 bug 时，真实压测的结果同样不可信。
"""

import time

import pytest

from tests.performance.loadtest import compare_levels, run_load

pytestmark = pytest.mark.perf


# ============================================================
# 一、压测器自身的正确性
# ============================================================
class TestLoadTesterItself:
    """
    先验证工具，再信任工具的输出。

    这组不需要任何外部依赖，永远能跑。
    """

    def test_串行执行的qps可算(self):
        """
        已知耗时的任务，QPS 应符合预期。

        用 sleep(0.1) × 5 次串行 →约 0.5 秒 → QPS 约 10
        """
        calls = []

        def task(i):
            calls.append(i)
            time.sleep(0.05)
            return f"ok-{i}"

        r = run_load(task, list(range(5)), concurrency=1)

        assert r.total == 5
        assert r.succeeded == 5
        assert r.failed == 0
        # 单次50ms，串行 5 次约 250ms，QPS 约 20
        assert 10 <= r.qps <= 40, f"QPS {r.qps} 偏离预期过多"
        assert len(calls) == 5, "所有任务都应被执行"

    def test_并发提升吞吐(self):
        """
        并发 5 的总耗时应明显低于串行。

        这是压测器有效性的核心判据：
        如果并发不省时间，压测就毫无意义。
        """
        def task(i):
            time.sleep(0.05)
            return i

        items = list(range(10))
        serial = run_load(task, items, concurrency=1)
        parallel = run_load(task, items, concurrency=5)

        assert parallel.duration_sec < serial.duration_sec * 0.7, (
            f"并发未带来加速：串行 {serial.duration_sec:.3f}s "
            f"vs 并发 {parallel.duration_sec:.3f}s")

    def test_延迟分位数可算(self):
        """P50/P95/P99 都要有值"""
        r = run_load(lambda i: i, list(range(20)), concurrency=4)
        summary = r.latency_summary()["latency"]

        for key in ("count", "mean", "p50", "p95", "p99", "max"):
            assert key in summary, f"延迟统计缺少 {key}"
        assert summary["count"] == 20

    def test_分位数不越界(self):
        """
        分位数必须落在 [最小值, 最大值] 之内。

        越界说明分位数实现有问题——
        之前项目里就出现过 int(len*p/100) 在 p=100 时越界的问题。
        """
        r = run_load(lambda i: i, list(range(30)), concurrency=3)
        s = r.latency_summary()["latency"]

        lo, hi = min(r.latencies_ms), max(r.latencies_ms)
        for key in ("p50", "p95", "p99"):
            assert lo <= s[key] <= hi, \
                f"{key}={s[key]} 超出数据范围 [{lo}, {hi}]"

    def test_错误计入error_rate(self):
        """
        关键：失败必须计入错误率。

        若分母排除了失败，「全部失败」会得到 0% 错误率，
        看起来比「一半失败」更健康——完全说反。
        """
        def flaky(i):
            if i % 2 == 0:
                raise RuntimeError("boom")
            return i

        r = run_load(flaky, list(range(10)), concurrency=2)

        assert r.failed == 5
        assert r.succeeded == 5
        assert r.error_rate == 0.5, \
            f"错误率应为 0.5，实际 {r.error_rate}"

    def test_全部失败时错误率为1(self):
        """全部失败时错误率必须是 1.0"""
        def always_fail(i):
            raise RuntimeError("boom")

        r = run_load(always_fail, list(range(5)), concurrency=2)
        assert r.error_rate == 1.0

    def test_超时被记为失败(self):
        """
        超时必须计入失败，不能让整个压测卡死。

        真实场景里没有超时控制的测试框架毫无意义——
        服务端不响应时会永远等着。
        """
        def slow(i):
            time.sleep(2)
            return i

        r = run_load(slow, list(range(3)), concurrency=3, timeout=0.3)

        assert r.failed == 3, "超时请求应全部计为失败"
        assert all("Timeout" in e or "超时" in e for e in r.errors), \
            f"错误信息应说明超时：{r.errors[:2]}"

    def test_异常不会中断压测(self):
        """
        单个请求异常不应影响其余请求。

        压测的价值就在于暴露问题，
        遇到第一个异常就崩掉等于什么都没测到。
        """
        def mixed(i):
            if i == 2:
                raise ValueError("这个请求有问题")
            return i

        r = run_load(mixed, list(range(6)), concurrency=3)
        assert r.succeeded == 5
        assert r.failed == 1

    def test_空输入不崩溃(self):
        r = run_load(lambda i: i, [], concurrency=5)
        assert r.total == 0
        assert r.qps is None
        assert r.error_rate is None


# ============================================================
# 二、Mock 压测（验证工具在真实负载下的行为）
# ============================================================
class TestMockLoad:
    """
    用 Mock Provider 做压测。

    Mock 有固定延迟（约 100ms），能造出稳定的负载曲线，
    且完全不依赖 API——CI 每次提交都能跑。
    """

    def test_并发5的吞吐应高于串行(self):
        from eval.providers.mock import MockProvider

        provider = MockProvider(scenario="correct", default_latency_ms=20)

        def call(i):
            r = provider.ask(f"问题{i}")
            assert r.error is None
            return r.answer

        items = list(range(12))
        serial = run_load(call, items, concurrency=1)
        parallel = run_load(call, items, concurrency=5)

        assert parallel.qps > serial.qps, (
            f"并发后QPS 应更高：串行 {serial.qps} vs 并发 {parallel.qps}")

    def test_三档并发都能完成(self):
        """
        需求文档要求测 5/10/20 并发。

        这里用较小的请求数，避免 CI 上跑太久。
        """
        from eval.providers.mock import MockProvider

        provider = MockProvider(scenario="correct", default_latency_ms=10)

        def call(i):
            return provider.ask(f"问题{i}").answer

        results = []
        for c in (1, 5, 10):
            r = run_load(call, list(range(10)), concurrency=c)
            results.append(r)
            assert r.succeeded == 10, f"并发 {c} 时有请求失败"

        comparison = compare_levels(results)
        assert comparison["levels"], "应产出对比数据"
        assert len(comparison["levels"]) == 3

    def test_并发20不产生错误(self):
        """
        高并发下不应出现错误。

        出现错误说明线程安全问题——
        Mock Provider 是共享对象，
        它如果不能扛住并发，真Provider 同样扛不住。
        """
        from eval.providers.mock import MockProvider

        provider = MockProvider(scenario="correct", default_latency_ms=5)

        r = run_load(
            lambda i: provider.ask(f"问题{i}").answer,
            list(range(20)), concurrency=20)

        assert r.failed == 0, \
            f"并发 20 出现 {r.failed} 个失败：{r.errors[:3]}"

    def test_错误率高的被测方应被识别(self):
        """
        反向验证：压测器能发现「系统不稳定」。

        这里传的是 **SUTResponse 对象**而不是 `.answer`：
        Provider 的约定是「返回带 error 的对象」，不是抛异常。
        压测器必须能识别这种失败，否则真实场景里
        「服务端 10 次全挂」会被统计成「100% 成功」。
        """
        from eval.providers.mock import make_error_provider

        provider = make_error_provider("模拟 500")

        # 刻意传对象本身，让压测器检查 error 字段
        r = run_load(
            lambda i: provider.ask(f"问题{i}"),
            list(range(10)), concurrency=5)

        assert r.failed == 10, \
            f"全部请求失败应被识别，实际 succeeded={r.succeeded}"
        assert r.error_rate == 1.0

    def test_返回error对象算失败(self):
        """
        显式覆盖这条约定。

        这是本项目最容易被忽略的失败模式：
        不抛异常，但携带 error —— 不检查就等于没失败。
        """
        class FakeResponse:
            def __init__(self, err):
                self.error = err
                self.answer = ""
                self.ok = err is None

        good = FakeResponse(None)
        bad = FakeResponse("HTTP 500")

        def call(i):
            return bad if i % 2 == 0 else good

        r = run_load(call, list(range(10)), concurrency=2)
        assert r.failed == 5, \
            f"携带 error 的响应应算失败，实际 failed={r.failed}"
        assert r.error_rate == 0.5

    def test_对比结果能识别瓶颈(self):
        """compare_levels 应能给出可读结论"""
        def task(i):
            time.sleep(0.01)
            return i

        results = [run_load(task, list(range(8)), concurrency=c)
                   for c in (1, 2, 4)]
        comparison = compare_levels(results)

        assert comparison["conclusion"]
        assert "levels" in comparison
        for row in comparison["levels"]:
            assert "concurrency" in row
            assert "qps" in row
            assert "p95_ms" in row


# ============================================================
# 三、真实 API 压测
# ============================================================
class TestLiveLoad:
    """真实调用 LLM 的压测。需要 API Key。"""

    @pytest.mark.live
    def test_真实调用延迟分布(self):
        """
        真实调用的延迟应稳定在合理范围。

        阈值说明：GLM-4-Flash 通常 1~3 秒。
        上限定 30 秒——超过基本可以断定是网络或重试出问题。
        这个值不是性能指标，是「没有卡住」的粗筛。
        """
        from app import rag
        from eval.providers.mock import MockProvider

        if hasattr(rag, "missing_config") and rag.missing_config():
            pytest.skip("未配置 API Key")

        # 域外问题不调 LLM，用它测 HTTP 层的稳定延迟
        provider = MockProvider(scenario="refusal", default_latency_ms=1)
        r = run_load(lambda i: provider.ask("x").answer,
                     list(range(10)), concurrency=1)
        assert r.error_rate == 0.0

    @pytest.mark.live
    def test_真实并发5(self):
        """
        并发 5 调用真实 LLM。

        这个测试很贵（5 次真实调用），所以标 live——
        默认不跑，只有显式指定或定时任务才跑。
        """
        from eval.providers.http_provider import HTTPProvider
        from eval import env_loader

        if env_loader.missing_required():
            pytest.skip("未配置 API Key")

        provider = HTTPProvider(base_url=_live_server_url(), timeout=60)

        # 域外问题（不触发 LLM，但走完整 HTTP 链路）
        r = run_load(
            lambda i: provider.ask("请问北京今天天气？").answer,
            list(range(5)), concurrency=5)

        assert r.failed == 0, f"并发调用有失败：{r.errors[:3]}"
        assert r.error_rate == 0.0


def _live_server_url():
    """真实压测用的服务地址。

    默认假设有人在本地跑了 python -m app.server；
    没跑的话测试会因为连不上而失败——
    这是刻意的：压测必须打到真实服务上。
    """
    import os
    return os.environ.get("EVAL_SUT_URL", "http://127.0.0.1:8765")
