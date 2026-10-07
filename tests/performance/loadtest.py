"""
并发压测器
--------------------------------
用标准库 `concurrent.futures`，零第三方依赖。

为什么并发性能测试对 LLM 应用特别重要
------------------------------------
LLM 应用的延迟天然很高（秒级），串行跑几十条用例就是几分钟。
一旦真实场景变成「多个用户同时问」，
单机的串行处理能力立刻成为瓶颈——而这个问题
在串行测试里完全看不出来。

所以必须测并发下的三件事：
  1. 吞吐（QPS）——并发上去，吞吐是否线性增长
  2. 延迟劣化——并发 20 时的 P95 相对并发 1 劣化多少
  3. 错误率——并发是否会引入新的失败

第 3 点最容易被忽略。压测时错误率飙升，
说明系统在并发下不稳定，这个「稳定」比「快」更重要。
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from eval import metrics as stats


@dataclass
class LoadResult:
    """一次压测的完整结果"""

    concurrency: int
    total: int
    succeeded: int
    failed: int
    duration_sec: float
    latencies_ms: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    payload_bytes: int = 0

    @property
    def qps(self) -> Optional[float]:
        """每秒完成请求数"""
        if self.duration_sec <= 0:
            return None
        return round(self.total / self.duration_sec, 2)

    @property
    def success_rate(self) -> Optional[float]:
        return stats.rate(self.succeeded, self.total)

    @property
    def error_rate(self) -> Optional[float]:
        """
        错误率。分母含错误本身。

        与 pass_rate 的分母不同：
        pass_rate 排除 error（因为「没测成」不等于「没通过」），
        但压测里我们关心的是「有多少请求没成功」，
        分母必须含错误——否则「全部超时」会得到 0错误率。
        """
        return stats.error_rate(self.failed, self.total)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "concurrency": self.concurrency,
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "duration_sec": round(self.duration_sec, 3),
            "qps": self.qps,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
        }
        d.update(self.latency_summary())
        if self.errors:
            # 只保留前若干条错误，避免报告被刷屏
            seen = []
            for e in self.errors:
                head = e.split(":")[0][:80]
                if head not in seen:
                    seen.append(head)
            d["error_samples"] = seen[:5]
        return d

    def latency_summary(self) -> Dict[str, Any]:
        """
        延迟分位数，统一走 eval.metrics。

        这里**不做二次舍入**——
        describe 内部已按数据量级自适应精度，
        再round(., 1) 会把亚毫秒级的值压成 0，
        统计结果失真却看起来「很整齐」。
        """
        if not self.latencies_ms:
            return {"latency": {"count": 0}}

        s = stats.describe(self.latencies_ms)
        # 压测场景最关心这几个分位数
        for p in (50, 90, 95, 99):
            v = stats.percentile(self.latencies_ms, p)
            if v is not None:
                s[f"p{p}"] = v
        return {"latency": s}


def run_load(fn: Callable[[Any], Any],
             items: List[Any],
             concurrency: int = 1,
             timeout: Optional[float] = None) -> LoadResult:
    """
    对 fn 施加并发压力。

    参数
    ----
    fn           可调用对象，返回值不关心，成功即算成功
    items        要处理的数据列表
    concurrency  并发度
    timeout      单个请求的超时（秒），None 表示不设

    设计取舍
    --------
    · 用线程池而非进程池：LLM 调用是 IO 密集的，
      GIL 不是瓶颈；进程池还会带来序列化开销。
    · 超时在**调用侧**判定，不依赖被测方自觉：
      真实场景里被测服务不会替你做超时控制。
    """
    if not items:
        return LoadResult(concurrency=concurrency, total=0,
                          succeeded=0, failed=0, duration_sec=0.0)

    latencies: List[float] = []
    errors: List[str] = []
    payload_bytes = 0
    succeeded = 0

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {}
        for item in items:
            f = pool.submit(_timed_call, fn, item, timeout)
            futures[f] = item

        for future in as_completed(futures, timeout=None):
            ok, elapsed_ms, size, err = future.result()
            latencies.append(elapsed_ms)
            payload_bytes += size
            if ok:
                succeeded += 1
            else:
                errors.append(err)

    duration = time.perf_counter() - start

    return LoadResult(
        concurrency=concurrency,
        total=len(items),
        succeeded=succeeded,
        failed=len(items) - succeeded,
        duration_sec=duration,
        latencies_ms=latencies,
        errors=errors,
        payload_bytes=payload_bytes,
    )


def _timed_call(fn, item, timeout):
    """
    执行单次调用并计时。

    失败判定有两种，不能只看异常
    ----------------------------
    1. **抛出异常** —— 网络错误、超时、代码 bug
    2. **返回带 error 的对象** —— 本项目 Provider 的约定

    第2 点极易被忽略：
    ``make_error_provider("模拟 500").ask(q)`` 不会抛异常，
    它返回一个 ``SUTResponse(error="模拟 500", answer="")``。
    压测器若只catch 异常，就会把「10 次全部失败」统计成
    **100% 成功**——因为确实没抛出东西。

    这不是假设性的风险：HTTP 服务返回 500 时，
    HTTPProvider 正是这样把错误包成 SUTResponse.error 返回的。
    真实压测里「服务端全挂」与「全部成功」显示成一回事，
    那这个压测就没有意义了。
    """
    t0 = time.perf_counter()
    try:
        if timeout:
            with ThreadPoolExecutor(max_workers=1) as sub:
                # 不用 future.result(timeout)，
                # 那样超时后线程仍在跑，会污染后续统计
                fut = sub.submit(fn, item)
                result = fut.result(timeout=timeout)
        else:
            result = fn(item)

        elapsed = (time.perf_counter() - t0) * 1000

        # ---- 关键：检查返回值是否携带错误 ----
        if _carries_error(result):
            return False, elapsed, 0, f"ProviderError: {result.error}"

        size = len(str(result).encode("utf-8")) if result is not None else 0
        return True, elapsed, size, ""

    except TimeoutError:
        elapsed = (time.perf_counter() - t0) * 1000
        return False, elapsed, 0, f"TimeoutError: 超过 {timeout}s"
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return False, elapsed, 0, f"{type(e).__name__}: {e}"


def _carries_error(result) -> bool:
    """
    判断返回值是否表示「调用失败」。

    支持两种形态：
      · SUTResponse（有 error 字段，ok 属性）
      · 任何带 error 属性的对象

    刻意不用 isinstance 判断具体类型——
    压测器应该对任何 Provider 实现通用，
    而不只认识 SUTResponse。
    """
    if result is None:
        # 没返回任何东西，视为失败
        return True

    err = getattr(result, "error", None)
    if err:
        return True

    # 有 ok 属性时一并检查
    ok = getattr(result, "ok", None)
    if ok is False:
        return True

    return False


def compare_levels(results: List[LoadResult]) -> Dict[str, Any]:
    """
    对比不同并发度的表现，找出瓶颈出现的位置。

    关注点不是「QPS 最高是多少」，而是**QPS 何时不再增长**——
    那就是系统的并发瓶颈点。
    """
    if not results:
        return {}

    baseline = results[0]
    base_p95 = stats.percentile(baseline.latencies_ms, 95)
    base_qps = baseline.qps or 0.0

    rows = []
    for r in results:
        p95 = stats.percentile(r.latencies_ms, 95)
        rows.append({
            "concurrency": r.concurrency,
            "qps": r.qps,
            "p95_ms": round(p95, 1) if p95 else None,
            "error_rate": r.error_rate,
            # 相对并发1 的吞吐增益。线性增长时接近并发倍数
            "throughput_gain": (round(r.qps / base_qps, 2)
                                if base_qps and r.qps else None),
            # 相对并发 1 的延迟劣化倍数
            "latency_degradation": (round(p95 / base_p95, 2)
                                    if base_p95 and p95 else None),
        })

    # 找出 QPS 增益明显小于并发倍数的第一个点
    bottleneck = None
    for row in rows:
        if row["throughput_gain"] is None:
            continue
        if row["throughput_gain"] < row["concurrency"] * 0.6:
            bottleneck = row["concurrency"]
            break

    return {
        "levels": rows,
        "bottleneck_concurrency": bottleneck,
        "conclusion": _conclude(rows, bottleneck),
    }


def _conclude(rows, bottleneck) -> str:
    """给出一句人能读懂的结论"""
    if not rows:
        return "无数据"

    if bottleneck is None:
        return ("QPS 随并发近似线性增长，"
                "在测试范围内未观察到明显瓶颈")

    last = rows[-1]
    return (f"并发 {bottleneck} 起吞吐增益明显落后于并发倍数，"
            f"提示存在并发瓶颈（该点错误率 {last['error_rate']}）")


# ============================================================
# CLI 入口：python -m tests.performance.loadtest
# ============================================================
def _demo():
    """用 Mock Provider 跑一次压测，演示工具输出格式"""
    from eval.providers.mock import MockProvider

    provider = MockProvider(scenario="correct", default_latency_ms=20)

    def call(i):
        return provider.ask(f"性能测试问题 {i}")

    print("=" * 66)
    print("并发压测（Mock Provider，模拟 20ms 延迟）")
    print("=" * 66)
    print()
    print("需求文档要求测5/10/20 并发，这里用 1/5/10 三档做演示。")
    print()

    results = []
    for c in (1, 5, 10):
        r = run_load(call, list(range(20)), concurrency=c)
        results.append(r)

        s = r.latency_summary()["latency"]
        print(f"并发 {r.concurrency:>2}  "
              f"QPS {str(r.qps):>7}  "
              f"平均 {s.get('mean', 0):>7.1f}ms  "
              f"P95 {s.get('p95', 0):>8.1f}ms  "
              f"错误率 {r.error_rate}")

    print()
    print("=" * 66)
    print("并发对比")
    print("=" * 66)
    for row in compare_levels(results)["levels"]:
        print(f"  并发 {row['concurrency']:>2}  "
              f"吞吐增益 {row['throughput_gain']}x  "
              f"延迟劣化 {row['latency_degradation']}x")

    print()
    print(compare_levels(results)["conclusion"])
    print()
    print("说明：Mock 的延迟是固定的，"
          "所以延迟劣化主要反映排队情况，不代表真实模型表现。")
    print("      真实压测见 pytest -m live tests/performance/")

    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
