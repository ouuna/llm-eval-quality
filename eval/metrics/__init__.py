"""
统计度量层
--------------------------------
为什么需要这一层
----------------
改之前，统计逻辑散落在三处，而且**口径不一致**：

  · schemas/result.py 的 latency_stats 用 round((p/100)*(n-1)) 取分位
  · runner.py 里的 P95 用 int(len*p/100) 取分位
  · 稳定性测试里的方差在函数内部临时算

同一个「P95」在报告里和在门禁里可能是两个数。
这类问题在数据量小的时候根本看不出来，
等发现时已经错了很多次报告。

所以把所有统计集中到这里，各处只调用不自己实现。

设计原则
--------
1. **空输入返回 None，不返回 0**。
   0 是一个合法的统计值，表示「真的是 0」。
   把「没有数据」也报成 0，会让门禁误判——
   早期项目里 MetricsSummary 就是这么错的：
   指标不可用时填 0.0 配 available=False，看起来没问题，
   但任何忘记检查 available 的下游都会把 0 当真值用。
2. **分位数用统一算法**，避免各处实现不一致。
3. 全部零依赖，与项目其他地方保持一致。
"""

import math
from typing import List, Optional, Sequence, Dict, Any


# ============================================================
# 基础统计
# ============================================================
def _clean(values: Sequence[float]) -> List[float]:
    """滤掉 None 与 NaN；非有限数值不该进统计"""
    out = []
    for v in values or []:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f) or math.isinf(f):
            continue
        out.append(f)
    return out


def mean(values: Sequence[float]) -> Optional[float]:
    """算术平均。无有效数据返回 None"""
    vals = _clean(values)
    return sum(vals) / len(vals) if vals else None


def median(values: Sequence[float]) -> Optional[float]:
    """
    中位数。

    偶数个样本取中间两个的平均——
    这与「取下中位数」不同，后者会系统性低估。
    监控指标算中位数时这个区别很要紧：
    取错会让告警提前触发。
    """
    vals = sorted(_clean(values))
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def min_value(values: Sequence[float]) -> Optional[float]:
    vals = _clean(values)
    return min(vals) if vals else None


def max_value(values: Sequence[float]) -> Optional[float]:
    vals = _clean(values)
    return max(vals) if vals else None


def count(values: Sequence[float]) -> int:
    return len(_clean(values))


def variance(values: Sequence[float], sample: bool = False) -> Optional[float]:
    """
    方差。

    sample=True 时按样本方差（除以 n-1）计算。
    稳定性测试里关心的是「模型输出有多跳」，
    那是总体性质还是样本性质要看场景——
    但**默认用总体方差**（除以 n），因为
    我们的重复次数通常只有 3，样本方差会明显放大噪声。
    """
    vals = _clean(values)
    n = len(vals)
    if n == 0 or (sample and n < 2):
        return None
    m = sum(vals) / n
    return sum((v - m) ** 2 for v in vals) / (n - 1 if sample else n)


def stdev(values: Sequence[float], sample: bool = False) -> Optional[float]:
    """标准差"""
    var = variance(values, sample=sample)
    return math.sqrt(var) if var is not None else None


# ============================================================
# 分位数
# ============================================================
def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """
    第 p 百分位，p ∈ [0, 100]。

    算法：线性插值法（与 numpy 默认的 linear 一致）。

    之前项目里有两套实现：
      · round((p/100)*(n-1))  —— 索引法，不插值
      · int(len*p/100)        —— 索引法，p=100 时会越界

    两者在小样本下结果不同。例如 [1,2,3,4,5] 的 P95：
      插值法 → 4.8
      索引法 → 5
    差距在小样本下相当可观，而稳定性测试恰恰只有 3~5 个样本。

    这里用插值法，并把边界处理好。
    """
    vals = sorted(_clean(values))
    n = len(vals)
    if n == 0:
        return None
    if n == 1:
        return vals[0]

    p = max(0.0, min(100.0, float(p)))
    if p <= 0:
        return vals[0]
    if p >= 100:
        return vals[-1]

    # 位置 = (n-1) * p/100，在相邻两个样本间线性插值
    pos = (n - 1) * p / 100.0
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))

    if lower == upper:
        return vals[lower]

    frac = pos - lower
    return vals[lower] * (1 - frac) + vals[upper] * frac


def p50(values: Sequence[float]) -> Optional[float]:
    return percentile(values, 50)


def p90(values: Sequence[float]) -> Optional[float]:
    return percentile(values, 90)


def p95(values: Sequence[float]) -> Optional[float]:
    return percentile(values, 95)


def p99(values: Sequence[float]) -> Optional[float]:
    return percentile(values, 99)


# ============================================================
# 质量统计
# ============================================================
def pass_rate(passed: int, total: int) -> Optional[float]:
    """通过率。total=0 时返回 None（不是 0）"""
    if not total:
        return None
    return passed / total


def error_rate(errors: int, total: int) -> Optional[float]:
    """
    错误率。分母含错误本身。

    注意与 pass_rate 的分母区别：
    pass_rate 排除 error（error 不算失败也不算通过），
    error_rate 则必须把 error 算进分母——
    否则「全部调用失败」会得到 0 错误率，
    看起来比「一半失败」更健康，完全说反。
    """
    if not total:
        return None
    return errors / total


def rate(numerator: int, denominator: int) -> Optional[float]:
    """通用比率，分母为 0 时返回 None"""
    if not denominator:
        return None
    return numerator / denominator


# ============================================================
# 汇总
# ============================================================
def describe(values: Sequence[float], percentiles=(50, 90, 95, 99),
             precision: Optional[int] = None) -> Dict[str, Any]:
    """
    一次性算出常用统计量，供报告直接渲染。

    返回的字典里**没有值的项不出现在字典中**，
    而不是填 0 或 None——这样调用方
    `if "p99" in stats` 就能判断该指标是否可用，
    不会把「没测到」误当成「测出来是 0」。

    precision
    ---------
    留空时**按数据量级自适应**，而不是固定小数位。

    为什么不能固定
    --------------
    固定 precision=1 时，一组亚毫秒级的延迟
    （比如本地压测的 0.0001~ 0.0007ms）会被 round(., 1)
    全部压成 0.0，于是 min=max=mean=p99=0——
    统计结果彻底失真，而且看起来「很整齐」，不检查就会信。

    自适应的做法：按最大值的量级决定小数位。
    这样无论数据在 0.0001 还是 100000 都能保住有效数字。
    """
    vals = _clean(values)
    out: Dict[str, Any] = {"count": len(vals)}

    if not vals:
        return out

    if precision is None:
        precision = _auto_precision(vals)

    out["mean"] = round(sum(vals) / len(vals), precision)
    out["min"] = round(min(vals), precision)
    out["max"] = round(max(vals), precision)
    out["median"] = round(median(vals), precision)
    sd = stdev(vals)
    out["stdev"] = round(sd, precision) if sd is not None else None

    for p in percentiles:
        val = percentile(vals, p)
        if val is not None:
            out[f"p{p}"] = round(val, precision)

    return out


def _auto_precision(values: Sequence[float],
                    min_digits: int = 1,
                    max_digits: int = 6) -> int:
    """
    根据数据量级推断合适的小数位。

    规则：保证最大值至少保留 min_digits 位有效数字。
        magnitude = floor(log10(max(|v|)))
        precision = clamp(min_digits - magnitude, 0, max_digits)
    """
    abs_vals = [abs(v) for v in values if v != 0]
    if not abs_vals:
        return min_digits

    peak = max(abs_vals)
    magnitude = math.floor(math.log10(peak)) if peak > 0 else 0
    return max(0, min(max_digits, min_digits - magnitude))


def stability_stats(values: Sequence[float],
                    precision: int = 4) -> Dict[str, Any]:
    """
    稳定性测试专用汇总。

    比 describe 多了极差（max - min）——
    LLM 输出的稳定性最直观的体现就是极差：
    同一问题重复 3 次，最高和最低差多少。
    """
    out = describe(values, percentiles=(50, 95), precision=precision)
    vals = _clean(values)
    if vals:
        out["range"] = round(max(vals) - min(vals), precision)
    return out


def distribution(values: Sequence[float], buckets: int = 5,
                 precision: int = 4) -> List[Dict[str, Any]]:
    """
    直方图数据，供报告画分布图。

    buckets 过多会让每个桶样本太少，反而看不出趋势；
    默认 5 桶是「能看出形状」的最小值。
    """
    vals = _clean(values)
    if not vals or buckets < 1:
        return []

    lo, hi = min(vals), max(vals)
    if lo == hi:
        return [{"lower": lo, "upper": hi, "count": len(vals)}]

    width = (hi - lo) / buckets
    out = []
    for i in range(buckets):
        lower = lo + width * i
        upper = lo + width * (i + 1)
        # 最后一个桶闭区间，确保最大值被计入
        if i == buckets - 1:
            n = sum(1 for v in vals if lower <= v <= upper)
        else:
            n = sum(1 for v in vals if lower <= v < upper)
        out.append({
            "lower": round(lower, precision),
            "upper": round(upper, precision),
            "count": n,
        })
    return out


if __name__ == "__main__":
    sample = [120, 350, 800, 1200, 1500, 1500, 3200, 2100]

    print("=" * 60)
    print("统计层自检")
    print("=" * 60)
    print(f"样本           = {sample}")
    print()
    print("describe():")
    for k, v in describe(sample).items():
        print(f"  {k:8} = {v}")
    print()
    print("空输入的行为（应全部为 None，而非 0）：")
    print(f"  mean([])      = {mean([])}")
    print(f"  median([])    = {median([])}")
    print(f"  p95([])       = {p95([])}")
    print(f"  describe([])  = {describe([])}")
    print()
    print("分位数插值验证：")
    print(f"  p95([1,2,3,4,5]) = {p95([1, 2, 3, 4, 5])}"
          f"  （插值法 4.8；旧索引法会得 5）")
    print()
    print("分布：")
    for b in distribution([1, 2, 2, 3, 3, 3, 4, 5]):
        print(f"  [{b['lower']}, {b['upper']}) → {b['count']}")
