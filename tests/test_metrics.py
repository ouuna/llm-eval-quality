"""
统计层测试
----------
这个模块看起来简单，但它是所有指标的地基。

值得测的不是「mean 能不能算对」——
那种测试一眼就知道对错。真正要测的是**边界行为**：

  · 空输入返回 None 还是 0？
  · 分位数会不会越界？
  · 单个样本怎么处理？

这些地方出错不会立刻暴露，而是悄悄让门禁判错。
比如返回 0 而不是 None，错误率就成了「0%」，
报告显示「一切正常」，实际上一条数据都没测到。
"""

import math

import pytest

from eval.metrics import (
    count, describe, distribution, error_rate, max_value, mean,
    median, min_value, p50, p90, p95, p99, pass_rate, percentile,
    rate, stability_stats, stdev, variance,
)


class TestEmptyInput:
    """
    空输入行为是本模块最重要的约定。

    必须是 None，不能是 0。
    0 是合法统计值（「真的是 0」），None 才表示「没测到」。
    两者混用会让门禁把「无数据」判成「质量完美」。
    """

    def test_基础统计空输入返回None(self):
        for fn in (mean, median, p95, p99, variance, stdev):
            assert fn([]) is None, f"{fn.__name__} 空输入应返回 None"

    def test_极值空输入返回None(self):
        assert min_value([]) is None
        assert max_value([]) is None

    def test_分位数空输入返回None(self):
        assert percentile([], 95) is None
        for fn in (p50, p90, p95, p99):
            assert fn([]) is None, f"{fn.__name__} 应返回 None"

    def test_比率分母为零返回None(self):
        """除零会抛异常，但「没数据」不是错误"""
        assert pass_rate(0, 0) is None
        assert error_rate(0, 0) is None
        assert rate(1, 0) is None

    def test_空输入不报异常(self):
        """这正是旧实现的问题：某些实现会在空输入时崩掉"""
        # 不加断言，只要求不抛异常
        describe([])
        distribution([])
        stability_stats([])

    def test_过滤None与NaN(self):
        """混入 None/NaN 不应污染结果，也不应崩溃"""
        vals = [1, None, 3, float("nan"), 5]
        assert mean(vals) == 3.0, "应只统计有效数字"
        assert count(vals) == 3

    def test_过滤无穷大(self):
        assert mean([1, float("inf"), 3]) == 2.0

    def test_非数字被忽略(self):
        assert mean([1, "abc", 3]) == 2.0


class TestBasicStats:

    def test_mean(self):
        assert mean([1, 2, 3, 4]) == 2.5

    def test_median_奇数个(self):
        assert median([3, 1, 2]) == 2

    def test_median_偶数个取中间平均(self):
        """
        偶数个样本必须取中间两个的平均。

        取下中位数（sorted[n//2]）会系统性低估：
        [1,2,3,4] 会得 3 而非 2.5。
        监控场景下这会让告警提前触发。
        """
        assert median([1, 2, 3, 4]) == 2.5

    def test_median不依赖输入顺序(self):
        assert median([4, 1, 3, 2]) == median([1, 2, 3, 4])

    def test_variance_总体(self):
        """[1,2,3,4,5] 的总体方差是 2.0"""
        assert variance([1, 2, 3, 4, 5]) == pytest.approx(2.0)

    def test_variance_样本(self):
        """
        [1,2,3,4,5]：均值 3，平方和 = 4+1+0+1+4 = 10
          总体方差 = 10/5 = 2.0
          样本方差 = 10/4 = 2.5
        """
        assert variance([1, 2, 3, 4, 5], sample=True) == pytest.approx(2.5)

    def test_stdev是方差的平方根(self):
        v = variance([2, 4, 4, 4, 5, 5, 7, 9])
        assert stdev([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(math.sqrt(v))

    def test_常量序列方差为零(self):
        assert variance([3, 3, 3]) == 0.0
        assert stdev([3, 3, 3]) == 0.0

    def test_单样本总体方差为零样本方差为None(self):
        """
        单个样本算不出样本方差（自由度为 0）。

        返回 None 而不是 0：0 表示「确定没有波动」，
        而实际是「样本太少，算不出来」。
        """
        assert variance([5], sample=False) == 0.0
        assert variance([5], sample=True) is None


class TestPercentile:
    """
    分位数是本模块最需要测试的部分。

    旧实现 int(len(p)/100) 在 p=100 时会越界，
    插值法与索引法在小样本下结果不同。
    """

    def test_单个样本(self):
        assert percentile([42], 95) == 42
        assert percentile([42], 50) == 42

    def test_全同值(self):
        assert percentile([7, 7, 7, 7], 95) == 7

    def test_插值法而非索引法(self):
        """
        [1,2,3,4,5] 的 P95：
          插值法 → 4.8
          索引法 → 5

        两者在小样本下差别明显。稳定性测试只有 3~5 个样本，
        用索引法会让 P95 直接等于最大值，失去意义。
        """
        assert p95([1, 2, 3, 4, 5]) == pytest.approx(4.8)

    def test_p100返回最大值不越界(self):
        """
        回归测试：旧实现 int(len*1.0) 会越界抛 IndexError。
        """
        assert percentile([1, 2, 3], 100) == 3
        assert p99([1, 2, 3, 4, 5]) == pytest.approx(4.96)

    def test_p0返回最小值(self):
        assert percentile([1, 2, 3], 0) == 1

    def test_超出范围的p被夹紧(self):
        """p=120 应等同100，不报错也不返回 None"""
        assert percentile([1, 2, 3], 120) == 3
        assert percentile([1, 2, 3], -5) == 1

    def test_结果在最小最大之间(self):
        vals = [10, 20, 30, 40, 50]
        for p in (0, 25, 50, 75, 90, 95, 99, 100):
            v = percentile(vals, p)
            assert 10 <= v <= 50, f"P{p} = {v} 超出数据范围"

    def test_单调递增(self):
        """分位数必须随p 单调不减"""
        vals = [1, 5, 3, 8, 2, 9, 4]
        prev = -1
        for p in range(0, 101, 5):
            v = percentile(vals, p)
            assert v >= prev - 1e-9, f"P{p} = {v} 小于前一个分位 {prev}"
            prev = v

    def test_两个样本(self):
        assert p50([10, 20]) == 15


class TestRateFunctions:
    """
    pass_rate 与 error_rate 的分母不同，这是本模块最容易被误用的地方。
    """

    def test_pass_rate(self):
        assert pass_rate(8, 10) == 0.8

    def test_error_rate分母含错误(self):
        """
        3 个错误 / 10 条 = 0.3。

        关键：错误必须计入分母。
        若像 pass_rate 那样把 error 排除出分母，
        「全部失败」会得到 0错误率——
        看起来比「一半失败」更健康，完全说反。
        """
        assert error_rate(3, 10) == 0.3

    def test_全部失败时error_rate为1(self):
        """这是 error_rate 存在的理由"""
        assert error_rate(10, 10) == 1.0

    def test_全部失败时pass_rate无意义(self):
        """
        pass_rate 的分母排除 error，
        所以「全失败」时它没有定义，必须是 None。
        """
        assert pass_rate(0, 0) is None


class TestDescribe:

    def test_包含所有统计量(self):
        d = describe([1, 2, 3, 4, 5])
        for key in ("count", "mean", "min", "max", "median",
                    "p50", "p90", "p95", "p99", "stdev"):
            assert key in d, f"缺少 {key}"

    def test_空输入只含count(self):
        """
        空输入时只返回 count: 0。

        不能出现 mean: None 之类——
        因为「键存在但值为 None」和「键不存在」
        在下游处理时很容易被当成一回事。
        """
        d = describe([])
        assert d == {"count": 0}
        assert "mean" not in d, "无数据时不应出现 mean 键"

    def test_单样本不崩溃(self):
        d = describe([100])
        assert d["count"] == 1
        assert d["mean"] == 100
        assert d["median"] == 100

    def test_精度控制(self):
        d = describe([1, 2, 3], precision=2)
        assert d["mean"] == 2.0
        assert isinstance(d["mean"], float)

    def test_自定义百分位(self):
        d = describe([1, 2, 3, 4, 5], percentiles=(50,))
        assert "p50" in d
        assert "p99" not in d


class TestStabilityStats:

    def test_含极差(self):
        """
        极差是稳定性最直观的指标：
        同一问题重复多次，最高与最低差多少。
        """
        s = stability_stats([0.9, 0.95, 0.88])
        assert s["range"] == pytest.approx(0.07)

    def test_空输入不崩(self):
        assert stability_stats([]) == {"count": 0}


class TestDistribution:

    def test_桶计数总和等于总数(self):
        """
        关键不变量：不能有样本被漏掉或重复计入。
        """
        vals = [1, 2, 2, 3, 3, 3, 4, 5]
        buckets = distribution(vals, buckets=4)
        assert sum(b["count"] for b in buckets) == len(vals)

    def test_最大值被计入最后一个桶(self):
        """
        边界场景：如果最大值因为右开区间被漏掉，
        分布图会显示不到样本总数，看起来像丢数据。
        """
        vals = [1, 2, 3, 4, 100]
        buckets = distribution(vals, buckets=5)
        assert sum(b["count"] for b in buckets) == 5
        assert buckets[-1]["count"] >= 1

    def test_全同值不除零(self):
        d = distribution([5, 5, 5], buckets=5)
        assert d[0]["count"] == 3

    def test_空输入返回空列表(self):
        assert distribution([]) == []

    def test_非法桶数(self):
        assert distribution([1, 2, 3], buckets=0) == []


class TestConsistencyWithOtherModules:
    """
    跨模块一致性。

    此前项目里 P95 有两套实现（result.py 与 runner.py），
    同一指标可能算出两个值。这里验证统计层是唯一来源。
    """

    def test_result_py改用统一实现(self):
        """schemas/result.py 的 latency_stats 应与本层一致"""
        from eval.schemas.result import CaseResult, EvalReport, Status

        results = []
        for ms in (100, 200, 300, 400, 5000):
            r = CaseResult(case_id=f"c{ms}", status=Status.PASSED.value)
            r.latency_ms = ms
            results.append(r)

        rep = EvalReport(dataset_name="t", dataset_tier="t", model="m")
        rep.cases = results
        stats = rep.latency_stats()

        assert stats["mean"] == pytest.approx(mean([100, 200, 300, 400, 5000]))
        assert stats["p95"] == pytest.approx(
            percentile([100, 200, 300, 400, 5000], 95))
        assert stats["max"] == 5000

    def test_latency_stats含新增分位数(self):
        """接统计层后应同时提供 p90 与 p99"""
        from eval.schemas.result import CaseResult, EvalReport, Status

        cases = []
        for i in range(100):
            r = CaseResult(case_id=f"c{i}", status=Status.PASSED.value)
            r.latency_ms = i + 1
            cases.append(r)

        rep = EvalReport(dataset_name="t", dataset_tier="t", model="m")
        rep.cases = cases
        stats = rep.latency_stats()

        assert stats["p90"] is not None
        assert stats["p99"] is not None
        assert stats["p50"] <= stats["p90"] <= stats["p99"]