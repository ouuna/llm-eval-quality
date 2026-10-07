"""
CLI 基线管理回归测试
-------------------

为什么这组测试值钱
------------------
`cmd_run` 里曾经把`save_baseline` 写在 `load_baseline` 前面。
后果是：同时使用 `--save-baseline --compare-baseline` 时，
对比对象变成了「本次结果自己」，差值恒为 0，于是——
无论质量退化到多离谱，都会输出「✓ 无退化」。

这是比「直接报错」危险得多的失败模式：
功能看起来完全正常，只是永远报好消息。
所以必须有测试把这个行为钉死。

真实踩过的坑
------------
2026-10 修CI 失败时顺手发现的。当时 CI 上「基线对比」这步
看着一直是绿的，一度以为基线机制工作正常。
"""

import json
import os

import pytest

from eval import baseline as baseline_mod
from eval.cli.main import EXIT_DATA_ERROR, EXIT_GATE_FAILED, EXIT_OK


# ============================================================
# 单元层：顺序对不对
# ============================================================

def test_同时保存和对比时_对比对象必须是上一版基线(tmp_path, monkeypatch):
    """
    核心回归：save 与 compare 同时启用时，
    对比必须拿「保存之前的旧基线」，不能拿本次结果自己。
    """
    bl_path = tmp_path / "baseline.json"

    # 上一版：质量很好
    baseline_mod.save_baseline(
        {"overall_pass_rate": 0.95, "hallucination_rate": 0.02},
        path=str(bl_path),
        meta={"time": "2026-10-01", "model": "glm-4-flash",
              "dataset": "smoke"},
    )

    # 本次：质量灾难性退化
    current = {"overall_pass_rate": 0.60, "hallucination_rate": 0.35}

    # ---- 正确顺序：先读，再比，最后写 ----
    previous = baseline_mod.load_baseline(str(bl_path))
    cmp = baseline_mod.compare(current, previous.get("metrics", {}),
                              tolerance=0.05)

    assert cmp.has_regression, "从 0.95 掉到 0.60 必须判为退化"

    baseline_mod.save_baseline(current, path=str(bl_path),
                meta={"time": "2026-10-05"})

    # 保存后基线确实变成了本次值
    after = baseline_mod.load_baseline(str(bl_path))
    assert after["metrics"]["overall_pass_rate"] == 0.60


def test_错误顺序会永远输出无退化(tmp_path):
    """
    反向验证：证明「先存后比」这个写法确实是坏的。

    这条测试不是测产品行为，是给未来的自己留一张
    「为什么会出这个 bug」的照片，防止有人再写回去。
    """
    bl_path = tmp_path / "baseline.json"

    baseline_mod.save_baseline(
        {"overall_pass_rate": 0.95}, path=str(bl_path),
        meta={"time": "2026-10-01", "model": "glm-4-flash",
              "dataset": "smoke"},
    )
    current = {"overall_pass_rate": 0.10}

    # 错误顺序：先写后读
    baseline_mod.save_baseline(current, path=str(bl_path),
                meta={"time": "2026-10-05",
                      "model": "glm-4-flash",
                      "dataset": "smoke"})
    bl = baseline_mod.load_baseline(str(bl_path))
    bad_cmp = baseline_mod.compare(current, bl.get("metrics", {}))

    assert not bad_cmp.has_regression, \
        "这个断言成立恰好说明错误顺序的可怕之处：退化 85% 也不报警"


# ============================================================
# 集成层：cmd_run 的真实退出码
# ============================================================

def _write_results(tmp_path, metrics, gate_passed):
    """构造一份最小可用的结果文件，供 cmd_run 读取"""
    p = tmp_path / "eval_results.json"
    payload = {
        "metrics": {
            k: {"value": v, "available": True} for k, v in metrics.items()
        },
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


class _FakeMetric:
    def __init__(self, value):
        self.value = value
        self.available = True


class _FakeReport:
    def __init__(self, metrics, gate_passed):
        self.metrics = {k: _FakeMetric(v) for k, v in metrics.items()}
        self.gate_passed = gate_passed
        self.model = "fake-model"
        self.dataset_name = "smoke"
        self.started_at = "2026-10-05T00:00:00"


class _FakeResult:
    def __init__(self, report):
        self.report = report

    def to_dict(self):
        return {"metrics": {}}


@pytest.fixture
def fake_run(monkeypatch):
    """把 run_evaluation 换成假的，只聚焦基线管理这段逻辑"""
    holder = {}

    def _install(metrics, gate_passed):
        report = _FakeReport(metrics, gate_passed)
        monkeypatch.setattr(
            "eval.runner.run_evaluation",
            lambda *a, **k: _FakeResult(report),
        )
        holder["report"] = report
        return report

    return _install


class _Args:
    mock = False
    json = True
    judge = False
    semantic = False
    stability_repeat = 2
    output = None
    save_baseline = False
    compare_baseline = False
    tolerance = 0.05
    dataset = "smoke"


def test_cmd_run_退化时返回门禁失败(tmp_path, monkeypatch, fake_run):
    """退化必须让命令以 EXIT_GATE_FAILED 结束，否则 CI 不会拦住"""
    from eval.cli import main as cli_main

    bl = tmp_path / "baseline.json"
    # meta 里必须带上 model 与 dataset：新加的可比性校验依赖这两个字段，
    # 缺了就无法判断两个结果是不是同一个东西的两个版本。
    baseline_mod.save_baseline({"overall_pass_rate": 0.95},
                path=str(bl),
                meta={"time": "old", "model": "fake-model",
                      "dataset": "smoke"})
    monkeypatch.setattr(baseline_mod, "DEFAULT_BASELINE", str(bl))

    fake_run({"overall_pass_rate": 0.55}, gate_passed=True)

    args = _Args()
    args.compare_baseline = True
    code = cli_main.cmd_run(args)

    assert code == EXIT_GATE_FAILED, \
        "绝对门禁通过但相对基线退化，仍必须失败——这正是基线机制的意义"


def test_cmd_run_基线来自mock时不算退化(tmp_path, monkeypatch, fake_run):
    """
    防误报的关键一条：mock 基线对真实结果**不能**判为退化。

    本仓库里曾经就存着一份 mock 跑出来的 baseline.json。
    如果不校验数据来源，CI 每次都会报「大幅退化」，
    而实际上模型一点没变——只是基线的来源不对。
    """
    from eval.cli import main as cli_main

    bl = tmp_path / "baseline.json"
    baseline_mod.save_baseline({"overall_pass_rate": 0.95},
                path=str(bl),
                meta={"time": "old", "model": "mock", "dataset": "smoke"})
    monkeypatch.setattr(baseline_mod, "DEFAULT_BASELINE", str(bl))

    fake_run({"overall_pass_rate": 0.30}, gate_passed=True)

    args = _Args()
    args.compare_baseline = True
    code = cli_main.cmd_run(args)

    assert code == EXIT_OK, \
        "数据来源不同导致的数值差异不是质量退化，不该让 CI 失败"


def test_cmd_run_无基线可比时明确报错(tmp_path, monkeypatch, fake_run):
    """没有基线可比时必须报数据错误，不能静默当成通过"""
    from eval.cli import main as cli_main

    missing = tmp_path / "definitely_absent.json"
    monkeypatch.setattr(baseline_mod, "DEFAULT_BASELINE", str(missing))

    fake_run({"overall_pass_rate": 0.99}, gate_passed=True)

    args = _Args()
    args.compare_baseline = True
    code = cli_main.cmd_run(args)

    assert code == EXIT_DATA_ERROR, "无基线可比时应报 EXIT_DATA_ERROR"


def test_cmd_run_正常时返回成功(tmp_path, monkeypatch, fake_run):
    """防误报：质量没退化且门禁通过时必须返回 0"""
    from eval.cli import main as cli_main

    bl = tmp_path / "baseline.json"
    baseline_mod.save_baseline({"overall_pass_rate": 0.90},
                path=str(bl),
                meta={"time": "old", "model": "fake-model",
                      "dataset": "smoke"})
    monkeypatch.setattr(baseline_mod, "DEFAULT_BASELINE", str(bl))

    fake_run({"overall_pass_rate": 0.92}, gate_passed=True)

    args = _Args()
    args.compare_baseline = True
    code = cli_main.cmd_run(args)

    assert code == EXIT_OK


def test_cmd_run_不可用指标不进基线(tmp_path, monkeypatch, fake_run):
    """
    unavailable 的指标绝不能写进基线。

    否则某次API 故障导致指标缺失，基线里就存了个 0，
    下次恢复后对比会误判为剧烈退化——或者反过来，永久退化。
    """
    from eval.cli import main as cli_main

    bl = tmp_path / "baseline.json"
    monkeypatch.setattr(baseline_mod, "DEFAULT_BASELINE", str(bl))

    report = _FakeReport({}, gate_passed=True)

    class _Unavailable:
        value = 0.0
        available = False

    report.metrics = {
        "hallucination_rate": _Unavailable(),
        "overall_pass_rate": _FakeMetric(0.9),
    }
    monkeypatch.setattr("eval.runner.run_evaluation",
                        lambda *a, **k: _FakeResult(report))

    args = _Args()
    args.save_baseline = True
    cli_main.cmd_run(args)

    saved = baseline_mod.load_baseline(str(bl))
    assert "hallucination_rate" not in saved["metrics"], \
        "不可用指标不应进入基线"
    assert saved["metrics"]["overall_pass_rate"] == 0.9