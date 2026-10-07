"""
阈值配置化回归测试
------------------
防的是「假配置」—— 看起来可调，实际不可调。

真实踩过的坑
------------
`config.yaml` 里一直写着：

    evaluation:
      hallucination_threshold: 0.6
      completeness_threshold: 0.7

读起来像「阈值可以在配置里调」，但代码里从来没有任何一行去读它们。
真正生效的是各评测器函数签名里的默认值（0.35 / 0.7）。

这类问题特别难发现：
  · 配置项写得清清楚楚，看不出问题
  · 代码能正常运行，不报错
  · 只有在「改了配置想知道为什么没效果」时才会暴露

所以这里用两种方式钉死：
  1. 每个阈值都必须能从 config.yaml 读到
  2. 改配置后，实际生效值必须真的跟着变
"""

import pytest

from eval import thresholds


class TestThresholdRegistry:

    def test_阈值取值在合理区间(self):
        """
        覆盖率类阈值必须在 0~1 之间。

        出现 1.5 或 -0.2 说明配置被写坏了，
        而这种值会让判定永远不成立却不报错。
        """
        assert thresholds.DEFAULT_THRESHOLDS, "阈值表不应为空"

        for name, value in thresholds.DEFAULT_THRESHOLDS.items():
            assert 0.0 <= value <= 1.0, (
                f"{name} = {value} 不在 [0,1] 区间，"
                f"覆盖率类阈值超出该范围会让判定永远不成立")
            assert "_" in name, f"阈值名 {name} 缺少语义分隔"

    def test_指标都有定义和局限说明(self):
        """
        报告里要展示「这个分数怎么算的」和「它不可靠在哪」。

        缺了局限说明，读者会把 0.87 当成精确值，
        而它其实建立在一堆工程初始阈值上。
        """
        assert thresholds.METRIC_DEFINITIONS, "指标定义不应为空"
        assert thresholds.METRIC_CAVEATS, "指标局限说明不应为空"

        for key in ("faithfulness", "correctness", "hallucination_rate"):
            assert key in thresholds.METRIC_DEFINITIONS, f"{key} 缺定义"
            assert thresholds.METRIC_DEFINITIONS[key], f"{key} 定义为空"


class TestConfigTakesEffect:
    """这组是重点：验证配置真的能改变行为。"""

    def test_默认阈值可读(self):
        for name in thresholds.DEFAULT_THRESHOLDS:
            assert isinstance(thresholds.get_threshold(name), (int, float))

    def test_显式传参优先级最高(self):
        assert thresholds.get_threshold(
            "hallucination_supported", override=0.99) == 0.99

    def test_配置文件覆盖真正生效(self, tmp_path, monkeypatch):
        """
        核心回归：写一个非默认阈值到配置里，
        _from_config 必须能读到新值。

        这条在修复前是失败的——配置项存在但从未被读取。
        """
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "evaluation:\n"
            "  hallucination_supported: 0.75\n"
            "  correctness_pass: 0.88\n",
            encoding="utf-8",
        )

        # 注意：这里替换成「按指定路径读取」而不是替换整个函数。
        # 早先写成 lambda: load_config(cfg)，
        # 而 load_config 本身又被替换掉了——无限递归，
        # 报出来的错是 "maximum recursion depth exceeded"，
        # 跟真正要测的东西（配置能否被读取）完全无关。
        monkeypatch.setattr(
            thresholds, "load_config",
            lambda path=None: thresholds._REAL_LOAD_CONFIG(cfg))
        thresholds.clear_cache()

        overrides = thresholds._from_config()
        assert overrides.get("hallucination_supported") == 0.75, \
            f"配置未被读取，实际读到：{overrides}"
        assert overrides.get("correctness_pass") == 0.88

    def test_配置文件缺失时用默认值(self, monkeypatch, tmp_path):
        """配置不存在不该崩，应退回内置默认值"""
        monkeypatch.setattr(
            thresholds, "load_config",
            lambda path=None: thresholds._REAL_LOAD_CONFIG(
                tmp_path / "nope.yaml"))
        thresholds.clear_cache()

        for name, default in thresholds.DEFAULT_THRESHOLDS.items():
            assert thresholds.get_threshold(name) == default

    def test_配置文件格式错误时用默认值(self, monkeypatch):
        """
        坏配置不能拖垮整个评测。

        配置是用户手写的，可能写错；
        错了应该退回默认值并继续跑，
        而不是让整个系统起不来。
        """
        def boom(path=None):
            raise ValueError("配置文件解析失败")

        monkeypatch.setattr(thresholds, "load_config", boom)
        thresholds.clear_cache()

        # 不应抛异常
        assert thresholds.get_threshold("correctness_pass") == \
            thresholds.DEFAULT_THRESHOLDS["correctness_pass"]

    def test_非法类型被忽略(self, tmp_path, monkeypatch):
        """
        配置项写成字符串时必须忽略，不能当数值用。

        比如此时写了 hallucination_supported: "0.8"，
        直接拿去比较会导致 TypeError，
        而且报错位置离出错配置很远，难排查。
        """
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "evaluation:\n"
            "  hallucination_supported: \"0.8\"\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            thresholds, "load_config",
            lambda path=None: thresholds._REAL_LOAD_CONFIG(cfg))
        thresholds.clear_cache()

        overrides = thresholds._from_config()
        assert "hallucination_supported" not in overrides, \
            "字符串类型的阈值应被忽略，而不是被当成数值"


class TestNoFakeConfig:
    """这组防的是「配置文件里写了但代码从不读」。"""

    def test_config中evaluation段全部被识别(self):
        """
        config.yaml 的 evaluation 段里每个阈值键都必须是有效名称。

        如果出现一个不认识的键，说明它又是「看起来可调实际无效」——
        这正是本次要消灭的东西，不能自己再犯。
        """
        try:
            from eval.config_loader import load_config
            cfg = load_config()
        except Exception:
            pytest.skip("config.yaml 不可用")

        section = (cfg or {}).get("evaluation") or {}

        # 非阈值的配置项（属于别的关注点，不走 thresholds）
        allowed_non_threshold = {
            "stability_repeat",
            "stability_metric",
            "between_cases_delay",
        }

        for key in section:
            if key in allowed_non_threshold:
                continue
            assert key in thresholds.DEFAULT_THRESHOLDS, (
                f"config.yaml 的 evaluation.{key} 不是有效阈值名，"
                f"代码不会读取它——属于假配置。"
                f"请改用 eval.thresholds 中已登记的名称")

    def test_已废弃的键不再出现(self):
        """
        hallucination_threshold / completeness_threshold 是废弃键名。

        它们曾出现在 config.yaml 里但从未被读取。
        现在应当彻底消失，避免有人以为改它们有用。
        """
        try:
            from eval.config_loader import load_config
            cfg = load_config()
        except Exception:
            pytest.skip("config.yaml 不可用")

        section = (cfg or {}).get("evaluation") or {}
        for dead_key in ("hallucination_threshold", "completeness_threshold"):
            assert dead_key not in section, (
                f"evaluation.{dead_key} 是废弃键名，"
                f"改它不会有任何效果。"
                f"正确键名见 eval/thresholds.py 的 DEFAULT_THRESHOLDS")