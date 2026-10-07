"""
评测阈值集中管理
--------------------------------
解决什么问题
-----------
阈值此前散落在三处，且**配置与代码脱节**：

1. ``config.yaml`` 里写了 ``evaluation.hallucination_threshold: 0.6``，
   但 ``detect_hallucination`` 从来不读它——真正生效的是
   函数签名里的默认值 0.35。改配置没反应，属于典型的假配置。
2. 各评测器内部还有一批硬编码阈值（correctness.py 里0.8/0.5、
   faithfulness.py 里 0.8/0.5、semantic.py 里 0.65…），
   想知道「当前判定标准是什么」只能翻源码。
3. 报告里 ``MetricSummary.definition`` 恒等于指标名，
   等于没有说明，用户看不懂分数怎么来的。

本模块把这些收拢到一处，并让配置真正生效。

阈值从哪来（必须诚实说明）
--------------------------
这些阈值**不是**从大规模标注实验里拟合出来的。
它们是工程初始值，来源是：
  · 部分参考了领域内常见做法（如Jaccard 0.6 判"高相似"）
  · 部分是在本项目 30 条 Gold Set 上观察到的分布手工设定

因此它们**需要后续用独立标注数据校准**，不能当作真理。
校准入口见 ``eval/calibrate.py``。

三层优先级
----------
    代码显式传参 > 本模块（config.yaml 可覆盖） > 内置默认值
"""

import os
from typing import Any, Dict

from eval.config_loader import load_config, get as cfg_get

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 保留原始引用，供测试用 monkeypatch 替换 load_config 时使用。
#
# 为什么要留这一手：测试若写成
#     monkeypatch.setattr(thresholds, "load_config",
#                         lambda: load_config(cfg))
# 而 load_config 本身已被替换，就会无限递归，
# 报出来的是「递归深度超限」——与真正要测的东西完全无关。
# 测试要验证「配置能否被读到」，就不该把读取器本身换掉。
_REAL_LOAD_CONFIG = load_config


# ============================================================
# 内置默认值
# ============================================================
# 每一项都标注了含义和来源等级：
#   "heuristic"  —— 领域经验值，无本地实验支撑
#   "observed"   —— 在本项目 Gold Set 上观察后手工设定
DEFAULT_THRESHOLDS: Dict[str, float] = {
    # ---- 幻觉检测（声明级）----
    # 一条声明的实词覆盖率低于此值，判为「无法被上下文支持」
    "hallucination_supported": 0.60,      # heuristic
    "hallucination_partial": 0.35,        # heuristic
    # forbidden 事实的匹配判定覆盖率
    "forbidden_match": 0.80,              # heuristic

    # ---- 正确性 ----
    "correctness_pass": 0.70,             # heuristic
    "required_fact_hit": 0.50,            # heuristic
    "forbidden_fact_hit": 0.80,           # heuristic

    # ---- 完整性 ----
    "completeness_pass": 0.70,            # heuristic
    "fact_covered": 0.50,                 # heuristic

    # ---- 相关性 ----
    "relevance_level1": 0.50,             # heuristic
    "relevance_level2": 0.50,             # heuristic
    "relevance_level3": 0.40,             # heuristic

    # ---- 语义（Embedding）----
    # 实测同义改写的余弦相似度落在 0.71~0.72，
    # 因此把阈值定在略低于该区间下沿的位置。
    "semantic_fact_coverage": 0.65,       # observed
    "semantic_agreement": 0.50,           # heuristic

    # ---- 稳定性 ----
    "stability_consistency": 0.85,        # heuristic

    # ---- 上下文冲突 ----
    "conflict_predicate_jaccard": 0.50,   # heuristic
}


# 指标说明。用于报告里展示「这个分数怎么算的」。
#
# 键名必须与 runner 产出的 metrics 键名一一对应：
#   overall_pass_rate / hallucination_rate / refusal_accuracy /
#   faithfulness / answer_correctness / completeness / relevance /
#   retrieval_hit_rate / consistency / score_std / p95_latency_ms /
#   error_rate / avg_cost
# 键名对不上（如写成 correctness）会导致报告里 definition 恒为空。
METRIC_DEFINITIONS: Dict[str, str] = {
    "overall_pass_rate": "通过门禁的用例占全部有效用例（排除 error）的比例",
    "answer_correctness": "关键事实的字符级覆盖率；未命中 forbidden 事实",
    "correctness": "关键事实的字符级覆盖率；未命中 forbidden 事实",
    "completeness": "required_facts 中被回答覆盖的比例",
    "relevance": "回答实词与问题/参考信息的重合度（分级判定）",
    "faithfulness": "回答拆分为原子声明后，逐条能否在 context 中找到支撑",
    "hallucination_rate": "存在至少一条无法被context 支撑的声明的用例占比",
    "refusal_accuracy": "应拒答时确实拒答、不该拒答时没有过度拒答",
    "retrieval_hit_rate": "检索返回了非空上下文的用例占比",
    "consistency": "同一问题重复调用的内容 Jaccard 相似度均值",
    "score_std": "稳定性测试中各次得分的标准差",
    "p95_latency_ms": "单次调用延迟的 95 分位数",
    "error_rate": "调用失败（超时、5xx、解析错误）的用例占全部用例的比例",
    "avg_cost": "单次调用的平均成本",
}

# 各指标的局限说明。写进报告是为了让读者不要过度解读数字。
#
# 与 METRIC_DEFINITIONS 一样，键名与 metrics 键名一一对应。
# 需求明确要求「每个指标都要有算法说明和局限说明」，
# 因此这里的键应覆盖全部 13 个指标，缺一个都要补。
METRIC_CAVEATS: Dict[str, str] = {
    "overall_pass_rate": "按用例计数而非按声明计数；分母排除了 error 用例",
    "answer_correctness": "字符级匹配，同义改写可能低估；英文场景额外按词匹配",
    "correctness": "字符级匹配，同义改写可能低估；英文场景额外按词匹配",
    "completeness": "依赖 required_facts 标注质量，标注不全会误判为不完整",
    "relevance": "跑题检测依赖实词交集，问题与答案用词完全不同会误判",
    "faithfulness": "覆盖率算法无法识别「换一种说法表达同一错误」",
    "hallucination_rate": "按用例计数，不按声明计数；粒度较粗",
    "refusal_accuracy": "依赖拒答标记词表，模型换个说法拒绝可能被判为未拒答",
    "retrieval_hit_rate": "只判断是否非空，不判断召回片段是否相关（召回质量需另测）",
    "consistency": "样本仅取前若干条 answer 类用例，不覆盖拒答类",
    "score_std": "仅在有多次重复调用的稳定性测试中才有意义",
    "p95_latency_ms": "受网络与模型服务负载影响，波动大；单次测量不可外推",
    "error_rate": "区分「配置错误」「服务故障」「调用方式错误」需看错误明细",
    "avg_cost": "仅当 API 返回用量信息时才有值；无信息时为 None",
}


# ============================================================
# 加载
# ============================================================
_cache: Dict[str, float] = {}


def _from_config() -> Dict[str, float]:
    """从 config.yaml 的 evaluation 段读取阈值覆盖项"""
    try:
        cfg = load_config()
    except Exception:
        return {}

    section = (cfg or {}).get("evaluation") or {}
    out = {}
    for key in DEFAULT_THRESHOLDS:
        if key in section and isinstance(section[key], (int, float)):
            out[key] = float(section[key])
    return out


def get_threshold(name: str, override: float = None) -> float:
    """
    取单个阈值。

    参数
    ----
    name     阈值名，见 DEFAULT_THRESHOLDS
    override 显式传值，优先级最高（函数参数级）
    """
    if override is not None:
        return float(override)

    if not _cache:
        _cache.update(DEFAULT_THRESHOLDS)
        _cache.update(_from_config())

    return _cache.get(name, 0.0)


def all_thresholds() -> Dict[str, float]:
    """返回当前生效的全部阈值（含配置覆盖）"""
    merged = dict(DEFAULT_THRESHOLDS)
    merged.update(_from_config())
    return merged


def thresholds_from_config_only() -> Dict[str, float]:
    """
    只返回被 config.yaml 覆盖的那些项。

    用途：让 CLI / 报告能告诉用户「你现在有哪几个阈值是配置过的」，
    而不是让人自己去翻配置猜哪些生效了。
    """
    return _from_config()


def definition_of(metric_name: str) -> str:
    """指标的算法说明；没有登记的返回空字符串"""
    return METRIC_DEFINITIONS.get(metric_name, "")


def caveat_of(metric_name: str) -> str:
    """指标的局限说明；没有登记的返回空字符串"""
    return METRIC_CAVEATS.get(metric_name, "")


def clear_cache():
    """清空缓存。改配置后测试用得上"""
    _cache.clear()


# ============================================================
# CLI 入口：python -m eval.thresholds
# ============================================================
if __name__ == "__main__":
    print("=" * 68)
    print("评测阈值一览")
    print("=" * 68)
    print()
    print("优先级：代码显式传参 > config.yaml > 内置默认值")
    print()

    overrides = thresholds_from_config_only()
    current = all_thresholds()

    for name in sorted(current):
        default = DEFAULT_THRESHOLDS[name]
        actual = current[name]
        if name in overrides:
            mark = f" ← 配置文件覆盖（原默认 {default}）"
        else:
            mark = ""
        print(f"  {name:<32} {actual:<8}{mark}")

    print()
    print(f"共 {len(current)} 项，其中 {len(overrides)} 项被配置覆盖")
    if not overrides:
        print("提示：在 config.yaml 的 evaluation 段添加同名项即可覆盖，")
        print("      例如 hallucination_supported: 0.65")

    print()
    print("=" * 68)
    print("阈值来源声明")
    print("=" * 68)
    print("""
这些阈值是**工程初始值，不是实验拟合结果**。

来源分两类：
  · heuristic —— 领域内常见做法，无本地实验支撑
  · observed  —— 在本项目 30 条 Gold Set 上观察后手工设定

Gold Set 仅 30 条且由项目作者本人标注，存在循环论证风险，
因此这些阈值不足以证明「0.6 就该判幻觉」。

要做真正的阈值校准，需要：
  1. 扩充到 200+ 条独立标注样本（含第二位标注者与一致率指标）
  2. 在这批数据上扫描阈值，画出准确率-召回率权衡曲线
  3. 按业务可接受的误报/漏报代价选拐点

校准脚本见 eval/calibrate.py。
""")
