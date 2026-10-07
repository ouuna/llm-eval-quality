"""
数据集注册表
---------------------------------
统一加载入口。三个层级：

    smoke       快速回归，CI 默认执行
    regression  历史缺陷回归
    full        完整评测，12 类覆盖

也支持"组合数据集"，如 smoke+regression 一起跑。
"""

from eval.schemas.dataset import Dataset, validate_dataset
from eval.datasets.smoke import DATASET as SMOKE
from eval.datasets.full import DATASET as FULL
from eval.datasets.extended import DATASET as EXTENDED
from eval.datasets.coverage import DATASET as COVERAGE
from eval.datasets import regression as _regression_module

# 回归集有两种形态，用途不同：
#   1) 业务视角的 RegressionDataset（Phase 1）—— 用例驱动被测系统评测
#   2) 缺陷视角的 CASES（Phase 7）—— 验证评测器自身不复发
# 二者数据不同，名称也不同，避免混淆
REGRESSION_CASES = _regression_module.CASES

REGISTRY = {
    "smoke": SMOKE,
    "full": FULL,
    "extended": EXTENDED,
    "coverage": COVERAGE,
}

TIER_DESCRIPTION = {
    "smoke": "快速回归集，CI 每次提交执行",
    "full": "完整评测集，手动触发或发版前执行",
    "extended": "扩展评测集：总结/条件/数据异常/边界/对抗五类场景",
    "coverage": "场景覆盖集：单事实/多跳/对比/拒答/诱导五类配比",
}


def get_dataset(name: str) -> Dataset:
    """
    按名称获取数据集，支持用 + 组合

    示例：
        get_dataset("smoke")
        get_dataset("smoke+regression")
    """
    if "+" in name:
        parts = [p.strip() for p in name.split("+") if p.strip()]
        merged = []
        seen = set()
        for p in parts:
            if p not in REGISTRY:
                raise KeyError(f"未知数据集：{p}，可选：{list(REGISTRY)}")
            for c in REGISTRY[p]:
                if c.id not in seen:
                    merged.append(c)
                    seen.add(c.id)
        return Dataset(
            name=name,
            tier="merged",
            description=f"组合数据集：{', '.join(parts)}",
            cases=merged,
        )

    if name not in REGISTRY:
        raise KeyError(f"未知数据集：{name}，可选：{list(REGISTRY)}")
    return REGISTRY[name]


def list_datasets() -> dict:
    """列出所有数据集及其统计"""
    out = {}
    for name, ds in REGISTRY.items():
        out[name] = {
            "tier": ds.tier,
            "description": ds.description,
            "case_count": len(ds),
            "categories": sorted({c.category for c in ds.cases}),
            "difficulties": _dist(ds, "difficulty"),
        }

    # 缺陷回归集单列，因为它不属于用例数据集体系
    reg_stats = _regression_module.statistics()
    out["regression_bugs"] = {
        "tier": "regression",
        "description": "历史缺陷回归集：验证评测器自身不复发",
        "case_count": reg_stats["total"],
        "p0_count": reg_stats["p0_count"],
        "unfixed": reg_stats["unfixed"],
        "by_layer": reg_stats["by_layer"],
    }
    return out


def _dist(ds, attr) -> dict:
    d = {}
    for c in ds.cases:
        k = getattr(c, attr)
        d[k] = d.get(k, 0) + 1
    return d


def validate_all() -> dict:
    """校验所有数据集，返回 {name: [issues]}"""
    return {name: validate_dataset(ds) for name, ds in REGISTRY.items()}


if __name__ == "__main__":
    print("=" * 66)
    print("数据集清单")
    print("=" * 66)
    for name, info in list_datasets().items():
        print(f"\n[{name}]  {info['case_count']} 条  — {info['description']}")
        print(f"  类别：{', '.join(info['categories'])}")
        print(f"  难度：{info['difficulties']}")

    print("\n" + "=" * 66)
    print("校验结果")
    print("=" * 66)
    issues_map = validate_all()
    total = 0
    for name, issues in issues_map.items():
        print(f"\n[{name}] {len(issues)} 个问题")
        for it in issues:
            print(f"  - {it}")
        total += len(issues)
    print(f"\n总计：{total} 个问题")
