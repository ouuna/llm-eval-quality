"""
配置加载器（转发层）
---------------------------------
实现已经移到 ``eval/config_loader.py``。

为什么改
--------
它原先放在这里，而 ``app/rag.py``（被测系统）会反向 import 它——
**生产代码依赖测试目录，方向是颠倒的**。
测试代码可以依赖生产代码，反过来不行：
一旦 tests/ 被视为「测试专用」，把生产逻辑塞进去就等于
让 SUT 绑死在测试布局上。

所以现在：
    真正的实现 → ``eval/config_loader.py``
    本文件→ 只做转发，保证既有引用不失效

这样既修正了依赖方向，又不用一次性改掉所有脚本。
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 转发 YAML 后端状态，兼容旧代码里对 _YAML_BACKEND 的读取
from eval.config_loader import (  # noqa: F401
    CONFIG_PATH,
    CASES_PATH,
    PROJECT_ROOT as _PROJECT_ROOT,
    _load_yaml_text,
    _fallback_parse,
    _strip_comment,
    _to_scalar,
    _coerce,
    _looks_like_list,
    _seq_key,
    load_config,
    get,
    load_cases,
    validate_cases,
)
from eval import config_loader as _impl


def _sync_backend():
    """把实现层的 YAML 后端状态同步过来"""
    return _impl._YAML_BACKEND


# 旧代码可能读 config_loader._YAML_BACKEND，
# 简单起见把模块级名字绑成属性代理。
class _ModuleProxy:
    """把未知属性转发到 eval.config_loader"""

    def __getattr__(self, name):
        return getattr(_impl, name)


_impl_module = _ModuleProxy()

if __name__ == "__main__":
    print("=" * 56)
    print("配置加载（转发至 eval/config_loader.py）")
    print("=" * 56)
    cfg = load_config()
    print(f"YAML 后端: {_impl._YAML_BACKEND}")
    for k, v in cfg.items():
        print(f"  {k}: {type(v).__name__}")

    print("\n关键项:")
    for path in ("system.knowledge_files", "system.top_k",
                 "model.temperature", "evaluation.hallucination_threshold",
                 "gate.min_refusal_rate"):
        print(f"  {path} = {get(cfg, path)}")

    if os.path.exists(CASES_PATH):
        cases = load_cases()
        print(f"\n用例文件: {len(cases)} 条")
        issues = validate_cases(cases)
        print(f"校验问题: {len(issues)} 项")
        for it in issues:
            print(f"  - {it}")
