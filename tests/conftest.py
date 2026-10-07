"""
pytest 共享 fixture 与 marker 分区
---------------------------------
本文件做两件事：
1. 提供 `no_api_config` fixture（屏蔽全部配置来源）
2. **自动给测试打 marker**

为什么要自动打 marker
-------------------
项目里有500+ 个测试。逐个加 `@pytest.mark.offline` 会带来两个问题：
  · 漏标——某个测试没标，默认就会跑它，可能因缺 API Key 而失败
  · 维护成本——加测试时忘记加，等于埋雷

按目录与文件内容自动判定更可靠：
    tests/performance/       → perf
    tests/api/ tests/negative/ → api
    文件里声明了 live 的     → live
    其余                     → offline

「live 怎么判定」是这里最需要小心的部分：
用**是否引用了真实 Provider 或 live_api fixture** 来判断，
而不是看文件名——因为叫 test_xxx.py 的文件里
可能既有离线用例也有 live 用例。
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval import env_loader


#配置来源涉及的全部变量名
_ALL_CONFIG_VARS = (
    env_loader.ENV_API_KEY,
    env_loader.ENV_BASE_URL,
    env_loader.ENV_MODEL_NAME,
    env_loader.ENV_JUDGE_MODEL,
    env_loader.LEGACY_API_KEY,
    env_loader.LEGACY_BASE_URL,
    env_loader.LEGACY_MODEL_NAME,
    env_loader.LEGACY_JUDGE_MODEL,
)


@pytest.fixture
def no_api_config(monkeypatch, tmp_path):
    """
    彻底模拟「什么都没配置」：清空环境变量，并把 .env 指向一个空文件。

    返回该空文件路径，方便需要时自行写入部分配置。
    """
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")

    for name in _ALL_CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(env_loader, "ENV_FILE", str(empty_env))

    return empty_env


# ============================================================
# marker 自动分类
# ============================================================

# 目录 → marker 的映射
_DIR_MARKERS = {
    "performance": "perf",
    "api": "api",
    "negative": "api",
}

# 出现这些名字的 fixture / 变量，说明该测试需要真实 API
_LIVE_HINTS = (
    "live_api_server",
    "http_sut_provider",
    "sut_server",
    "run_all_eval",
)

# 文件里出现这些标记，说明该文件的部分或全部测试需要真实 API。
#
# 为什么必须看内容而不是只看 fixture 名
# ------------------------------------
# tests/evaluators/test_judge.py 用的是
#     @needs_api = pytest.mark.skipif(not API_AVAILABLE, ...)
# 这种写法——没配key 时会跳过，配了就真跑。
# 从 fixture 名上看不出来，只能读源码。
#
# 不做这一步的后果：本地有 API Key 时，
# 「默认 pytest」会跑几十次真实 LLM 调用，
# 三分钟都跑不完，于是大家干脆不跑了。
_LIVE_SOURCE_HINTS = (
    "@needs_api",
    "@needs_api_key",
    "pytest.mark.live",
    "API_AVAILABLE",
)


def _file_needs_api(path: str) -> bool:
    """读源码判断该文件是否涉及真实 API 调用"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except OSError:
        return False
    return any(h in src for h in _LIVE_SOURCE_HINTS)


def pytest_collection_modifyitems(config, items):
    """
    给每个测试补上 marker。

    规则（按优先级）：
    1. 已显式声明的 marker 不动——显式优先于自动
    2. 用到 live fixture 的 → live
    3. 源码里含 needs_api / skipif 等真实 API 标记 → live
    4. 位于性能测试目录的 → perf
    5. 位于 API/异常测试目录的 → api
    6. 其余 → offline
    """
    # 预先算好每个文件是否需要 API，避免逐个测试重复读盘
    file_needs_api = {}

    for item in items:
        path = str(getattr(item, "fspath", ""))

        # 1. 显式 marker 优先
        if (item.get_closest_marker("live")
                or item.get_closest_marker("perf")
                or item.get_closest_marker("api")
                or item.get_closest_marker("offline")):
            continue

        # 2. 依赖真实 API 的 fixture
        fixturenames = set(getattr(item, "fixturenames", []))
        if fixturenames & set(_LIVE_HINTS):
            item.add_marker("live")
            continue

        # 3. 源码层面判断（结果按文件缓存）
        if path not in file_needs_api:
            file_needs_api[path] = _file_needs_api(path)
        if file_needs_api[path]:
            item.add_marker("live")
            continue

        # 4~5. 按目录判定
        try:
            rel = os.path.relpath(path, PROJECT_ROOT)
        except ValueError:
            rel = path

        parts = rel.replace("\\", "/").split("/")
        for part in parts[:-1]:
            if part in _DIR_MARKERS:
                item.add_marker(_DIR_MARKERS[part])
                break
        else:
            item.add_marker("offline")