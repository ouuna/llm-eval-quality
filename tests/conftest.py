"""
pytest 共享 fixture
-------------------

`no_api_config` 用于测试「API 未配置时必须降级」这类逻辑。

为什么需要它
------------
配置有两个来源：进程环境变量和项目根目录的 `.env` 文件。
以前只有一个来源时，`monkeypatch.delenv` 就能模拟"未配置"；
加入 `.env` 之后只清环境变量不够——`.env` 里的真实 key 仍会被读到，
于是本该报"不可用"的用例反而拿到了可用配置，测试就假通过了。

（这类假通过比直接失败更危险：它意味着降级逻辑其实没被测到。
 CI 上没有 `.env`，本地有，同一份代码两种行为。）

所以凡是断言"未配置"的行为，都必须用这个 fixture 同时屏蔽两个来源。
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