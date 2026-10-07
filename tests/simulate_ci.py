"""
CI 环境模拟
--------------------------------
回答一个反复出现的问题：**为什么本地全过、CI 挂？**

真实案例（2026-10-07）
----------------------
tests/test_cli_baseline.py 的 7 项测试：
    本地（Python 3.14 + 有 .env）  → 7 passed
    GitHub Actions（3.11 + 无 .env） → 5 failed，退出码全是 2

根因：`cmd_run` 开头有
    if not args.mock:
        missing = check_config(require_api=True)
        if missing: return EXIT_CONFIG_ERROR   # ← 2

测试直接调 `cmd_run(args)`，而 `_Args.mock = False`。
于是测试能不能过，**取决于运行它的机器上有没有 .env**。

这类问题的恶劣之处
------------------
1. 失败信息（退出码 2）与被测逻辑（基线对比）毫无关系
2. 改代码时很难联想到「是因为本机多了个 .env」
3. 每修一次 CI 都要 push 一轮，等几分钟才知道对不对

怎么办
------
在 CI 上**主动模拟本地环境**——
先跑一遍「无配置」的全量测试，把环境依赖型问题在本地暴露出来。

用法
----
    python -m tests.simulate_ci          # 隔离 .env 跑全量
    python -m tests.simulate_ci -k cli   # 只跑匹配项
"""

import os
import subprocess
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _isolate_env():
    """
    返回 (env_dict, temp_path)，其中 env_dict 不含任何 API 配置。

    做法是把 ENV_FILE 指向一个空文件，并清掉全部相关环境变量——
    与tests/conftest.py 的 no_api_config fixture 同一套思路。
    """
    tmp = tempfile.mkdtemp(prefix="simulate_ci_")
    empty_env = os.path.join(tmp, "empty.env")
    with open(empty_env, "w", encoding="utf-8") as f:
        f.write("")

    env = dict(os.environ)
    for name in (
        "EVAL_API_KEY", "EVAL_BASE_URL", "EVAL_MODEL_NAME",
        "EVAL_JUDGE_MODEL_NAME",
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "OPENAI_MODEL_NAME", "OPENAI_JUDGE_MODEL_NAME",
    ):
        env.pop(name, None)

    env["ENV_FILE"] = empty_env
    return env, tmp


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    print("=" * 66)
    print("CI 环境模拟")
    print("=" * 66)
    print()
    print("目的：暴露「本地过、CI 挂」的环境依赖问题。")
    print("做法：屏蔽全部 API 配置（等价于 CI 上没有 .env）。")
    print()

    env, tmp = _isolate_env()

    cmd = [sys.executable, "-m", "pytest", "tests/",
           "-q", "--tb=short", "-p", "no:cacheprovider"]
    if argv:
        cmd += argv

    print(f"执行：{' '.join(cmd[1:])}")
    print(f"环境：ENV_FILE={env['ENV_FILE']}（空文件）")
    print("-" * 66)

    try:
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                           capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print("超时")
        return 2
    finally:
        # 清理临时目录
        try:
            if os.path.exists(os.path.join(tmp, "empty.env")):
                os.remove(os.path.join(tmp, "empty.env"))
            os.rmdir(tmp)
        except OSError:
            pass

    out = r.stdout or ""
    print(out[-4000:])

    if r.returncode == 0:
        print("=" * 66)
        print("模拟 CI 通过：本地与 CI 表现一致")
        print("=" * 66)
        return 0

    print("=" * 66)
    print("模拟 CI 失败 —— 本地能过但 CI 会挂")
    print("=" * 66)
    print()
    print("常见原因：")
    print("  · 测试依赖了本机的 .env / 环境变量")
    print("  · 测试依赖了某个只在本地存在的文件")
    print("  · 用了本机特有版本的行为（如 Python 3.14 vs 3.11）")
    print()
    print("修法：让测试显式声明它需要什么，")
    print("     而不是依赖运行环境恰好具备什么。")
    print("=" * 66)
    return 1


if __name__ == "__main__":
    sys.exit(main())