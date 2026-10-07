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
import shutil
import subprocess
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 把项目根加入 sys.path。
#
# 为什么需要这一步
# ----------------
# `python -m tests.simulate_ci` 依赖 cwd 在 sys.path 上。
# GitHub Actions 里workflow 的默认 working-directory 是仓库根，
# 但如果将来加了 `defaults: run: working-directory:`，
# 或在别的目录里调用，这个模块就会ImportError。
#
# 实测踩过：CI 上这个 step 4 秒就失败且没有任何 pytest 输出，
# 典型的「模块都没导入成功」而不是「测试跑挂了」。
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _isolate_env():
    """
    返回 (env_dict, temp_path)，其中 env_dict 不含任何 API 配置。

    两种屏蔽方式都要做，缺一不可
    --------------------------
    1. **清掉环境变量** —— os.environ 里的 EVAL_* / OPENAI_*
    2. **把 .env 临时挪走** —— 这是最关键的一步

    为什么第2 步不可省
    -----------------
    `eval/env_loader.py` 里：
        ENV_FILE = os.path.join(PROJECT_ROOT, ".env")   # 模块级常量
        def load_env_file(path=None):
            target = path or ENV_FILE                # 只认这个常量

    它**不读**环境变量 ENV_FILE。
    早先我只设了环境变量 ENV_FILE，看着像屏蔽了，
    实际上 .env 照读不误——本地因为有 .env 所以"通过"，
    CI 上没有 .env 才暴露这是无效的。

    所以只能真的把文件挪走。
    注意要确保无论测试是否崩溃都能挪回来，
    否则用户会丢配置——那比测试失败严重得多。
    """
    tmp = tempfile.mkdtemp(prefix="simulate_ci_")

    env = dict(os.environ)
    for name in (
        "EVAL_API_KEY", "EVAL_BASE_URL", "EVAL_MODEL_NAME",
        "EVAL_JUDGE_MODEL_NAME",
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "OPENAI_MODEL_NAME", "OPENAI_JUDGE_MODEL_NAME",
    ):
        env.pop(name, None)

    # 关键：把真实的 .env 挪到临时目录
    real_env = os.path.join(PROJECT_ROOT, ".env")
    moved_to = None
    if os.path.exists(real_env):
        moved_to = os.path.join(tmp, "real.env")
        shutil.move(real_env, moved_to)

    return env, tmp, moved_to


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    print("=" * 66)
    print("CI 环境模拟")
    print("=" * 66)
    print()
    print("目的：暴露「本地过、CI 挂」的环境依赖问题。")
    print("做法：屏蔽全部 API 配置（等价于 CI 上没有 .env）。")
    print()

    env, tmp, moved_to = _isolate_env()

    cmd = [sys.executable, "-m", "pytest", "tests/",
           "-q", "--tb=short", "-p", "no:cacheprovider"]
    if argv:
        cmd += argv

    print(f"执行：{' '.join(cmd[1:])}")
    if moved_to:
        print("环境：.env 已临时挪走 + 环境变量已清空（等价 CI）")
    else:
        print("环境：未找到 .env（本身就是 CI 条件），"
              "已清空全部环境变量")

    # 用 try/finally 保证 .env 一定被挪回去。
    #
    # 这是整个工具里最不能出错的地方——
    # 一旦 .env 丢了，用户下次跑真实评测就没得用了，
    # 那比测试失败严重得多。
    #
    # 注意：**finally 里不能 return**。
    # Python 3.14 会警告，3.11 行为也不同（会吞掉正在传播的异常）。
    # 所以恢复失败只记录，不改变返回值。
    restore_failed = False

    try:
        print("-" * 66)
        try:
            r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                               capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            print("超时")
            return 2
    finally:
        if moved_to:
            try:
                shutil.move(moved_to, os.path.join(PROJECT_ROOT, ".env"))
                print("（.env 已恢复）")
            except (OSError, shutil.Error) as e:
                restore_failed = True
                print("=" * 66)
                print(f"严重错误：.env 恢复失败！{e}")
                print(f"你的配置文件在：{moved_to}")
                print("请手动把它挪回项目根目录并重命名为 .env")
                print("=" * 66)
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except OSError:
            pass

    if restore_failed:
        return 3

    out = r.stdout or ""
    err = r.stderr or ""
    print(out[-4000:])
    if err.strip():
        print("--- stderr ---")
        print(err[-1500:])

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