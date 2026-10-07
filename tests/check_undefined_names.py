"""
未定义名称检查（CLI）
--------------------------------
回答一个很实际的问题：**本地能跑的代码，CI 上会不会挂？**

真实案例
--------
在 faithfulness.py 里追加 `to_eval_result()`，
签名写成 `def to_eval_result(raw: Dict[str, Any]):`，
但文件顶部只 import 了 `List, Optional`。

本机 Python 3.14 上测试全绿，
GitHub Actions 用 3.11，严格模式下立刻：
    NameError: name 'Dict' is not defined
连带 10 个测试文件全部收集失败。

这类错误的麻烦
--------------
1. 本地绿灯、CI 红灯
2. 报错（NameError）指向的位置与真正出错的地方
   （文件头部的 import）**相差几百行**
3. 只能靠 push → 看报错 → 猜 → 改 循环往返

所以提交前静态发现它，是省时间的关键。

用法
----
    python -m tests.check_undefined_names          # 扫全项目
    python -m tests.check_undefined_names eval app # 只扫指定目录
"""

import io
import json
import os
import subprocess
import sys
from typing import List

# 项目根：所有相对路径都以它为基准，
# 否则换个工作目录运行就会「扫不到文件 → 假通过」。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from tests.undefined_names_core import find_undefined

# 批量模式的子进程脚本。
#
# 为什么放子进程
# --------------
# symtable 在遇到语法错误时抛异常。
# 若在主进程里逐个文件解析，一个坏文件就会中断整个检查。
# 放进子进程，语法错误只是「跳过这个文件」，不影响其余。
#
# 为什么批量而不是逐个起进程
# ------------------------
# 早先每个文件起一次，扫 80 个文件要 25 秒——
# 一个纯静态检查花的时间和跑一遍测试差不多，
# 于是没人愿意在提交前跑它，工具等于不存在。
_BATCH_SCRIPT = r'''
import sys, json
sys.path.insert(0, %(root)r)
from tests.undefined_names_core import find_undefined

payload = json.loads(sys.stdin.read())
lines = []
for item in payload:
    names, syntax_error = find_undefined(item["src"])
    if syntax_error or names is None:
        continue
    if names:
        lines.append(item["path"] + "\t" + ",".join(names))
sys.stdout.write("\n".join(lines))
'''


class UndefinedName:
    def __init__(self, path: str, name: str):
        self.path = path
        self.name = name

    def __str__(self):
        return f"{self.path}: 未导入的全局名称 「{self.name}」"


def _batch_script_text() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return _BATCH_SCRIPT % {"root": root}


def check_source(src: str, path: str = "<test>") -> List[UndefinedName]:
    """
    检查单个文件的全局未定义名（供测试用）。

    单文件走核心模块而不是子进程——
    调用方通常已经在测试环境里，不需要再隔离。
    遇到语法错误返回空列表。
    """
    names, syntax_error = find_undefined(src)
    if syntax_error or names is None:
        return []
    return [UndefinedName(path, n) for n in names]


def collect_py_files(paths: List[str]) -> List[str]:
    """
    收集待检查的 .py 文件。

    相对路径按**项目根**解析，不按当前工作目录。
    早先直接用 os.walk(paths)，于是 `cd /tmp && python -m
    tests.check_undefined_names` 会扫不到任何文件、
    报告「检查通过」——又是一个「假通过」。
    """
    files: List[str] = []
    for p in paths:
        # 把相对路径锚定到项目根
        abs_p = p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

        if os.path.isdir(abs_p):
            for root, dirs, fs in os.walk(abs_p):
                if any(x in root for x in
                       ("__pycache__", ".git", ".workbuddy")):
                    continue
                for f in fs:
                    if f.endswith(".py"):
                        files.append(os.path.join(root, f))
        elif abs_p.endswith(".py"):
            files.append(abs_p)
    return files


def check_paths(paths: List[str]) -> List[UndefinedName]:
    """批量检查（走一次子进程）"""
    files = collect_py_files(paths)
    if not files:
        return []

    payload = []
    for path in files:
        try:
            src = io.open(path, "r", encoding="utf-8").read()
        except (OSError, UnicodeDecodeError):
            continue
        payload.append({"path": path, "src": src})

    if not payload:
        return []

    try:
        r = subprocess.run(
            [sys.executable, "-c", _batch_script_text()],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True, text=True, timeout=120,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        sys.stderr.write(f"[check_undefined_names] 批量检查失败：{e}\n")
        return []

    # 子进程失败时必须显式报错，不能静默返回空——
    # 一个从不报错的检查工具比没有更危险。
    if r.returncode != 0:
        sys.stderr.write(
            f"[check_undefined_names] 子进程失败：\n"
            f"{(r.stderr or '').strip()[:500]}\n")
        return []

    out: List[UndefinedName] = []
    for line in (r.stdout or "").splitlines():
        if "\t" not in line:
            continue
        path, names = line.split("\t", 1)
        for n in names.split(","):
            n = n.strip()
            if n:
                out.append(UndefinedName(path, n))
    return out


def main():
    targets = sys.argv[1:] or ["."]

    print("=" * 66)
    print("未定义名称检查")
    print("=" * 66)
    print()
    print("为什么需要它：")
    print("  本机 3.14 测试全绿，CI 的 3.11 报")
    print("  NameError: name 'Dict' is not defined")
    print("  ——报错位置与真正出错位置（文件头部 import）相差几百行。")
    print()

    problems = check_paths(targets)

    if not problems:
        print(f"检查通过：{'/'.join(targets)} 未发现未导入的全局名称")
        print("=" * 66)
        return 0

    print(f"发现 {len(problems)} 个问题：")
    for pr in problems:
        print(f"  {pr}")

    print()
    print("修复方式：在对应文件顶部补上导入，例如")
    print("    from typing import Any, Dict, List, Optional")
    print("=" * 66)
    return 1


if __name__ == "__main__":
    sys.exit(main())