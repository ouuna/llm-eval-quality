"""
GitHub Actions workflow 静态检查
--------------------------------
为什么需要这个
--------------
Phase 13 写下的 `if: ${{ secrets.OPENAI_API_KEY != '' }}` 一直有语法错误，
但连续 3 次 push 的 CI 全部秒失败（#4 #5 #6）都没被发现——因为：

  ·本地测试不跑 YAML，325 项全绿也照样漏
  · 这个错误让**整个 workflow 文件作废**，不是单个 job 失败，
    所以连"某个 job 变黄"都看不到，是整个 run 直接 Failure
  · 失败耗时 1 秒，更容易让人以为"只是排队"

代价是：真实 API 评测连着三次没在CI 上跑过，而本地一直是绿的。
这正是 CI 的意义——但前提是有人去看结果。

本文件把最常见的几个坑拦在本地。
只做针对性检查，不追求完整的 YAML 解析（项目零第三方依赖）。
"""

import os
import re
import sys

WORKFLOW_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".github", "workflows", "eval.yml",
)

# job 级 if 里不允许出现的上下文变量。
# 报错原文：Unrecognized named-value: 'secrets'
JOB_LEVEL_FORBIDDEN = ("secrets.", "env.", "steps.", "inputs.")


class Issue:
    def __init__(self, line_no, message):
        self.line_no = line_no
        self.message = message

    def __str__(self):
        return f"  行 {self.line_no}: {self.message}"


def _job_level_if_lines(lines):
    """
    找出 job 级的 if 行。

    job 级 if 位于 jobs.<name> 之下、steps 之上，缩进 4 空格。
    step 级 if 缩进 8 空格且在 steps 之后 —— 那里用 ${{ secrets }} 是允许的。
    """
    result = []
    in_steps = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        if stripped.startswith("steps:"):
            in_steps = True
        elif indent <= 2 and stripped and not stripped.startswith("#"):
            # 回到 jobs: 层级或更深，说明换下一个 job 了
            if not stripped.startswith("-"):
                in_steps = False

        if stripped.startswith("if:") and indent == 4 and not in_steps:
            result.append((i + 1, stripped))

    return result


def check_workflow(path=WORKFLOW_PATH):
    """返回Issue 列表，空列表表示通过"""
    issues = []

    if not os.path.exists(path):
        return [Issue(0, f"workflow 文件不存在：{path}")]

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    lines = text.splitlines()

    # --- 坑 1：job 级 if 误用 ${{ }} ---
    for line_no, stmt in _job_level_if_lines(lines):
        if "${{" not in stmt:
            continue
        for token in JOB_LEVEL_FORBIDDEN:
            if token in stmt:
                var = token[:-1]  # 去掉结尾的点号
                issues.append(Issue(
                    line_no,
                    f"job 级 if 中使用了 {token}——此处只能写裸表达式。"
                    f"正确写法：去掉 ${{{{ 与 }}}}，"
                    f"如 if: {var}.XXX != ''"))
                break

    # --- 坑 2：tab 缩进（YAML 完全不接受） ---
    for i, line in enumerate(lines, 1):
        if line.startswith("\t"):
            issues.append(Issue(i, "使用了 tab 缩进，YAML 不允许，必须用空格"))

    # --- 坑 3：缩进非 2 的倍数 ---
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent % 2 != 0:
            issues.append(Issue(
                i, f"缩进 {indent} 不是 2 的倍数，会导致 YAML 解析失败"))

    # --- 坑 4：顶层键缺失 ---
    for key in ("name:", "on:", "jobs:"):
        if not re.search(rf"^{re.escape(key)}", text, re.M):
            issues.append(Issue(0, f"缺少顶层键 {key}"))

    # --- 坑 5：job 缺少 runs-on（永远不会被调度） ---
    # 只在 jobs: 块之后、保持 jobs 自身的缩进层级（2 空格）里找 job
    jobs_start = None
    for i, line in enumerate(lines):
        if re.match(r"^jobs:\s*$", line):
            jobs_start = i
            break

    if jobs_start is not None:
        current_job = None
        for line in lines[jobs_start + 1:]:
            if not line.strip() or line.lstrip().startswith("#"):
                continue

            indent = len(line) - len(line.lstrip())
            if indent < 2:
                break# 退出 jobs 块

            if indent == 2 and line.rstrip().endswith(":"):
                current_job = line.strip().rstrip(":")
                continue

            # job 内部的属性行：只关心有没有声明 runs-on
            if indent == 4 and current_job:
                if re.match(r"runs-on:", line.strip()):
                    current_job = None  # 已确认，等待下一个 job

        if current_job:
            issues.append(Issue(
                0, f"job '{current_job}' 缺少 runs-on，永远不会被调度"))

    # --- 坑 6：${{ }} 未闭合 ---
    if text.count("${{") != text.count("}}"):
        issues.append(Issue(
            0, f"${{{{ 与 }}}} 数量不匹配（{text.count('${{')} vs "
                f"{text.count('}}}')}），表达式未闭合"))

    return issues


def main():
    issues = check_workflow()

    print("=" * 62)
    print("GitHub Actions workflow 检查")
    print("=" * 62)
    print(f"文件：{WORKFLOW_PATH}")

    if not issues:
        print("\n检查通过：未发现上述常见语法问题")
        print("=" * 62)
        return 0

    print(f"\n发现 {len(issues)} 个问题：")
    for issue in issues:
        print(str(issue))
    print("\n提示：workflow 语法错误会让整个文件作废，")
    print("      表现为 CI 瞬间失败（1 秒左右），而非某个 job 变红。")
    print("=" * 62)
    return 1


if __name__ == "__main__":
    sys.exit(main())