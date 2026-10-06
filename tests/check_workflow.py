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

    # --- 坑 1：job 级 if 里出现 secrets ---
    #
    # 历史踩坑记录（三次）：
    #   1. job 级 if 写成表达式包裹的 secrets
    #   2. 为解释坑 1 而写的注释里含表达式包裹的 secrets
    #   3. 去掉包裹后写成裸 secrets 表达式 —— 仍然报同样的错
    #
    # 正确认知：job 级 if 在文件解析阶段求值，此时 secrets 尚未注入，
    #因此 secrets 在该位置**任何形态都不可用**。
    # 「无密钥则跳过」必须放到 step 级（step 级 if 可用 secrets）。
    for line_no, stmt in _job_level_if_lines(lines):
        if "secrets" not in stmt:
            continue
        issues.append(Issue(
            line_no,
            "job 级 if 中引用了 secrets——该位置 secrets 完全不可用"
            "（文件解析阶段求值，secrets 尚未注入），"
            "无论是否用表达式包裹都会报Unrecognized named-value 并使整个文件作废。"
            "请把密钥判断移到 step 级"))

    # --- 坑 1b：注释中出现 ${{ secrets... }} ---
    #
    # 这是本项目真实踩过的坑，且极具讽刺性：
    # 为了说明「job 级 if 不能用 ${{ secrets }}」而写的注释，
    # 本身含有 ${{ secrets... }}，再次触发同一个错误，
    # workflow 继续作废。
    #
    # GitHub 的表达式解析器不区分注释与代码，
    # 注释里的 ${{ }} 同样会被求值。
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        if "${{" not in stripped:
            continue
        for token in JOB_LEVEL_FORBIDDEN:
            if token in stripped:
                issues.append(Issue(
                    i,
                    f"注释中出现 ${{{{ {token}... }}}}——GitHub 仍会解析注释里的表达式，"
                    f"会触发与代码中相同的错误。请改写注释，"
                    f"例如把大括号拆开或用文字描述"))
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

    # --- 坑 7：表达式的上下文与所在位置不匹配 ---
    #
    # 这才是 GitHub 报错的真正判据：表达式里出现的上下文对象，
    # 必须在该位置可用。
    #
    #   job 级 if     只能用 github.* / needs.* / inputs.*
    #   step 级 env   可以用 secrets.*
    #   注释          全部不可用（但仍会被解析！）
    #
    # 只检查 job 级 if 不够——本项目就栽在注释里：
    # 为了说明错误写法而写的注释，本身含 secrets 上下文，
    # 再次触发同一个错误。
    issues.extend(_check_expression_contexts(lines))

    return issues


# 各位置可用的表达式上下文。
#
# 关键：job 级 if 里 secrets **完全不可用**，不是"不能包在表达式里"，
# 而是连裸表达式都不行。GitHub 在解析 workflow 文件时就求值 job 级 if，
# 那时 secrets 尚未注入。任何形式的 secrets 引用都会报
# Unrecognized named-value: 'secrets' 并让整个文件作废。
_ALLOWED_CONTEXTS = {
    "job_if": ("github", "needs", "inputs", "vars", "hashFiles"),
    "step": ("github", "steps", "inputs", "env", "secrets",
             "needs", "strategy", "matrix", "job", "runner", "hashFiles"),
    "concurrency": ("github", "inputs", "vars"),
}

# 所有合法的上下文前缀
_VALID_PREFIXES = ("github", "secrets", "steps", "inputs", "env",
                   "needs", "strategy", "matrix", "job", "runner",
                   "vars", "hashFiles")


def _extract_contexts(expr):
    """
    从表达式文本里取出所有上下文引用，如 'secrets.XXX' 里的 'secrets'。
    """
    found = set()
    for token in re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\.", expr):
        if token in _VALID_PREFIXES:
            found.add(token)
    return found


def _iter_expressions(text):
    """
    遍历所有 ${{ ... }} 表达式。

    产出 (行号, 所在行文本, 是否注释, 表达式内容)。
    """
    for i, line in enumerate(text.splitlines(), 1):
        for match in re.finditer(r"\$\{\{(.*?)\}\}", line):
            yield i, line, line.strip().startswith("#"), match.group(1)


def _check_expression_contexts(lines):
    issues = []
    seen = set()

    for line_no, line, is_comment, expr in _iter_expressions("\n".join(lines)):
        contexts = _extract_contexts(expr)
        if not contexts:
            continue

        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        if is_comment:
            # 注释：任何上下文都非法（GitHub 仍会解析）
            bad = sorted(contexts)
            key = (line_no, "comment", tuple(bad))
            if key not in seen:
                seen.add(key)
                issues.append(Issue(
                    line_no,
                    f"注释中出现引用 {'/'.join(bad)} 的表达式——"
                    f"GitHub 会解析注释里的表达式并按代码处理，"
                    f"注释中不允许引用任何上下文。"
                    f"请用文字描述，不要写出表达式形态"))
            continue

        # job 级 if：缩进 4 空格、以 if: 开头
        if stripped.startswith("if:") and indent == 4:
            allowed = _ALLOWED_CONTEXTS["job_if"]
            bad = sorted(contexts - set(allowed))
            if bad:
                key = (line_no, "job_if", tuple(bad))
                if key not in seen:
                    seen.add(key)
                    issues.append(Issue(
                        line_no,
                        f"job 级 if 中引用了 {'/'.join(bad)}，"
                        f"该位置只允许 {'/'.join(allowed)}。"
                        f"job 级 if 不需要表达式包裹，"
                        f"直接写裸表达式即可"))
            continue

        # 其他位置：step 级等，允许 secrets
        allowed = _ALLOWED_CONTEXTS["step"]
        bad = sorted(contexts - set(allowed))
        if bad:
            key = (line_no, "step", tuple(bad))
            if key not in seen:
                seen.add(key)
                issues.append(Issue(
                    line_no, f"此处引用了 {'/'.join(bad)}，该位置不可用"))

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