"""
workflow 检查器的自检用例
------------------------
一个只会说「通过」的检查器比没有检查器更危险。

所以这里用真实的历史错误当测试数据：
把eval.yml 改回Phase 13 那个写法，检查器必须报错。
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.check_workflow import (
    all_workflow_paths, check_workflow, WORKFLOW_PATH,
)


@pytest.fixture
def real_workflow_text():
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _write(tmp_path, text, name="eval.yml"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


class TestRealFile:

    def test_仓库中的_workflow_无语法问题(self):
        """这个断言就是防止 Phase 13 那个错误再次发生"""
        issues = check_workflow(WORKFLOW_PATH)
        assert not issues, "\n".join(str(i) for i in issues)

    def test_五个_job_都存在且有_runs_on(self, real_workflow_text):
        for job in ("workflow-lint", "unit-tests", "offline-eval",
                    "evaluator-validation", "evaluation"):
            assert f"\n  {job}:" in real_workflow_text, f"缺少 job：{job}"
        assert real_workflow_text.count("runs-on:") == 5, (
            f"应有 5 个 runs-on，实际 {real_workflow_text.count('runs-on:')}")

    def test_api_job有密钥守卫(self, real_workflow_text):
        """
        真实 API job 必须保留「无密钥则跳过」逻辑，不能改成无条件跑。

        注意它现在在 step 级而非 job 级——job 级 if 里 secrets 不可用。
        """
        assert "id: keycheck" in real_workflow_text
        assert "skip=true" in real_workflow_text, "无密钥时必须能跳过"
        assert "::warning::" in real_workflow_text, "跳过时应输出 warning 而非静默"

    def test_job级if不带表达式包裹(self, real_workflow_text):
        """回归用例：Phase 13 写成 ${{ secrets... }} 导致整个文件作废"""
        for line in real_workflow_text.splitlines():
            if line.strip().startswith("if:") and "secrets" in line:
                assert "${{" not in line, (
                    f"job 级 if 不能用 ${{ }} 包裹：{line.strip()}")

    def test_语法检查job排在最前(self, real_workflow_text):
        """
        workflow-lint 必须是第一个 job。

        理由：workflow 本身语法错误时整个文件作废，
        这时若 lint job 被排到别人之后就永远不会执行——
正是「配置坏了，导致检查配置的步骤也跑不了」的死锁。
        """
        jobs_section = real_workflow_text.split("jobs:", 1)[1]
        first_job = re.search(r"\n  ([a-zA-Z0-9_-]+):", jobs_section)
        assert first_job, "解析不到第一个 job"
        assert first_job.group(1) == "workflow-lint", (
            f"第一个 job 应为 workflow-lint，实际是 {first_job.group(1)}")

    def test_其余job都依赖前置检查(self, real_workflow_text):
        """除 workflow-lint 外，每个 job 都必须有 needs，否则绕过检查"""
        jobs_section = real_workflow_text.split("jobs:", 1)[1]

        blocks = {}
        current = None
        for line in jobs_section.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            if indent < 2:
                break
            if indent == 2 and line.rstrip().endswith(":"):
                current = line.strip().rstrip(":")
                blocks[current] = []
            elif current:
                blocks[current].append(line.strip())

        assert len(blocks) == 5, f"应有 5 个 job，实际 {len(blocks)}：{list(blocks)}"

        for name, lines in blocks.items():
            if name == "workflow-lint":
                continue
            assert any(l.startswith("needs:") for l in lines), (
                f"job '{name}' 没有 needs，会绕过 workflow-lint")


class TestDetectsErrors:
    """
    用真实错误当输入，验证检查器确实能报出来。
    只会报「通过」的工具没有价值。
    """

    def test_抓出job级if_裸表达式中的secrets(self, tmp_path, real_workflow_text):
        """
        真实踩过的坑（第三次，也是最隐蔽的一次）。

        前两次修复的错误认知是「job 级 if 不能用表达式包裹」，
        于是去掉包裹改成裸表达式——结果报同样的错。

        真相：job 级 if 在文件解析阶段求值，那时 secrets 尚未注入，
        因此 secrets 在该位置**任何形态都不可用**。
        """
        broken = real_workflow_text.replace(
            "needs: [ unit-tests, offline-eval ]",
            "needs: [ unit-tests, offline-eval ]\n"
            "    if: secrets.OPENAI_API_KEY != ''")
        issues = check_workflow(_write(tmp_path, broken))

        assert issues, "!!! 没抓到裸表达式形态的 secrets 引用"
        assert any("job 级 if" in str(i) for i in issues)
        # 报错信息必须说清是位置问题而非包裹问题
        msg = " ".join(str(i) for i in issues)
        assert "step 级" in msg, f"提示应指向 step 级解法：{msg}"

    def test_抓出job级if_包裹形态中的secrets(self, tmp_path, real_workflow_text):
        """历史错误 1：表达式包裹形态"""
        broken = real_workflow_text.replace(
            "needs: [ unit-tests, offline-eval ]",
            "needs: [ unit-tests, offline-eval ]\n"
            "    if: ${{ secrets.OPENAI_API_KEY != '' }}")
        issues = check_workflow(_write(tmp_path, broken))
        assert issues, "!!! 没抓到包裹形态的 secrets 引用"
        assert any("job 级 if" in str(i) for i in issues)

    def test_step级if_用secrets是合法的(self, tmp_path, real_workflow_text):
        """
        防误报：step 级 if 里 secrets 完全可用，
        这是密钥判断的正确落点，不能报错。
        """
        assert "if: steps.keycheck.outputs.skip == 'false'" in real_workflow_text

        # 把 step 级 if 换成 secrets 判断，仍应合法
        modified = real_workflow_text.replace(
            "if: steps.keycheck.outputs.skip == 'false'",
            "if: secrets.OPENAI_API_KEY != ''")
        assert check_workflow(_write(tmp_path, modified)) == [], (
            "step 级 if 用 secrets 被误报了")

    def test_密钥判断在step级(self, real_workflow_text):
        """
        结构性约束：密钥判断必须在 step 级。
        这是本项目踩坑三次后确定下来的正确做法，不应回退。
        """
        assert "id: keycheck" in real_workflow_text, "缺少 step 级密钥检查"
        assert "skip=true" in real_workflow_text
        assert "skip=false" in real_workflow_text
        # 评测步骤必须受它控制
        assert "if: steps.keycheck.outputs.skip == 'false'" in real_workflow_text

    def test_仓库中job级if不含secrets(self, real_workflow_text):
        """逐行确认：任何 job 级 if 都不含 secrets"""
        jobs_section = real_workflow_text.split("jobs:", 1)[1]
        for i, line in enumerate(jobs_section.splitlines(), 1):
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if stripped.startswith("if:") and indent == 4:
                assert "secrets" not in stripped, (
                    f"job 级 if 含 secrets：{stripped}")

    def test_抓出注释中的secrets表达式(self, tmp_path, real_workflow_text):
        """
        真实踩过的坑（第二次）：为了说明「注释里也别写表达式」而写的注释，
        本身含有 secrets 引用，再次把 workflow 弄挂。

        GitHub 解析注释里的表达式，与代码同等对待。
        """
        broken = real_workflow_text.replace(
            "# 所以「无密钥则跳过」的逻辑放在 step 级（见下方密钥检查 step），",
            "# 反例：${{ secrets.OPENAI_API_KEY }}")
        issues = check_workflow(_write(tmp_path, broken))

        assert issues, "!!! 没抓到注释里的表达式问题"
        assert any("注释" in str(i) for i in issues)

    def test_抓出注释中的steps表达式(self, tmp_path, real_workflow_text):
        """注释里引用 steps 同样非法"""
        broken = real_workflow_text.replace(
            "# 所以「无密钥则跳过」的逻辑放在 step 级（见下方密钥检查 step），",
            "# 详见 ${{ steps.ds.outputs.value }}")
        issues = check_workflow(_write(tmp_path, broken))
        assert any("注释" in str(i) for i in issues)

    def test_step级env用secrets是合法的(self, real_workflow_text):
        """
        防误报：step 级 env 里用 secrets 完全合法，不能报错。

        一个见什么都报的检查器等于没检查器。
        """
        assert "OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}" in real_workflow_text
        assert check_workflow(WORKFLOW_PATH) == []

    def test_concurrency用github是合法的(self, real_workflow_text):
        assert "eval-${{ github.ref }}" in real_workflow_text
        assert check_workflow(WORKFLOW_PATH) == []

    def test_所有secrets引用都在合法位置(self, real_workflow_text):
        """
        逐行确认 secrets 只出现在两种合法位置：
          1. job 级 if 的裸表达式（缩进 4）
          2. step 级 env 块（缩进 >= 8）
        绝不能出现在注释里，也不能被 ${{ }} 包裹。
        """
        seen_any = False

        for i, line in enumerate(real_workflow_text.splitlines(), 1):
            if "secrets." not in line:
                continue
            seen_any = True
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())

            assert not stripped.startswith("#"), (
                f"第 {i} 行注释里出现 secrets 引用：{stripped}")

            is_job_level_if = stripped.startswith("if:") and indent == 4
            is_step_level = indent >= 8

            assert is_job_level_if or is_step_level, (
                f"第 {i} 行 secrets 引用位置不合法（缩进 {indent}）：{stripped}")

            # 合法位置都不该带表达式包裹
            if is_job_level_if:
                assert "${{" not in stripped, (
                    f"第 {i} 行 job 级 if 不能用表达式包裹：{stripped}")

        assert seen_any, "测试前提失效：文件里已经没有 secrets 引用了"

    def test_抓出tab缩进(self, tmp_path, real_workflow_text):
        broken = real_workflow_text.replace(
            "    runs-on: ubuntu-latest", "\t\truns-on: ubuntu-latest", 1)
        issues = check_workflow(_write(tmp_path, broken))
        assert issues, "!!! 没抓到 tab 缩进"
        assert any("tab" in str(i) for i in issues)

    def test_抓出奇数缩进(self, tmp_path, real_workflow_text):
        broken = real_workflow_text.replace(
            "    runs-on: ubuntu-latest", "     runs-on: ubuntu-latest", 1)
        issues = check_workflow(_write(tmp_path, broken))
        assert issues, "!!! 没抓到奇数缩进"

    def test_抓出未闭合表达式(self, tmp_path, real_workflow_text):
        broken = real_workflow_text.replace(
            "${{ github.event_name }}", "${{ github.event_name }", 1)
        issues = check_workflow(_write(tmp_path, broken))
        assert any("未闭合" in str(i) or "不匹配" in str(i) for i in issues)

    def test_抓出文件不存在(self, tmp_path):
        issues = check_workflow(str(tmp_path / "nope.yml"))
        assert issues
        assert "不存在" in str(issues[0])

    def test_抓出缺少顶层键(self, tmp_path):
        minimal = "name: x\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        issues = check_workflow(_write(tmp_path, minimal))
        assert any("on:" in str(i) for i in issues)

    def test_干净文件不报错(self, tmp_path, real_workflow_text):
        assert check_workflow(_write(tmp_path, real_workflow_text)) == []


class TestSecretsAliasConsistency:
    """
    变量名别名检查（2026-10 新增）

    真实踩过的坑
    ------------
    修 CI「密钥检查」step 失败时定位到的：
    那个 step 的 env 写的是 KEY / URL / MODEL 三个别名，
    只给 bash 做空值判断用。但同一个 step 里还有一句
    `python -m eval config`，它读的是 EVAL_API_KEY。

    CI 上没有 .env（被 gitignore），所以程序判定「未配置」，
    返回退出码 2，整个 step 失败。

    整个过程持续了 5 次 push才发现，
    因为 YAML 完全合法、GitHub 不给任何提示、本地测试全绿。
    """

    def test_抓出别名变量名配python调用(self, tmp_path):
        """核心用例：别名 + 同 step 内调CLI，必须报错"""
        broken = """name: T
on:
  push:
    branches: [ main ]
jobs:
  evaluation:
    name: 质量评测
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: 密钥检查
        id: keycheck
        env:
          KEY: ${{ secrets.OPENAI_API_KEY }}
          URL: ${{ secrets.OPENAI_BASE_URL }}
        run: |
          if [ -z "$KEY" ]; then exit 0; fi
          python -m eval config
"""
        issues = check_workflow(_write(tmp_path, broken))
        assert issues, "!!! 没抓到别名变量名问题"
        msg = " ".join(str(i) for i in issues)
        assert "别名" in msg, f"提示应说明问题性质：{msg}"
        assert "子进程" in msg, f"提示应说清根因：{msg}"

    def test_别名但没有python调用不算错(self, tmp_path):
        """
        防误报：纯 shell 步骤用别名是完全合法的，
        不该被这个检查器拦下。
        """
        ok = """name: T
on:
  push:
    branches: [ main ]
jobs:
  evaluation:
    name: 质量评测
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: 纯 shell 判断
        env:
          KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          if [ -z "$KEY" ]; then
            echo "跳过"
          fi
"""
        issues = check_workflow(_write(tmp_path, ok))
        assert issues == [], (
            "纯 shell 步骤用别名不该报错："
            + " ".join(str(i) for i in issues))

    def test_仓库中映射到程序认识的名字(self, real_workflow_text):
        """
        当前修复后的写法：直接映射到 EVAL_API_KEY 等程序认识的名字。
        """
        # 注意不能写成 "KEY: ${{ secrets." —— 那是子串匹配，
        # 会连合法的 EVAL_API_KEY 一起误判。
        # 必须要求行首缩进后紧跟 KEY，即「整段就是 KEY」。
        for line in real_workflow_text.splitlines():
            if "secrets." not in line:
                continue
            name = line.strip().split(":")[0]
            assert name in ("EVAL_API_KEY", "EVAL_BASE_URL",
                            "EVAL_MODEL_NAME", "OPENAI_API_KEY",
                            "OPENAI_BASE_URL", "OPENAI_MODEL_NAME"), \
                f"secrets 被映射到了程序不认识的名字：{name}"

        assert "EVAL_API_KEY: ${{ secrets.OPENAI_API_KEY }}" \
            in real_workflow_text, \
            "secrets 应映射到程序真正读取的变量名"


class TestMultipleWorkflows:
    """
    CI 拆成三个 workflow 后的回归测试。

    为什么要专门测
    --------------
    检查器早先只认 eval.yml 一个文件。拆分后新增的三个文件
    完全不在检查范围内——而 workflow 语法错误会让**整个文件作废**，
    不检查等于没部署过。
    """

    def test_应找到全部workflow(self):
        paths = all_workflow_paths()
        assert len(paths) >= 3, \
            f"只找到 {len(paths)} 个 workflow，目录扫描可能坏了"

    def test_三个拆分后的文件都存在(self):
        import os
        names = {os.path.basename(p) for p in all_workflow_paths()}
        for expected in ("test.yml", "evaluation.yml", "regression.yml"):
            assert expected in names, f"缺少 {expected}"

    def test_每个workflow都无问题(self):
        """逐个检查，而不是只看默认的那个"""
        for path in all_workflow_paths():
            issues = check_workflow(path)
            assert not issues, (
                f"{os.path.basename(path)} 有问题："
                f"{'; '.join(str(i) for i in issues)}")

    def test_能解析为合法yaml(self):
        """
        用真实 YAML 解析器验证。

        自己的检查器是零依赖的正则实现，
        可能有盲区——只有真正的解析器能证明文件可用。
        没有 PyYAML 时跳过，不强制依赖。
        """
        yaml = pytest.importorskip("yaml",
                                    reason="未安装 PyYAML，跳过解析验证")

        for path in all_workflow_paths():
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = yaml.safe_load(f.read())
                except Exception as e:
                    pytest.fail(f"{os.path.basename(path)} YAML 解析失败：{e}")

            assert isinstance(data, dict), \
                f"{os.path.basename(path)} 顶层不是字典"
            jobs = data.get("jobs")
            assert isinstance(jobs, dict) and jobs, \
                f"{os.path.basename(path)} 缺少 jobs 或 jobs 为空"

    def test_每个workflow都有workflow_lint(self):
        """
        每个 workflow 都要有语法检查 job。

        没有它的workflow 一旦写错就是「1 秒失败」，
        排查时连哪个 job 挂了都看不出来。
        """
        for path in all_workflow_paths():
            name = os.path.basename(path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "workflow-lint" in content, \
                f"{name} 缺少 workflow-lint job"
