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

from tests.check_workflow import check_workflow, WORKFLOW_PATH


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
        """真实 API job 必须保留无密钥跳过逻辑，不能改成无条件跑"""
        assert "secrets.OPENAI_API_KEY != ''" in real_workflow_text

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

    def test_抓出job级if误用表达式(self, tmp_path, real_workflow_text):
        broken = real_workflow_text.replace(
            "if: secrets.OPENAI_API_KEY != ''",
            "if: ${{ secrets.OPENAI_API_KEY != '' }}")
        issues = check_workflow(_write(tmp_path, broken))

        assert issues, "!!! 没抓到 Phase 13 的历史错误，检查器失效"
        assert any("job 级 if" in str(i) for i in issues)
        # 提示里必须含变量名，且不能被截断成 'secrets'
        msg = str(issues[0])
        assert "secrets" in msg
        assert "secrets!=" not in msg, f"提示里的正确写法被截断了：{msg}"

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
