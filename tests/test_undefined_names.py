"""
未定义名称检查工具的测试
--------------------------------
检查工具本身如果不可靠，比没有检查工具更糟——
人一旦发现它老报错，就会开始忽略它的输出。

所以这里测两件事：
  1. 该报的能报（别漏检）
  2. 不该报的别报（别制造噪音）

真实背景
--------
本机 Python 3.14 测试全绿，CI 的 3.11 报：
    NameError: name 'Dict' is not defined
连带 10 个测试文件全部收集失败。
"""

from tests.check_undefined_names import check_source


def _names(src):
    r = check_source(src, "x.py") or []
    return sorted(p.name for p in r)


class TestCatchesRealProblems:
    """必须能抓到——漏检就没有意义"""

    def test_缺typing导入_真实案例(self):
        """
        本次真实踩的坑：追加 to_eval_result() 时
        用了 Dict[str, Any] 但文件顶部只 import 了 List, Optional。
        """
        src = "from typing import List\ndef f(x: Dict[str, int]):\n    return x\n"
        assert "Dict" in _names(src)

    def test_模块级未定义名(self):
        src = "print(undefined_thing)\n"
        assert "undefined_thing" in _names(src)

    def test_函数体引用未定义全局(self):
        src = "def f():\n    return missing_name\n"
        assert "missing_name" in _names(src)

    def test_注解里的未定义名(self):
        """注解是本次出问题的位置，必须能查"""
        src = ("from typing import List\n"
               "def f(x: List) -> Set[str]:\n    return set()\n")
        assert "Set" in _names(src)

    def test_多个缺失都能报出来(self):
        src = "def f():\n    return Alpha + Beta\n"
        names = _names(src)
        assert "Alpha" in names and "Beta" in names


class TestNoFalsePositives:
    """不能制造噪音——满屏误报的检查会被直接忽略"""

    def test_导入完整(self):
        src = ("from typing import Dict, List, Optional\n"
               "def f(x: Dict) -> List[str]:\n    return []\n")
        assert _names(src) == []

    def test_局部变量(self):
        src = ("def f():\n"
               "    r = 1\n"
               "    for i in range(3):\n"
               "        r += i\n"
               "    return r\n")
        assert _names(src) == []

    def test_函数参数(self):
        src = "def f(a, b=1, *args, **kw):\n    return a\n"
        assert _names(src) == []

    def test_except_as(self):
        src = ("def f():\n"
               "    try:\n        pass\n"
               "    except ValueError as e:\n        print(e)\n")
        assert _names(src) == []

    def test_with_as(self):
        src = "def f():\n    with open('x') as fh:\n        return fh.read()\n"
        assert _names(src) == []

    def test_推导式变量(self):
        src = "def f(items):\n    return [x * 2 for x in items]\n"
        assert _names(src) == []

    def test_闭包free变量(self):
        src = ("def outer():\n"
               "    z = 1\n"
               "    def inner():\n        return z\n"
               "    return inner\n")
        assert _names(src) == []

    def test_类与方法(self):
        src = "class C:\n    def m(self):\n        return self\n"
        assert _names(src) == []

    def test_dataclass(self):
        src = ("from dataclasses import dataclass\n"
               "@dataclass\n"
               "class A:\n    x: int = 1\n")
        assert _names(src) == []

    def test_内置名(self):
        src = "def f():\n    return len([]) + str(1) + int('2')\n"
        assert _names(src) == []

    def test_dunder不报(self):
        src = "def f():\n    return __name__\n"
        assert _names(src) == []

    def test_语法错误不崩溃(self):
        """语法错误的文件应被跳过，而不是让工具崩掉"""
        assert check_source("def broken(:\n", "x.py") in (None, [])

    def test_空文件(self):
        assert check_source("", "x.py") in (None, [])


class TestProjectClean:
    """当前项目应该是干净的"""

    def test_全项目无未定义名(self):
        from tests.check_undefined_names import check_paths

        problems = check_paths(["eval", "app", "tests"])
        assert not problems, \
            f"发现 {len(problems)} 个未定义名：" \
            f"{[str(p) for p in problems[:5]]}"

    def test_faithfulness已补Dict导入(self):
        """
        回归测试：确保本次修的 bug 没有复发。

        faithfulness.py 曾因缺 Dict 导入导致 CI 挂掉。
        """
        with open("eval/evaluators/faithfulness.py",
                  encoding="utf-8") as f:
            lines = f.readlines()

        typing_line = next(
            (ln for ln in lines if ln.startswith("from typing import")),
            "")

        assert "Dict" in typing_line, \
            f"faithfulness.py 的 typing 导入缺 Dict：{typing_line.strip()}"
        assert "Any" in typing_line, \
            f"faithfulness.py 的 typing 导入缺 Any：{typing_line.strip()}"