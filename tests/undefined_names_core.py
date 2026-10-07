"""
未定义名称检查的核心逻辑
--------------------------------
被 `tests/check_undefined_names.py` 调用。

为什么要用 symtable 而不是 ast
---------------------------
自己用 `ast` 遍历作用域，判��「这个名字在当前作用域绑定了没」，
必须处理：函数参数、for 变量、with/as、except as、推导式变量、闭包 free 变量、
类作用域、global/nonlocal 声明……

早先试过 ast，漏了几类就产生一批误报
（`r`/`i`/`report` 全是函数内的局部变量）。
**一个满屏误报的检查工具比没有更糟**——
人会立刻开始忽略它的输出，于是真正的 bug 混在里面也看不见。

symtable 是 CPython 官方的作用域分析器，
符号的 local/global/free/assigned 属性由解释器给出，不用自己推。

它能抓到的典型问题
------------------
    from typing import List
    def f(x: Dict[str, int]):   # ← Dict 没导入
        return x

本机 Python 3.14 上测试全绿（注解里的名字在 3.14 宽松处理），
GitHub Actions 的 3.11 严格模式下立刻 NameError。
这类「本地绿、CI 红」的错最费时间，必须静态拦截。
"""

import builtins
import symtable

# CPython 内部机制，不是用户代码的缺陷。
# Python 3.13+ 引入注解惰性求值，
# symtable 会把它暴露成 __conditional_annotations__ 符号。
_INTERNAL_NAMES = frozenset({"__conditional_annotations__"})


def find_undefined(src: str):
    """
    返回 (未定义名列表, 是否语法错误)。

    语法错误返回 (None, True)，由调用方决定怎么处理。
    """
    try:
        top = symtable.symtable(src, "<check>", "exec")
    except SyntaxError:
        return None, True

    # 模块级绑定的名字：顶层定义 + import + 内置
    module_bound = set()
    for sym in top.get_symbols():
        if (sym.is_assigned() or sym.is_imported()
                or sym.is_parameter()):
            module_bound.add(sym.get_name())
    module_bound |= set(dir(builtins))
    module_bound |= {"__file__", "__name__", "__doc__", "__package__"}

    missing = set()

    def walk(table, bound):
        for sym in table.get_symbols():
            name = sym.get_name()

            if not sym.is_referenced():
                continue
            if name in _INTERNAL_NAMES:
                continue
            # 本作用域自己绑定 → 没问题
            if sym.is_assigned() or sym.is_parameter() or sym.is_imported():
                continue
            # 局部名字 → 没问题
            if sym.is_local():
                continue
            # free：闭包变量，绑定在外层，由外层负责检查
            if sym.is_free():
                continue

            # is_global() == True → 引用全局名字，
            # 而它不在 bound 里 → 真的没定义
            if name not in bound:
                missing.add(name)

        for child in table.get_children():
            walk(child, bound)

    # 遍历所有子表，包括 __annotate__。
    # 早先跳过了 dunder 表，结果漏掉了函数注解里的名字——
    # 而注解恰恰是本次真实踩坑的地方：
    # `def f(x: Dict[str, Any])` 里的 Dict 就在 __annotate__ 表中。
    for child in top.get_children():
        walk(child, module_bound)

    # 模块级自身也要查：
    # `print(undefined_thing)` 这类直接写在模块层的未定义名，
    # 才是最常见也最容易被漏掉的情况。
    walk(top, module_bound)

    return sorted(missing), False