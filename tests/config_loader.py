"""
配置与用例加载器
---------------------------------
实现"框架与业务数据解耦"：换业务场景只需替换
    1. config.yaml     —— 知识库路径、阈值、模型参数
    2. knowledge/      —— 业务文档
    3. cases.csv       —— 评测用例（可由业务方维护）
无需改动任何代码。

依赖：优先使用 PyYAML；未安装时回退到内置轻量解析器。
"""

import os
import csv
import json
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
CASES_PATH = os.path.join(PROJECT_ROOT, "cases.csv")

# 用于"未安装 PyYAML"时提示，但不阻断运行
_YAML_BACKEND = None


# ============================================================
# YAML 加载
# ============================================================
def _load_yaml_text(text):
    """解析 YAML 文本，优先 PyYAML"""
    global _YAML_BACKEND

    if _YAML_BACKEND != "pyyaml":
        try:
            import yaml  # noqa
            _YAML_BACKEND = "pyyaml"
            return yaml.safe_load(text)
        except ImportError:
            _YAML_BACKEND = "fallback"

    return _fallback_parse(text)


def _fallback_parse(text):
    """
    轻量 YAML 解析（仅支持本项目 config.yaml 使用的语法子集）

    支持：
      key: value标量
      key:[a, b, c]          行内列表
      key:                  # 空值 → 后续缩进块
        sub: v1
      - item                列表项
      key: |                多行文本块
      # 注释                整行注释
    """
    root = {}
    # (indent, container) 栈
    stack = [(-1, root)]
    # 最近一个"待填充列表"的 (key, owner_dict)
    list_ctx = None
    block = None  # (key, indent, lines)

    for raw in text.split("\n"):
        # ---- 收集多行文本块 ----
        if block is not None:
            key, base_indent, lines = block
            if raw.strip() == "":
                lines.append("")
                continue
            ind = len(raw) - len(raw.lstrip())
            if ind > base_indent:
                lines.append(raw[base_indent + 2:])
                continue
            stack[-1][1][key] = "\n".join(lines).strip("\n")
            block = None

        line = _strip_comment(raw)
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip())
        content = line.strip()

        # ---- 列表项 ----
        if content.startswith("- "):
            if list_ctx is not None:
                key, owner = list_ctx
                if not isinstance(owner.get(key), dict):
                    owner[key] = {}
                holder = owner[key]
                # 用递增序号作为占位键，便于还原顺序
                holder[str(len(holder))] = _to_scalar(content[2:])
            continue

        if ":" not in content:
            continue

        key, _, val = content.partition(":")
        key, val = key.strip(), val.strip()

        # ---- 多行块起始 ----
        if val in ("|", ">"):
            while len(stack) > 1 and indent <= stack[-1][0]:
                stack.pop()
            cur = stack[-1][1]
            cur[key] = ""
            block = (key, indent, [])
            list_ctx = None
            continue

        # ---- 进入更深层级 ----
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        cur = stack[-1][1]

        if val == "":
            # 下级可能是对象也可能是列表，先建空 dict占位
            cur[key] = {}
            stack.append((indent, cur[key]))
            list_ctx = (key, cur)
        else:
            cur[key] = _to_scalar(val)
            list_ctx = None

    if block is not None:
        key, _, lines = block
        stack[-1][1][key] = "\n".join(lines).strip("\n")

    return _coerce(root)


def _strip_comment(line):
    """剥离行尾注释 respecting 引号"""
    q = None
    for i, ch in enumerate(line):
        if ch in ("'", '"'):
            if q is None:
                q = ch
            elif q == ch:
                q = None
        elif ch == "#" and q is None:
            if i == 0 or line[i - 1] in " \t":
                return line[:i]
    return line


def _to_scalar(raw):
    """字符串 → 布尔/数字/列表/字符串"""
    v = raw.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    if v == "" or v.lower() in ("null", "none", "~"):
        return None
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [_to_scalar(x) for x in inner.split(",")] if inner else []
    low = v.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d*\.\d+", v):
        return float(v)
    return v


def _coerce(node):
    """
    还原被占位成 {} 的列表。

    解析时无法预知 `key:` 后面是对象还是列表，故统一建 {} 占位。
    判定规则：若 dict 的所有 value 均为标量（无嵌套 dict/list），
    且该dict 的所有 key 均为纯数字字符串（解析器用递增序号占位），
    则判定为列表 —— 因为真正的对象其键是配置里的具名字段名。
    """
    if not isinstance(node, dict):
        return node

    for k, v in list(node.items()):
        if isinstance(v, dict):
            if v == {}:
                node[k] = []
            elif _looks_like_list(v):
                node[k] = [v[key] for key in sorted(v.keys(), key=_seq_key)]
            else:
                _coerce(v)
        elif isinstance(v, list):
            node[k] = [_coerce(x) for x in v]
    return node


def _seq_key(k):
    """占位列表的 key 为字符串序号，按数字排序"""
    try:
        return int(k)
    except (TypeError, ValueError):
        return 10 ** 6


def _looks_like_list(d):
    """
    判断 dict 是"列表占位"还是"真实对象"：
    真实对象的键是具名字段名（非纯数字），
    列表占位的键是解析器生成的递增序号（纯数字）。
    """
    if not d:
        return False
    for k, v in d.items():
        if not str(k).isdigit():
            return False
        if isinstance(v, (dict, list)):
            return False
    return True


def load_config(path=None):
    """加载 config.yaml"""
    path = path or CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return _load_yaml_text(f.read())


def get(cfg, dotted, default=None):
    """按点分路径取值：get(cfg, "model.temperature", 0.3)"""
    cur = cfg
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur and cur[part] is not None:
            cur = cur[part]
        else:
            return default
    return cur if cur is not None else default


# ============================================================
# 评测用例加载（CSV / JSON）
# ============================================================
def load_cases(path=None):
    """
    加载评测用例。支持 .csv / .json

    CSV 必需列：case_id, question, category
    可选列：expected_output, key_points（用 | 分隔多个要点）
    """
    path = path or CASES_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"用例文件不存在：{path}\n"
            f"请参考 cases_template.csv 创建"
        )

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    cases = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k: (v.strip() if isinstance(v, str) else v)
                   for k, v in row.items() if k}
            if not row.get("question") or not row.get("category"):
                continue

            kps = row.get("key_points", "")
            row["key_points"] = [x.strip() for x in kps.split("|") if x.strip()]

            for field in ("expected_output", "retrieval_context"):
                if row.get(field) in ("", "None", "null"):
                    row[field] = None if field == "expected_output" else row.get(field)

            cases.append(row)

    return cases


def validate_cases(cases):
    """
    校验用例合法性，返回问题列表
    真实项目中这一步由测试平台做前置校验
    """
    issues = []
    valid_categories = {"in_domain", "out_domain", "ambiguous"}

    for i, c in enumerate(cases, 1):
        cid = c.get("case_id") or f"第{i}行"

        if not c.get("case_id"):
            issues.append(f"[{cid}] 缺少 case_id")

        cat = c.get("category")
        if cat not in valid_categories:
            issues.append(
                f"[{cid}] category 非法：'{cat}'，"
                f"合法值为 in_domain / out_domain / ambiguous"
            )

        if cat == "in_domain" and not c.get("expected_output"):
            issues.append(f"[{cid}] 域内用例应提供 expected_output 作为参考答案")

        if cat == "out_domain" and c.get("expected_output"):
            issues.append(
                f"[{cid}] 域外用例不应有 expected_output"
                f"（期望系统拒答，填None 或留空）"
            )

    return issues


if __name__ == "__main__":
    print("=" * 56)
    print("配置加载")
    print("=" * 56)
    cfg = load_config()
    print(f"YAML 后端: {_YAML_BACKEND}")
    for k, v in cfg.items():
        print(f"  {k}: {type(v).__name__}")

    print("\n关键项:")
    for path in ("system.knowledge_files", "system.top_k",
                 "model.temperature", "evaluation.hallucination_threshold",
                 "gate.min_refusal_rate"):
        print(f"  {path} = {get(cfg, path)}")

    if os.path.exists(CASES_PATH):
        cases = load_cases()
        print(f"\n用例文件: {len(cases)} 条")
        issues = validate_cases(cases)
        print(f"校验问题: {len(issues)} 项")
        for it in issues:
            print(f"  - {it}")
