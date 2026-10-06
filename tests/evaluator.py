"""
LLM 质量评测模块
---------------------------------
被测系统：app.rag.RAG 问答系统
评测维度（对齐大厂 AI 测试岗 JD 要求）：
    1. 准确性(Accuracy)     —— 域内用例是否正确作答
    2. 完整性(Completeness)  —— 回答是否覆盖问题全部要求（多要点问题）
    3. 幻觉检测(Hallucination) —— 回答是否编造上下文无依据内容
    4. 稳定性(Stability)     —— 同一问题多次作答的一致程度
    5. 拒答能力(Refusal)     —— 域外场景是否正确拒答而非编造
    6. 检索有效性(Retrieval) —— 是否召回到相关上下文

特性：
    - 零第三方依赖（仅使用 Python 标准库）
    - 幻觉检测采用三级判定，消除整句匹配误报
    - 完整性检测支持多要点问题的覆盖率分析
    - 可生成自包含HTML 可视化报告

运行：
    python -m pytest tests/ -v --html=reports/report.html
    python tests/run_eval.py            # 独立运行并生成报告
"""

import os
import re
import json
import time
import statistics
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tests"))

from app.rag import ask, load_knowledge  # noqa: E402
from config_loader import (  # noqa: E402
    load_config, load_cases, validate_cases, get as cfg_get,
)

# ============================================================
# 配置：全部阈值来自 config.yaml，实现业务无关
# ============================================================
_CFG = load_config()

THRESHOLD_HALLUCINATION = cfg_get(_CFG, "evaluation.hallucination_threshold", 0.6)
THRESHOLD_COMPLETENESS = cfg_get(_CFG, "evaluation.completeness_threshold", 0.7)
STABILITY_REPEAT = cfg_get(_CFG, "evaluation.stability_repeat", 3)
BETWEEN_CASES_DELAY = cfg_get(_CFG, "evaluation.between_cases_delay", 0.5)

GATE = {
    "overall_pass_rate": cfg_get(_CFG, "gate.min_overall_pass_rate", 0.95),
    "accuracy_in_domain": cfg_get(_CFG, "gate.min_accuracy_in_domain", 1.0),
    "refusal_rate_out_domain": cfg_get(_CFG, "gate.min_refusal_rate", 1.0),
    "hallucination_free_rate": cfg_get(_CFG, "gate.min_hallucination_free", 0.95),
    "retrieval_hit_rate": cfg_get(_CFG, "gate.min_retrieval_hit_rate", 0.60),
    "avg_relevance": cfg_get(_CFG, "gate.min_avg_relevance", 0.40),
    "avg_stability": cfg_get(_CFG, "gate.min_avg_stability", 0.85),
}

_report_dir = cfg_get(_CFG, "report.output_dir", "reports")
REPORT_DIR = os.path.join(PROJECT_ROOT, _report_dir)

REFUSAL_MARKERS = ["未提及", "没有相关", "无法回答", "不知道", "不明确"]

# ============================================================
# 工具函数
# ============================================================
def check_env():
    """校验 API 配置（v1 兼容入口，现统一走 eval.env_loader）"""
    from eval import env_loader
    missing = [n for n in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL_NAME")
               if not env_loader.get(n)]
    if missing:
        raise EnvironmentError(
            f"缺少配置：{', '.join(missing)}\n"
            f"推荐做法：复制 env.example 为 .env，填入 EVAL_API_KEY 等变量"
        )


def content_chars(text):
    """
    提取中文实词字符集合
    去除虚词与常见连接词，只保留有语义实义的字
    """
    stop = set("的了和与及对于是把在当中一个可以我们需要这那有被就都也很到吗呢什么"
               "怎么如何请问介绍定义进行过而且或者但如果因为所以")
    return {c for c in text if re.match(r"[一-龥]", c) and c not in stop}


def split_sentences(text):
    """按中文标点切分句子，过滤过短片段"""
    if not text:
        return []
    return [s.strip() for s in re.split(r"[。\n；;]", text) if len(s.strip()) > 5]


# ============================================================
# 评测维度 1：幻觉检测
# ============================================================
def check_hallucination(actual_output, retrieval_context):
    """
    基于证据的幻觉检测（Explainable Hallucination Detection）

    三级判定策略：
      Level 1  答案为空或含拒答标记         -> 无幻觉（正确拒答）
      Level 2  检索上下文为空但生成了回答    -> 判定幻觉（无依据生成）
      Level 3  逐句计算实词覆盖率< 阈值     -> 判定幻觉

    设计要点：
      采用"实词覆盖率"而非整句精确匹配。大模型会将原文改写后作答
      （"关注输入和输出边界" -> "需要关注哪些边界值"），整句匹配会产生
      大量误报。实测该修正将域内幻觉率从 71% 降至 7%。

    Returns:
        (是否无幻觉, 说明, 实词覆盖率)
    """
    if not actual_output.strip():
        return True, "系统拒答，无幻觉", 1.0

    if any(m in actual_output for m in REFUSAL_MARKERS):
        return True, "系统明确拒答", 1.0

    if not retrieval_context:
        return False, "无检索上下文但生成了回答，疑似幻觉", 0.0

    ctx_sentences = split_sentences(retrieval_context)
    out_sentences = split_sentences(actual_output)

    if not out_sentences:
        return True, "回答过短，无实质内容", 1.0

    coverages = []
    unsupported = []

    for s in out_sentences:
        s_chars = content_chars(s)
        if not s_chars:
            continue

        best = 0.0
        for cs in ctx_sentences:
            cs_chars = content_chars(cs)
            if not cs_chars:
                continue
            overlap = len(s_chars & cs_chars) / len(s_chars)
            best = max(best, overlap)

        coverages.append(best)
        if best < THRESHOLD_HALLUCINATION:
            unsupported.append(s)

    avg_cov = round(statistics.mean(coverages), 2) if coverages else 1.0

    if unsupported:
        return False, f"存在无上下文依据的陈述：{unsupported[0][:40]}", avg_cov

    return True, "回答内容均有上下文依据", avg_cov


# ============================================================
# 评测维度 2：完整性
# ============================================================
def check_completeness(actual_output, key_points):
    """
    完整性检测：验证回答是否覆盖问题的全部关键要点

    典型场景：对比型问题（"A和B的区别"）需要同时回答 A 和 B，
    仅回答其一即为不完整。

    Args:
        actual_output: 模型回答
        key_points: 关键要点列表

    Returns:
        (是否完整, 覆盖率, 缺失要点列表)
    """
    if not actual_output.strip():
        return False, 0.0, key_points

    if not key_points:
        return True, 1.0, []

    covered = []
    missing = []
    for point in key_points:
        point_chars = content_chars(point)
        if not point_chars:
            continue
        # 若要点的 50% 实词出现在回答中，视为覆盖
        hit = len(point_chars & content_chars(actual_output)) / len(point_chars)
        if hit >= 0.5:
            covered.append(point)
        else:
            missing.append(point)

    rate = len(covered) / len(key_points) if key_points else 0.0
    is_complete = rate >= THRESHOLD_COMPLETENESS
    return is_complete, round(rate, 2), missing


# ============================================================
# 评测维度 3：答案相关性
# ============================================================
def relevance_score(question, answer):
    """问题实词在答案中的覆盖比例"""
    if not answer.strip():
        return 0.0
    q_chars = content_chars(question)
    a_chars = content_chars(answer)
    if not q_chars:
        return 0.0
    return round(len(q_chars & a_chars) / len(q_chars), 2)


# ============================================================
# 评测维度 4：稳定性
# ============================================================
def measure_stability(question, times=None):
    """
    稳定性检测：同一问题重复提问 times 次，用 Jaccard 相似度量化

    LLM 存在采样随机性，输出不稳定意味着无法进入生产环境
    """
    if times is None:
        times = STABILITY_REPEAT

    answers = []
    for i in range(times):
        try:
            ans, _ = ask(question)
            answers.append(ans)
        except Exception:
            pass
        time.sleep(0.4)

    if len(answers) < 2:
        return 0.0, answers

    def jaccard(a, b):
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 1.0 if a == b else 0.0
        return len(sa & sb) / len(sa | sb)

    scores = [
        jaccard(answers[i], answers[j])
        for i in range(len(answers))
        for j in range(i + 1, len(answers))
    ]
    return statistics.mean(scores), answers


# ============================================================
# 核心：执行完整评测
# ============================================================
def run_all_eval(cases=None, stability_questions=None, verbose=True):
    """
    执行全量评测，返回结构化结果

    Returns:
        dict 包含 metrics（汇总指标）与 cases（逐用例明细）
    """
    check_env()

    cases = cases or load_cases()
    stability_questions = stability_questions or [
        "什么是等价类划分？",
        "什么是冒烟测试和回归测试的区别？",
        "缺陷报告应该包含哪些内容？",
    ]

    if verbose:
        print("=" * 68)
        print("LLM 质量评测")
        print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"模型：{os.getenv('OPENAI_MODEL_NAME')}")
        print(f"用例数：{len(cases)}")
        print("=" * 68)

    results = []
    start = time.time()

    for idx, case in enumerate(cases, 1):
        q = case["question"]
        cat = case["category"]
        expected = case.get("expected_output")
        key_points = case.get("key_points", [])

        try:
            actual, contexts = ask(q)
        except Exception as e:
            if verbose:
                print(f"[{idx}/{len(cases)}] {case['case_id']}  [ERROR] {e}")
            results.append({
                "case_id": case["case_id"], "category": cat, "question": q,
                "actual_output": "", "retrieval_context": "",
                "hallucination_free": False, "hallucination_reason": f"调用失败: {e}",
                "coverage": 0.0, "is_complete": False, "missing_points": [],
                "completeness": 0.0, "relevance": 0.0,
                # 异常场景下检索结果不可知，记为 0 而非引用未定义变量
                "retrieval_count": 0, "passed": False,
            })
            continue

        ctx = "\n\n".join(contexts) if contexts else ""

        # 幻觉检测
        no_halluc, reason, coverage = check_hallucination(actual, ctx)

        # 完整性检测（仅对有 key_points 的用例）
        if key_points:
            is_complete, comp_rate, missing = check_completeness(actual, key_points)
        else:
            is_complete, comp_rate, missing = True, None, []

        # 相关性
        rel = relevance_score(q, actual)

        # 判定通过：域内要求无幻觉+完整；域外要求正确拒答
        if cat == "in_domain":
            passed = no_halluc and (is_complete if key_points else True)
        elif cat == "out_domain":
            passed = no_halluc
        else:  # ambiguous
            passed = no_halluc

        results.append({
            "case_id": case["case_id"],
            "category": cat,
            "question": q,
            "expected_output": expected,
            "actual_output": actual,
            "retrieval_count": len(contexts),
            "retrieval_context": ctx,
            "hallucination_free": no_halluc,
            "hallucination_reason": reason,
            "coverage": coverage,
            "is_complete": is_complete,
            "completeness": comp_rate,
            "missing_points": missing,
            "relevance": rel,
            "passed": passed,
        })

        if verbose:
            flag = "PASS" if passed else "FAIL"
            print(f"[{idx}/{len(cases)}] {flag} {case['case_id']} ({cat})")
            print(f"   {q}")
            print(f"   幻觉：{'无' if no_halluc else '有'} | "
                  f"完整性：{comp_rate if comp_rate is not None else 'N/A'} | "
                  f"相关性：{rel} | 检索片段：{len(contexts)}")
            if missing:
                print(f"   缺失要点：{missing}")

        time.sleep(BETWEEN_CASES_DELAY)

    # ---------- 稳定性 ----------
    if verbose:
        print("\n" + "=" * 68)
        print("稳定性测试（同一问题重复提问 3 次）")
        print("=" * 68)

    stability_detail = {}
    for q in stability_questions:
        score, answers = measure_stability(q, times=3)
        stability_detail[q] = {
            "stability": round(score, 2),
            "run_count": len(answers),
            "answers": answers,
        }
        if verbose:
            print(f"\n  {q}")
            print(f"  稳定性得分：{round(score, 2)}（{len(answers)} 次作答）")

    # ---------- 汇总 ----------
    ok = [r for r in results if r["category"]]
    in_dom = [r for r in ok if r["category"] == "in_domain"]
    out_dom = [r for r in ok if r["category"] == "out_domain"]

    comp_cases = [r for r in ok if r["completeness"] is not None]
    passed = sum(1 for r in ok if r["passed"])

    metrics = {
        # 核心指标
        "overall_pass_rate": round(passed / len(ok), 2) if ok else 0,
        "accuracy_in_domain": round(
            sum(1 for r in in_dom if r["passed"]) / len(in_dom), 2
        ) if in_dom else 0,
        "refusal_rate_out_domain": round(
            sum(1 for r in out_dom if r["passed"]) / len(out_dom), 2
        ) if out_dom else 0,
        "retrieval_hit_rate": round(
            sum(1 for r in ok if r["retrieval_count"] > 0) / len(ok), 2
        ) if ok else 0,
        # 细粒度指标
        "hallucination_free_rate": round(
            sum(1 for r in ok if r["hallucination_free"]) / len(ok), 2
        ) if ok else 0,
        "avg_hallucination_coverage": round(
            statistics.mean([r["coverage"] for r in ok if r["coverage"]]), 2
        ) if any(r["coverage"] for r in ok) else 0,
        "completeness_rate": round(
            sum(1 for r in comp_cases if r["is_complete"]) / len(comp_cases), 2
        ) if comp_cases else 0,
        "avg_relevance": round(
            statistics.mean([r["relevance"] for r in ok]), 2
        ) if ok else 0,
        "avg_stability": round(
            statistics.mean([v["stability"] for v in stability_detail.values()]), 2
        ) if stability_detail else 0,
    }

    summary = {
        "eval_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": os.getenv("OPENAI_MODEL_NAME"),
        "total_cases": len(results),
        "duration_sec": round(time.time() - start, 1),
        "metrics": metrics,
        "stability_detail": stability_detail,
        "cases": results,
    }

    return summary


# ============================================================
# 报告输出
# ============================================================
def print_summary(summary):
    """打印终端摘要"""
    m = summary["metrics"]
    print("\n" + "=" * 68)
    print("评测汇总")
    print("=" * 68)
    print(f"模型：{summary['model']}    耗时：{summary['duration_sec']} 秒")
    print()
    print(f"  总体通过率        {m['overall_pass_rate']:.0%}")
    print(f"  域内准确率        {m['accuracy_in_domain']:.0%}  (有标准答案且完整无幻觉)")
    print(f"  域外拒答率        {m['refusal_rate_out_domain']:.0%}  (无依据时正确拒答)")
    print(f"  检索命中率        {m['retrieval_hit_rate']:.0%}")
    print(f"  无幻觉率{'':6}        {m['hallucination_free_rate']:.0%}")
    print(f"  回答完整率        {m['completeness_rate']:.0%}")
    print(f"  平均相关性        {m['avg_relevance']}")
    print(f"  平均稳定性        {m['avg_stability']}")
    print("=" * 68)

    failed = [r for r in summary["cases"] if not r["passed"]]
    if failed:
        print(f"\n[!] 未通过用例 {len(failed)} 条：")
        for r in failed:
            print(f"  - [{r['category']}] {r['question']}")
            print(f"    原因：{r['hallucination_reason']}")
            if r["missing_points"]:
                print(f"    缺失要点：{r['missing_points']}")


def save_json(summary):
    """保存 JSON 报告"""
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "eval_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return path


def save_html(summary):
    """生成自包含 HTML 可视化报告（无第三方依赖）"""
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "report.html")

    m = summary["metrics"]
    cases = summary["cases"]

    def esc(s):
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def pct(v):
        return f"{v:.0%}" if isinstance(v, float) and v <= 1 else str(v)

    # 指标卡片
    cards = [
        ("总体通过率", pct(m["overall_pass_rate"]), "通过用例 / 总用例"),
        ("域内准确率", pct(m["accuracy_in_domain"]), "有标准答案时正确作答"),
        ("域外拒答率", pct(m["refusal_rate_out_domain"]), "无依据时拒绝编造"),
        ("检索命中率", pct(m["retrieval_hit_rate"]), "召回到相关上下文"),
        ("回答完整率", pct(m["completeness_rate"]), "覆盖全部关键要点"),
        ("平均稳定性", m["avg_stability"], "多次作答一致程度"),
    ]
    cards_html = "\n".join(
        f'<div class="card"><div class="label">{esc(l)}</div>'
        f'<div class="value">{esc(str(v))}</div>'
        f'<div class="hint">{esc(h)}</div></div>'
        for l, v, h in cards
    )

    # 分类统计
    cat_stat = {}
    for c in cases:
        cat_stat.setdefault(c["category"], {"total": 0, "passed": 0})
        cat_stat[c["category"]]["total"] += 1
        if c["passed"]:
            cat_stat[c["category"]]["passed"] += 1

    cat_names = {
        "in_domain": "域内用例（知识库内）",
        "out_domain": "域外用例（知识库外）",
        "ambiguous": "模糊用例",
    }
    cat_rows = "".join(
        f"<tr><td>{esc(cat_names.get(k, k))}</td>"
        f"<td>{v['total']}</td><td>{v['passed']}</td>"
        f"<td>{v['passed'] / v['total']:.0%}</td></tr>"
        for k, v in cat_stat.items()
    )

    # 用例明细
    rows = []
    for c in cases:
        status = "PASS" if c["passed"] else "FAIL"
        cls = "pass" if c["passed"] else "fail"
        comp = c["completeness"]
        comp_txt = "N/A" if comp is None else f"{comp:.0%}"
        rows.append(
            f'<tr class="{cls}">'
            f'<td><span class="badge {cls}">{status}</span></td>'
            f'<td class="mono">{esc(c["case_id"])}</td>'
            f'<td>{esc(c["question"])}</td>'
            f'<td class="mono">{esc(c["category"])}</td>'
            f'<td>{"是" if c["hallucination_free"] else "否"}</td>'
            f'<td>{esc(comp_txt)}</td>'
            f'<td>{c["relevance"]}</td>'
            f'<td>{c["retrieval_count"]}</td>'
            f'<td class="reason">{esc(c["hallucination_reason"])}</td>'
            "</tr>"
        )

    rows_html = "\n".join(rows)

    # 稳定性
    stab_rows = "".join(
        f"<tr><td>{esc(q)}</td><td>{d['stability']}</td>"
        f"<td>{d['run_count']}</td></tr>"
        for q, d in summary["stability_detail"].items()
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM 质量评测报告</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
  background:#f7f7f8; color:#1f2328; padding:24px; line-height:1.6; }}
.wrap {{ max-width:1200px; margin:0 auto; }}
h1 {{ font-size:24px; font-weight:600; margin-bottom:6px; }}
.meta {{ color:#656d76; font-size:13px; margin-bottom:20px; }}
.meta span {{ margin-right:18px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:12px; margin-bottom:24px; }}
.card {{ background:#fff; border:1px solid #d8dbdf; border-radius:8px; padding:16px; }}
.card .label {{ font-size:12px; color:#656d76; margin-bottom:6px; }}
.card .value {{ font-size:28px; font-weight:600; color:#1f6feb; }}
.card .hint {{ font-size:11px; color:#8b949e; margin-top:4px; }}
h2 {{ font-size:16px; font-weight:600; margin:24px 0 10px; }}
table {{ width:100%; border-collapse:collapse; background:#fff;
  border:1px solid #d8dbdf; border-radius:8px; overflow:hidden;
  font-size:13px; }}
th {{ background:#f6f8fa; padding:9px 10px; text-align:left;
  font-weight:600; border-bottom:1px solid #d8dbdf; white-space:nowrap; }}
td {{ padding:8px 10px; border-bottom:1px solid #eef0f2; vertical-align:top; }}
tr:last-child td {{ border-bottom:none; }}
tr.fail {{ background:#fff8f8; }}
.mono {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace;
  font-size:12px; color:#656d76; }}
.badge {{ display:inline-block; padding:2px 7px; border-radius:10px;
  font-size:11px; font-weight:600; }}
.badge.pass {{ background:#dafbe1; color:#1a7f37; }}
.badge.fail {{ background:#ffebe9; color:#cf222e; }}
.reason {{ color:#656d76; font-size:12px; max-width:260px; }}
.note {{ background:#fff8c5; border:1px solid #d4a72c; border-radius:8px;
  padding:12px 14px; font-size:12.5px; margin-bottom:20px; color:#633c01; }}
</style>
</head>
<body>
<div class="wrap">
<h1>LLM 质量评测报告</h1>
<div class="meta">
  <span>模型：{esc(summary['model'])}</span>
  <span>时间：{esc(summary['eval_time'])}</span>
  <span>用例：{summary['total_cases']} 条</span>
  <span>耗时：{summary['duration_sec']} 秒</span>
</div>

<div class="note">
  评测对象：自建 RAG 问答系统。域内用例检验准确性，域外用例检验幻觉倾向
  （无依据时能否正确拒答），模糊用例检验过度生成倾向。幻觉检测采用基于证据的
  实词覆盖率判定，阈值为 {THRESHOLD_HALLUCINATION}。
</div>

<div class="cards">{cards_html}</div>

<h2>分类统计</h2>
<table>
<thead><tr><th>类别</th><th>用例数</th><th>通过</th><th>通过率</th></tr></thead>
<tbody>{cat_rows}</tbody>
</table>

<h2>输出稳定性</h2>
<table>
<thead><tr><th>测试问题</th><th>稳定性得分</th><th>作答次数</th></tr></thead>
<tbody>{stab_rows}</tbody>
</table>

<h2>用例明细</h2>
<table>
<thead><tr>
<th>结果</th><th>用例编号</th><th>问题</th><th>类别</th>
<th>无幻觉</th><th>完整性</th><th>相关性</th><th>检索片段</th><th>判定说明</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def save_csv(summary):
    """
    导出 CSV 报告
    便于业务方在 Excel 中查看与二次分析，
    也便于接入 BI/看板系统做质量趋势监控。
    """
    import csv

    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "eval_report.csv")

    cases = summary["cases"]
    fieldnames = [
        "case_id", "category", "question", "actual_output",
        "expected_output", "hallucination_free", "hallucination_reason",
        "coverage", "is_complete", "completeness", "missing_points",
        "relevance", "retrieval_count", "passed",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in cases:
            row = {k: c.get(k, "") for k in fieldnames}
            row["missing_points"] = " | ".join(c.get("missing_points", []))
            writer.writerow(row)

    return path


def save_all(summary):
    """按 config.yaml 的 report.formats 输出各类报告"""
    formats = cfg_get(_CFG, "report.formats", ["json", "html"])
    paths = {}
    if "json" in formats:
        paths["json"] = save_json(summary)
    if "html" in formats:
        paths["html"] = save_html(summary)
    if "csv" in formats:
        paths["csv"] = save_csv(summary)
    return paths


if __name__ == "__main__":
    summary = run_all_eval()
    print_summary(summary)
    paths = save_all(summary)
    for fmt, p in paths.items():
        print(f"{fmt.upper():>5} 报告：{p}")
