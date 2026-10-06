"""
HTML 报告生成（零依赖）
---------------------------------
从 EvalReport 生成自包含的可视化报告。

设计要点
--------
1. **报告要能回答三个问题**：
   - 整体质量如何？（指标卡）
   - 哪些用例失败了，为什么？（失败清单+ 诊断）
   - 比上次退化了还是改善了？（与 baseline 对比）
2. **失败样本必须可定位**：id、问题、回答、参考、判定理由
3. **不依赖任何第三方库**：手写 HTML + 内联 CSS
"""

import os
import json
import html as html_mod
from typing import Any, Dict, List, Optional


def _esc(s: Any) -> str:
    """HTML 转义"""
    return html_mod.escape(str(s) if s is not None else "")


CSS = """
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
  background:#f6f7f9; color:#1f2328; padding:24px; line-height:1.6; }
.wrap { max-width:1280px; margin:0 auto; }
h1 { font-size:24px; font-weight:600; margin-bottom:4px; }
h2 { font-size:16px; font-weight:600; margin:28px 0 10px;
  padding-bottom:8px; border-bottom:1px solid #d8dbdf; }
.sub { color:#656d76; font-size:13px; margin-bottom:20px; }
.sub span { margin-right:16px; }

/* 门禁横幅 */
.banner { padding:14px 18px; border-radius:8px; margin-bottom:24px;
  font-size:14px; font-weight:500; }
.banner.pass { background:#dafbe1; color:#1a7f37; border:1px solid #4ac26b; }
.banner.fail { background:#ffebe9; color:#cf222e; border:1px solid #ff8182; }

/* 指标卡 */
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
  gap:12px; margin-bottom:8px; }
.card { background:#fff; border:1px solid #d8dbdf; border-radius:8px;
  padding:14px 16px; }
.card .label { font-size:12px; color:#656d76; margin-bottom:4px; }
.card .value { font-size:26px; font-weight:600; }
.card .hint { font-size:11px; color:#8b949e; margin-top:3px; }
.card.ok .value { color:#1a7f37; }
.card.bad .value { color:#cf222e; }
.card.na .value { color:#8b949e; font-size:18px; }

/* 表格 */
table { width:100%; border-collapse:collapse; background:#fff;
  border:1px solid #d8dbdf; border-radius:8px; overflow:hidden;
  font-size:13px; margin-bottom:8px; }
th { background:#f6f8fa; padding:9px 10px; text-align:left;
  font-weight:600; border-bottom:1px solid #d8dbdf; white-space:nowrap; }
td { padding:8px 10px; border-bottom:1px solid #eef0f2; vertical-align:top; }
tr:last-child td { border-bottom:none; }
tr.row-fail { background:#fff8f8; }
tr.row-error { background:#fff4e5; }
.mono { font-family:ui-monospace,SFMono-Regular,Consolas,monospace;
  font-size:12px; color:#656d76; white-space:nowrap; }

/* 徽章 */
.badge { display:inline-block; padding:2px 8px; border-radius:10px;
  font-size:11px; font-weight:600; }
.badge.pass { background:#dafbe1; color:#1a7f37; }
.badge.fail { background:#ffebe9; color:#cf222e; }
.badge.error { background:#fff1e5; color:#bc4c00; }

/* 违反标签 */
.vt { display:inline-block; padding:1px 6px; border-radius:4px;
  font-size:11px; margin:1px 2px; background:#ffebe9; color:#cf222e; }

/* 详情块 */
.detail { background:#fff; border:1px solid #d8dbdf; border-radius:8px;
  padding:12px 14px; margin-bottom:8px; }
.detail .q { font-weight:500; margin-bottom:6px; }
.detail .ans { background:#f6f8fa; border-radius:6px; padding:8px 10px;
  margin:4px 0; font-size:12.5px; white-space:pre-wrap; }
.detail .meta { color:#656d76; font-size:12px; }

/* 基线对比 */
.delta { font-size:12px; font-weight:500; }
.delta.up { color:#cf222e; }
.delta.down { color:#1a7f37; }
.delta.flat { color:#8b949e; }
.note { background:#fff8c5; border:1px solid #d4a72c; border-radius:8px;
  padding:10px 12px; font-size:12.5px; margin-bottom:16px; color:#633c01; }
.pareto { display:flex; align-items:center; gap:8px; margin-bottom:4px; }
.pareto .bar { height:16px; background:#ff8182; border-radius:3px; }
.pareto .label { font-size:12px; min-width:200px; }
.pareto .num { font-size:12px; color:#656d76; }
"""


def save_html(report, output_dir: str, baseline: Dict[str, Any] = None) -> str:
    """生成 HTML 报告"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "report.html")

    m = report.metrics
    s = report.summary() if hasattr(report, "summary") else {}
    total = len(report.cases)
    passed = sum(1 for c in report.cases if c.is_passed)
    failed = sum(1 for c in report.cases if c.is_failed)
    errors = sum(1 for c in report.cases if c.is_error)

    # ---- 门禁横幅 ----
    gate_pass = report.gate_passed
    gate_fail_list = ""
    if not gate_pass and report.gate_failures:
        items = []
        for f in report.gate_failures:
            items.append(
                f"<li><code>{_esc(f.get('name'))}</code>："
                f"当前 {_esc(f.get('current_value'))}，"
                f"要求 {_esc(f.get('comparison'))} "
                f"{_esc(f.get('threshold'))}"
                + (f"（失败样本 {len(f.get('failed_samples') or [])} 个）"
                   if f.get("failed_sample_count") else "")
                + f"</li>")
        gate_fail_list = f"<ul style='margin:8px 0 0 20px'>{''.join(items)}</ul>"

    # ---- 指标卡 ----
    cards = []
    metric_defs = [
        ("overall_pass_rate", "总体通过率", "高为好"),
        ("hallucination_rate", "幻觉率", "低为好"),
        ("refusal_accuracy", "拒答准确率", "高为好"),
        ("faithfulness", "有据性", "高为好"),
        ("answer_correctness", "答案正确性", "高为好"),
        ("completeness", "完整率", "高为好"),
        ("relevance", "相关性", "高为好"),
        ("retrieval_hit_rate", "检索命中率", "高为好"),
        ("consistency", "输出一致性", "高为好"),
        ("score_std", "分数波动", "低为好"),
        ("p95_latency_ms", "P95延迟(ms)", "低为好"),
        ("avg_cost", "平均成本(元)", "低为好"),
    ]
    lower_better = {"hallucination_rate", "score_std", "p95_latency_ms",
                    "avg_cost"}

    for key, label, hint in metric_defs:
        sm = m.get(key)
        if sm is None or not sm.available:
            cards.append(
                f'<div class="card na"><div class="label">{_esc(label)}</div>'
                f'<div class="value">unavailable</div>'
                f'<div class="hint">{_esc(hint)}</div></div>')
            continue

        val = sm.value
        gate_passed = sm.passed
        cls = "ok" if gate_passed else ("bad" if gate_passed is False else "")

        # 基线对比
        delta_html = ""
        if baseline and key in baseline:
            old = baseline[key]
            if isinstance(old, (int, float)) and old != 0:
                diff = val - old
                pct = diff / abs(old) * 100
                arrow = "▲" if diff > 0 else "▼"
                dcls = ("down" if ((diff > 0) == (key not in lower_better))
                        else "up")
                delta_html = (
                    f'<div class="delta {dcls}">{arrow} {abs(pct):.1f}%</div>')

        disp = f"{val:.1%}" if val <= 1 and key not in (
            "p95_latency_ms", "avg_cost", "score_std") else f"{val:.4g}"
        cards.append(
            f'<div class="card {cls}"><div class="label">{_esc(label)}'
            f'{delta_html}</div><div class="value">{_esc(disp)}</div>'
            f'<div class="hint">{_esc(hint)}</div></div>')

    # ---- 缺陷 Pareto ----
    pareto_rows = ""
    if report.failure_pareto():
        mx = max(r["count"] for r in report.failure_pareto()) or 1
        for r in report.failure_pareto():
            w = int(r["count"] / mx * 260)
            pareto_rows += (
                f'<div class="pareto"><span class="label mono">'
                f'{_esc(r["type"])}</span>'
                f'<span class="bar" style="width:{w}px"></span>'
                f'<span class="num">{r["count"]} 条 · '
                f'{r["rate"]:.0%}（累计 {r["cumulative"]:.0%}）</span></div>')

    # ---- 分类统计 ----
    cat_rows = "".join(
        f'<tr><td class="mono">{_esc(c["category"])}</td>'
        f'<td>{c["total"]}</td><td>{c["passed"]}</td>'
        f'<td>{c["failed"]}</td><td>{c["error"]}</td>'
        f'<td>{c["pass_rate"]:.0%}</td></tr>'
        for c in report.category_stats())

    # ---- 失败用例详情 ----
    fail_details = ""
    for c in report.cases:
        if c.is_passed:
            continue
        vts = "".join(f'<span class="vt">{_esc(v)}</span>'
                      for v in c.violation_types())
        reasons = "<br>".join(
            f"<code>{_esc(v.severity)}</code> {_esc(v.message)}"
            for v in c.violations[:4])
        fail_details += f"""
<div class="detail">
  <div class="q">
    <span class="badge {'error' if c.is_error else 'fail'}">
      {_esc(c.status.upper())}</span>
    <span class="mono">{_esc(c.case_id)}</span>
    {vts}
  </div>
  <div class="meta">问题：{_esc(c.question)}</div>
  <div class="ans">{_esc(c.actual_output) or '（空回答）'}</div>
  {f'<div class="meta">参考答案：{_esc(c.reference_answer)}</div>'
   if c.reference_answer else ''}
  <div class="meta" style="margin-top:6px">{reasons}</div>
  {f'<div class="meta">检索上下文：{len(c.retrieval_context)} 段</div>'
   if c.retrieval_context else ''}
  {f'<div class="meta" style="color:#cf222e">错误：{_esc(c.error)}</div>'
   if c.error else ''}
</div>"""

    if not fail_details:
        fail_details = ('<div class="note">本次评测无失败用例。</div>')

    # ---- 全量明细 ----
    rows = []
    for c in report.cases:
        cls = ("row-fail" if c.is_failed
               else "row-error" if c.is_error else "")
        badge = ("pass" if c.is_passed
                 else "error" if c.is_error else "fail")
        scores = " ".join(
            f"{k.split('_')[-1]}={v:.2f}"
            for k, v in c.scores.items()
            if v is not None and isinstance(v, float))
        rows.append(
            f'<tr class="{cls}">'
            f'<td><span class="badge {badge}">{_esc(c.status[:5].upper())}</span></td>'
            f'<td class="mono">{_esc(c.case_id)}</td>'
            f'<td class="mono">{_esc(c.category)}</td>'
            f'<td>{_esc(c.question)}</td>'
            f'<td class="mono">{_esc(scores) or "-"}</td>'
            f'<td>{c.latency_ms}</td>'
            f'<td>{len(c.retrieval_context)}</td>'
            f'<td>{"; ".join(c.violation_types()[:2]) or "-"}</td>'
            "</tr>")

    # ---- 基线对比表 ----
    baseline_table = ""
    if baseline:
        bl_rows = []
        for key, sm in m.items():
            if not sm.available or key not in baseline:
                continue
            old = baseline[key]
            if not isinstance(old, (int, float)):
                continue
            diff = sm.value - old
            sign = "+" if diff > 0 else ""
            bl_rows.append(
                f'<tr class="{"row-error" if (diff > 0) == (key not in lower_better) and diff != 0 else ""}">'
                f'<td class="mono">{_esc(key)}</td>'
                f'<td>{old:.4g}</td><td>{sm.value:.4g}</td>'
                f'<td>{sign}{diff:.4g}</td></tr>')
        baseline_table = f"""
<h2>与基线对比</h2>
<table><thead><tr><th>指标</th><th>基线</th><th>当前</th><th>变化</th></tr></thead>
<tbody>{''.join(bl_rows)}</tbody></table>"""

    lat = report.latency_stats()
    tok = report.token_stats()

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM 质量评测报告</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<h1>LLM 质量评测报告</h1>
<div class="sub">
  <span>数据集：{_esc(report.dataset_name)}（{_esc(report.dataset_tier)}）</span>
  <span>模型：{_esc(report.model)}</span>
  <span>时间：{_esc(report.started_at)}</span>
  <span>耗时：{report.duration_sec}s</span>
  <span>评测器：{_esc(', '.join(report.evaluators))}</span>
</div>

<div class="banner {'pass' if gate_pass else 'fail'}">
  Quality Gate: {'PASSED' if gate_pass else 'FAILED'}
  — 通过 {passed} / 失败 {failed} / 错误 {errors}，共 {total} 条
  {gate_fail_list}
</div>

<h2>质量指标</h2>
<div class="cards">{''.join(cards)}</div>

<h2>运行统计</h2>
<table>
<thead><tr><th>维度</th><th>数值</th></tr></thead>
<tbody>
<tr><td>总用例数</td><td>{total}</td></tr>
<tr><td>通过 / 失败 / 错误</td><td>{passed} / {failed} / {errors}</td></tr>
<tr><td>延迟 mean / P50 / P95 / max</td>
<td>{_fmt(lat.get('mean'))} / {_fmt(lat.get('p50'))} /
    {_fmt(lat.get('p95'))} / {_fmt(lat.get('max'))} ms</td></tr>
<tr><td>Token 总量 / 均值</td>
<td>{_fmt(tok.get('total_tokens'))} / {_fmt(tok.get('avg_tokens'))}</td></tr>
<tr><td>成本合计 / 均值</td>
<td>{_fmt(tok.get('total_cost'))} / {_fmt(tok.get('avg_cost'))} 元</td></tr>
</tbody>
</table>

{baseline_table}

<h2>缺陷 Pareto 分布</h2>
{pareto_rows or '<div class="note">无缺陷记录。</div>'}

<h2>分类统计</h2>
<table>
<thead><tr><th>类别</th><th>总数</th><th>通过</th><th>失败</th><th>错误</th>
<th>通过率</th></tr></thead>
<tbody>{cat_rows}</tbody>
</table>

<h2>失败用例详情</h2>
{fail_details}

<h2>全量用例明细</h2>
<table>
<thead><tr><th>结果</th><th>ID</th><th>类别</th><th>问题</th>
<th>指标</th><th>延迟ms</th><th>上下文</th><th>违规类型</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def _fmt(v) -> str:
    """格式化数值，不可用时显示 unavailable"""
    return "unavailable" if v is None else f"{v}"
