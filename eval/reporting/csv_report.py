"""
CSV 报告生成
---------------------------------
导出逐用例明细，便于业务方在 Excel 中筛选、排序、二次分析。

为什么同时提供 JSON / HTML / CSV
--------------------------------
    JSON  机器读，供 CI 对比历史
    HTML  人读，含诊断与可视化
    CSV   Excel 打开，供业务方分析
"""

import os
import csv
from typing import Any, Dict, List

FIELDNAMES = [
    "case_id", "status", "category", "difficulty", "expected_behavior",
    "question", "actual_output", "reference_answer",
    "retrieval_context_count",
    "faithfulness", "correctness", "completeness", "relevance", "refusal",
    "semantic_groundedness",
    "violation_types", "violation_messages", "severities",
    "latency_ms", "input_tokens", "output_tokens", "total_tokens", "cost",
    "error", "evidence",
]


def save_csv(report, output_dir: str) -> str:
    """生成 CSV 明细报告"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "eval_report.csv")

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for c in report.cases:
            row: Dict[str, Any] = {
                "case_id": c.case_id,
                "status": c.status,
                "category": c.category,
                "difficulty": c.difficulty,
                "expected_behavior": c.expected_behavior,
                "question": _clean(c.question),
                "actual_output": _clean(c.actual_output),
                "reference_answer": _clean(c.reference_answer),
                "retrieval_context_count": len(c.retrieval_context),
                "violation_types": "; ".join(c.violation_types()),
                "violation_messages": " | ".join(
                    v.message for v in c.violations[:5]),
                "severities": "; ".join(
                    v.severity for v in c.violations),
                "latency_ms": c.latency_ms,
                "error": c.error or "",
                "evidence": " | ".join(c.evidence[:5]),
            }

            for k in ("faithfulness", "correctness", "completeness",
                      "relevance", "refusal", "semantic_groundedness"):
                row[k] = c.scores.get(k, "")

            tu = c.token_usage
            row["input_tokens"] = tu.input_tokens if tu.available else ""
            row["output_tokens"] = tu.output_tokens if tu.available else ""
            row["total_tokens"] = tu.total_tokens if tu.available else ""
            row["cost"] = tu.cost if tu.available else ""

            writer.writerow(row)

    return path


def _clean(s) -> str:
    """清理文本：去掉换行，避免 CSV 串行"""
    if s is None:
        return ""
    return str(s).replace("\r", " ").replace("\n", " ")


def save_metrics_csv(report, output_dir: str) -> str:
    """导出指标汇总表（便于趋势监控）"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "metrics.csv")

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "available",
                         "gate_passed", "threshold_min", "threshold_max",
                         "eval_time", "model", "dataset"])

        for name, sm in report.metrics.items():
            writer.writerow([
                name, sm.value, sm.available, sm.passed,
                sm.threshold_min, sm.threshold_max,
                report.started_at, report.model,
                f"{report.dataset_name}({report.dataset_tier})",
            ])

    return path
