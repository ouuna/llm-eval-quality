"""
报告生成
---------------------------------
三种格式覆盖不同用途：
    HTML  人读，含诊断与可视化
    JSON  机器读，供 CI 对比历史
    CSV   Excel 打开，供业务方分析
"""

import json
import os
from typing import Any, Dict, Optional

from eval.reporting.html_report import save_html
from eval.reporting.csv_report import save_csv, save_metrics_csv

__all__ = ["save_html", "save_csv", "save_metrics_csv",
           "load_results", "load_and_render", "extract_baseline"]


def load_results(path: str) -> Dict[str, Any]:
    """从 JSON 报告文件加载结果"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"结果文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_baseline(results: Dict[str, Any]) -> Dict[str, float]:
    """从历史结果中提取基线指标"""
    out = {}
    for name, m in (results.get("metrics") or {}).items():
        v = m.get("value") if isinstance(m, dict) else None
        if isinstance(v, (int, float)):
            out[name] = v
    return out


def load_and_render(input_path: str, output_dir: str) -> Dict[str, str]:
    """
    从已有 JSON 结果重新生成报告

    用途：调整报告模板后无需重跑评测
    """
    from eval.schemas.result import EvalReport, CaseResult, TokenUsage

    data = load_results(input_path)
    baseline = extract_baseline(data)

    # 重建 EvalReport
    from eval.schemas.result import MetricSummary
    report = EvalReport(
        dataset_name=data["meta"]["dataset"],
        dataset_tier=data["meta"]["tier"],
        model=data["meta"]["model"],
        provider=data["meta"].get("provider", "unknown"),
        started_at=data["meta"].get("started_at", ""),
        duration_sec=data["meta"].get("duration_sec", 0),
    )

    for cd in data.get("cases", []):
        from eval.schemas.result import Violation
        c = CaseResult(
            case_id=cd["case_id"], status=cd["status"],
            scores=cd.get("scores") or {},
            evidence=cd.get("evidence") or [],
            latency_ms=cd.get("latency_ms", 0),
            error=cd.get("error"),
            category=cd.get("category", ""),
            question=cd.get("question", ""),
            expected_behavior=cd.get("expected_behavior", ""),
            actual_output=cd.get("actual_output", ""),
            reference_answer=cd.get("reference_answer"),
            retrieval_context=cd.get("retrieval_context") or [],
            difficulty=cd.get("difficulty", "medium"),
            tags=cd.get("tags") or [],
        )
        tu = cd.get("token_usage") or {}
        c.token_usage = TokenUsage(
            input_tokens=tu.get("input_tokens"),
            output_tokens=tu.get("output_tokens"),
            total_tokens=tu.get("total_tokens"),
            cost=tu.get("cost"),
            available=tu.get("available", False),
        )
        for vd in cd.get("violations") or []:
            c.violations.append(Violation(
                type=vd["type"], severity=vd["severity"],
                message=vd["message"], evidence=vd.get("evidence") or [],
                details=vd.get("details") or {},
            ))
        report.cases.append(c)

    for name, m in (data.get("metrics") or {}).items():
        report.metrics[name] = MetricSummary(
            name=name, value=m.get("value", 0),
            available=m.get("available", True),
            definition=m.get("definition", ""),
            caveat=m.get("caveat", ""),
        )

    gate = data.get("gate") or {}
    report.gate_passed = gate.get("passed", True)
    report.gate_failures = gate.get("failures") or []
    report.gate_message = gate.get("message", "")

    return {
        "html": save_html(report, output_dir, baseline=baseline),
        "csv": save_csv(report, output_dir),
        "metrics": save_metrics_csv(report, output_dir),
    }
