"""
评测 CLI
---------------------------------
统一入口，提供 run / report / gate / case 四个子命令。

设计原则
--------
1. **失败必须给出可执行信息**：退出码非 0 时，
   明确说清哪个指标失败、当前值、阈值、失败样本
2. **配置缺失要明确区分**：无API key 时报 CONFIGURATION_ERROR，
   不伪装成"测试通过"（需求第二十条）
3. **--help 完整**：任何复杂工具都要能自解释

子命令
------
    run      执行评测
    report   生成报告
    gate     只跑质量门禁
    case     查看单条用例详情
    list     列出可用数据集
    validate 验证数据集合法性
"""

import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 退出码约定
EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_CONFIG_ERROR = 2
EXIT_DATA_ERROR = 3
EXIT_RUNTIME_ERROR = 4


from eval import env_loader


# ============================================================
# 配置检查
# ============================================================
def check_config(require_api: bool = True) -> list:
    """
    检查配置完整性，返回缺失项列表。

    只检查 API Key：base_url 与 model_name 有内置默认值，
    缺失时框架会用默认值继续工作，不该把这种情况当成配置错误。
    """
    if not require_api:
        return []
    return [name for name, _ in env_loader.missing_required()]


def print_config_error(missing: list):
    """输出配置错误说明"""
    print("=" * 66, file=sys.stderr)
    print("CONFIGURATION ERROR", file=sys.stderr)
    print("=" * 66, file=sys.stderr)
    print(f"缺少配置项：{', '.join(missing)}", file=sys.stderr)
    print("", file=sys.stderr)
    print("推荐做法（只对本项目生效，不影响系统其他工具）：", file=sys.stderr)
    print(f"  1. 复制 {os.path.join(PROJECT_ROOT, 'env.example')} 为 "
          f"{env_loader.ENV_FILE}", file=sys.stderr)
    print(f"  2. 打开该文件，填入 {', '.join(missing)}", file=sys.stderr)
    print("", file=sys.stderr)
    print("若你的电脑上还有别的工具也在用 OPENAI_* 这组变量名", file=sys.stderr)
    print("（某些 API 切换类工具会写同名的系统环境变量），", file=sys.stderr)
    print(f"用上面这种 .env 方式可以彻底避免互相覆盖。", file=sys.stderr)
    print("", file=sys.stderr)
    print("如需在无 API 环境下验证评测器本身，可加 --mock 参数：", file=sys.stderr)
    print("  python -m eval run --dataset smoke --mock", file=sys.stderr)
    print("=" * 66, file=sys.stderr)


def cmd_config(args):
    """查看当前生效的配置（密钥脱敏）"""
    info = env_loader.describe()
    print("=" * 66)
    print("当前 API 配置")
    print("=" * 66)
    print(f"  配置文件      {info['env_file']}"
          f"  {'（已存在）' if info['env_file_exists'] else '（不存在）'}")
    print(f"  生效的变量名  {info['api_key_env']}")
    print(f"  API Key       {info['api_key']}")
    print(f"  接口地址      {info['base_url']}")
    print(f"  模型名        {info['model_name']}")
    print(f"  Judge 模型    {info['judge_model_name']}")
    if info["missing"]:
        print(f"  缺失项        {', '.join(info['missing'])}")
    print("=" * 66)
    return EXIT_OK if not info["missing"] else EXIT_CONFIG_ERROR


# ============================================================
# 子命令
# ============================================================
def cmd_list(args):
    """列出可用数据集"""
    from eval.datasets import list_datasets, validate_all

    print("=" * 66)
    print("可用数据集")
    print("=" * 66)
    for name, info in list_datasets().items():
        print(f"\n  {name:<20} {info['case_count']} 条")
        print(f"  {'':<20} {info['description']}")
        if info.get("categories"):
            print(f"  {'':<20} 类别：{', '.join(info['categories'])}")

    print("\n" + "=" * 66)
    print("用法示例")
    print("=" * 66)
    print("  python -m eval list")
    print("  python -m eval validate --dataset smoke")
    print("  python -m eval run --dataset smoke")
    print("  python -m eval run --dataset full --mock   # 无需 API")
    print("  python -m eval run --dataset smoke --judge   # 启用 Judge")
    print("  python -m eval gate --dataset smoke")
    print("  python -m eval case --id smoke_in_01")
    return EXIT_OK


def cmd_validate(args):
    """验证数据集"""
    from eval.datasets import get_dataset, validate_dataset

    try:
        ds = get_dataset(args.dataset)
    except KeyError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return EXIT_DATA_ERROR

    issues = validate_dataset(ds)
    print(f"数据集：{ds.name}（{ds.tier}）  用例 {len(ds)} 条")

    if not issues:
        print("校验通过，无问题。")
        return EXIT_OK

    print(f"\n发现 {len(issues)} 个问题：")
    for i in issues:
        print(f"  - {i}")
    return EXIT_DATA_ERROR


def cmd_run(args):
    """执行评测"""
    from eval.datasets import get_dataset

    # 配置检查（--mock 模式不需要 API）
    if not args.mock:
        missing = check_config(require_api=True)
        if missing:
            if args.json:
                print(json.dumps({
                    "status": "configuration_error",
                    "missing": missing,
                }, ensure_ascii=False))
            else:
                print_config_error(missing)
            return EXIT_CONFIG_ERROR

    try:
        ds = get_dataset(args.dataset)
    except KeyError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return EXIT_DATA_ERROR

    from eval.runner import run_evaluation

    try:
        result = run_evaluation(
            dataset=ds,
            use_judge=args.judge,
            use_semantic=args.semantic,
            use_mock=args.mock,
            stability_repeat=args.stability_repeat,
        )
    except Exception as e:
        print(f"[ERROR] 评测执行失败：{type(e).__name__}: {e}",
              file=sys.stderr)
        return EXIT_RUNTIME_ERROR

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.report.text())

    if args.output:
        path = result.save_all(args.output)
        if not args.json:
            print(f"\n报告已保存：")
            for k, v in path.items():
                print(f"  {k.upper():>5}: {v}")

    # ---- 基线管理 ----
    from eval.baseline import (
        compare, save_baseline, load_baseline, extract_metrics,
    )

    current_metrics = {
        k: v.value for k, v in result.report.metrics.items() if v.available
    }

    exit_code = EXIT_OK if result.report.gate_passed else EXIT_GATE_FAILED

    # 顺序至关重要：必须先把「上一版基线」读出来，再决定是否覆盖它。
    #
    # 反过来写（先save 后 load）会拿本次结果跟自己比，差值恒为 0，
    # 于是无论质量退化多严重都输出「无退化」——
    # 这是整个基线机制里最隐蔽的一种假通过，必须避免。
    previous_baseline = load_baseline() if args.compare_baseline else None

    if args.compare_baseline:
        if not previous_baseline:
            if not args.json:
                print(f"\n[ERROR] 无基线可对比，请先运行 --save-baseline",
                      file=sys.stderr)
            return EXIT_DATA_ERROR

        current_meta = {"model": result.report.model,
                        "dataset": result.report.dataset_name}

        cmp = compare(
            current_metrics, previous_baseline.get("metrics", {}),
            tolerance=args.tolerance,
            baseline_source=previous_baseline.get("meta", {}).get(
                "time", "unknown"),
            baseline_meta=previous_baseline.get("meta", {}),
            current_meta=current_meta,
        )
        print()
        print(cmp.text())

        # 只有在确实可比、且真的检出退化时才判失败。
        # 不可比时 has_regression 恒为 False —— 拿 mock 基线去比真实结果
        # 必然显示「大幅退化」，那不是退化，那是数据来源不同。
        if cmp.has_regression:
            exit_code = EXIT_GATE_FAILED

    # 保存放在对比之后：本次结果成为「下一版」的基线，
    # 但不会污染本次的对比对象。
    if args.save_baseline:
        p = save_baseline(
            current_metrics,
            meta={"model": result.report.model,
                  "dataset": result.report.dataset_name,
                  "time": result.report.started_at},
        )
        if not args.json:
            print(f"\n基线已保存：{p}")

    return exit_code


def cmd_report(args):
    """生成报告（从已有结果）"""
    from eval.reporting import load_and_render

    try:
        out = load_and_render(args.input, args.output)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return EXIT_DATA_ERROR
    except Exception as e:
        print(f"[ERROR] 报告生成失败：{type(e).__name__}: {e}",
              file=sys.stderr)
        return EXIT_RUNTIME_ERROR

    print("报告已生成：")
    for k, v in out.items():
        print(f"  {k.upper():>5}: {v}")
    return EXIT_OK


def cmd_gate(args):
    """只跑质量门禁"""
    from eval.reporting import load_results
    from eval.quality_gate.gate import QualityGate, load_gate_config

    try:
        report = load_results(args.input)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return EXIT_DATA_ERROR

    if not report.get("metrics"):
        print("[ERROR] 结果文件中无 metrics，无法评估门禁", file=sys.stderr)
        return EXIT_DATA_ERROR

    gate = QualityGate(load_gate_config(args.config))
    gate_report = gate.evaluate(
        report["metrics"],
        failed_samples=report.get("failed_samples"),
    )
    print(gate_report.text())
    return EXIT_OK if gate_report.passed else EXIT_GATE_FAILED


def cmd_case(args):
    """查看单条用例详情"""
    from eval.datasets import get_dataset
    from eval.evaluators.faithfulness import detect_hallucination

    try:
        ds = get_dataset(args.dataset)
    except KeyError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return EXIT_DATA_ERROR

    case = ds.get(args.id)
    if case is None:
        print(f"[ERROR] 未找到用例 {args.id}", file=sys.stderr)
        return EXIT_DATA_ERROR

    print("=" * 66)
    print(f"用例：{case.id}")
    print("=" * 66)
    print(f"类别    {case.category}")
    print(f"难度    {case.difficulty}")
    print(f"目的    {case.purpose}")
    print(f"期望行为 {case.expected_behavior}")
    print(f"标签    {', '.join(case.tags) or '无'}")
    print(f"\n问题    {case.question}")

    gt = case.ground_truth
    if gt.reference_answer:
        print(f"参考答案{gt.reference_answer}")
    if gt.required_facts:
        print(f"必需事实{', '.join(gt.required_facts)}")
    if gt.forbidden_facts:
        print(f"禁止事实{', '.join(gt.forbidden_facts)}")
    if case.context:
        print(f"指定上下文{chr(10).join(case.context)}")

    if args.detect:
        print(f"\n{'-' * 66}")
        print("演示：给定一段错误回答，评测器如何判定")
        print("-" * 66)
        ctx = "\n\n".join(case.context) or "（无上下文）"
        fake = args.detect
        r = detect_hallucination(
            fake, ctx,
            gt.required_facts, gt.forbidden_facts,
            should_refuse=case.should_refuse,
        )
        print(f"判定    {'有幻觉' if r['has_hallucination'] else '无幻觉'}")
        print(f"理由    {r['reason']}")
        for c in r["claims"]:
            mark = "OK" if c.supported else "NG"
            print(f"  [{mark}] {c.text[:44]}")
            print(f"       {c.reason[:60]}")

    return EXIT_OK


# ============================================================
# 参数解析
# ============================================================
def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m eval",
        description="LLM/RAG 自动化质量评测与质量门禁系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python -m eval list                      列出可用数据集
  python -m eval validate --dataset full   验证数据集合法性
  python -m eval run --dataset smoke       执行评测
  python -m eval run --dataset full --mock 无需 API 的离线评测
  python -m eval run --dataset smoke --judge --semantic   全指标评测
  python -m eval report --input reports/eval_results.json
  python -m eval gate --input reports/eval_results.json
  python -m eval case --id smoke_in_01 --detect "测试用的错误回答"

退出码：
  0  成功 / 门禁通过
  1  门禁未通过
  2  配置错误（缺少环境变量）
  3  数据错误（数据集或用例不存在）
  4  运行时错误
        """,
    )

    sub = p.add_subparsers(dest="command", metavar="<command>")

    # list
    p_list = sub.add_parser("list", help="列出可用数据集")
    p_list.set_defaults(func=cmd_list)

    # validate
    p_val = sub.add_parser("validate", help="验证数据集合法性")
    p_val.add_argument("--dataset", default="smoke", help="数据集名称")
    p_val.set_defaults(func=cmd_validate)

    # config
    p_cfg = sub.add_parser("config", help="查看当前生效的 API 配置（密钥脱敏）")
    p_cfg.set_defaults(func=cmd_config)

    # run
    p_run = sub.add_parser("run", help="执行评测")
    p_run.add_argument("--dataset", default="smoke",
                       help="数据集名称（smoke / full / smoke+full）")
    p_run.add_argument("--mock", action="store_true",
                       help="使用 Mock Provider，无需 API")
    p_run.add_argument("--judge", action="store_true",
                       help="启用 LLM Judge（成本较高）")
    p_run.add_argument("--semantic", action="store_true",
                       help="启用 Embedding 语义评测（成本较高）")
    p_run.add_argument("--stability-repeat", type=int, default=3,
                       help="稳定性测试重复次数")
    p_run.add_argument("--output", default="reports",
                       help="报告输出目录")
    p_run.add_argument("--json", action="store_true",
                       help="以JSON 输出到 stdout")
    p_run.add_argument("--save-baseline", action="store_true",
                       help="将当前指标保存为基线")
    p_run.add_argument("--compare-baseline", action="store_true",
                       help="与基线对比，检测退化")
    p_run.add_argument("--tolerance", type=float, default=0.05,
                       help="退化容忍度（相对变化，默认 5%%）")
    p_run.set_defaults(func=cmd_run)

    # report
    p_rep = sub.add_parser("report", help="从已有结果生成报告")
    p_rep.add_argument("--input", default="reports/eval_results.json",
                       help="结果文件路径")
    p_rep.add_argument("--output", default="reports",
                       help="报告输出目录")
    p_rep.set_defaults(func=cmd_report)

    # gate
    p_gate = sub.add_parser("gate", help="执行质量门禁评估")
    p_gate.add_argument("--input", default="reports/eval_results.json",
                        help="结果文件路径")
    p_gate.add_argument("--config", default="configs/quality_gate.yaml",
                        help="门禁配置路径")
    p_gate.set_defaults(func=cmd_gate)

    # case
    p_case = sub.add_parser("case", help="查看单条用例详情")
    p_case.add_argument("--id", required=True, help="用例 ID")
    p_case.add_argument("--dataset", default="full", help="数据集名称")
    p_case.add_argument("--detect", metavar="ANSWER",
                        help="对给定回答运行评测器，展示判定过程")
    p_case.set_defaults(func=cmd_case)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_OK

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] 用户中断", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())
