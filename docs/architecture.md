# 架构说明

## 定位

面向 RAG 问答系统的**自动化质量评测与质量门禁系统**。

一句话：给定一个 RAG 系统和评测数据集，自动跑完整评测链路，量化输出质量，不达标时阻断 CI。

## 核心链路

```
测试数据集 Dataset
    ↓
被测系统 SUT (Provider)
    ↓
模型输出 (answer + contexts + latency + token)
    ↓
多维 Evaluator
    ├── Faithfulness  声明级幻觉检测
    ├── Correctness   正确性 / 完整性
    ├── Relevance     相关性
    ├── Refusal       拒答准确性
    └── Semantic / Judge  (可选增强)
    ↓
指标汇总 Metrics
    ↓
Quality Gate (分层门禁)
    ↓
Report (HTML / JSON / CSV)
    ↓
GitHub Actions (阻断质量回归)
```

## 分层设计

系统严格区分以下 8 个概念，边界不可混淆：

| 概念 | 载体 | 说明 |
|---|---|---|
| **被测系统 SUT** | `eval/providers/sut.py` → `Provider` | 被评测的对象，抽象为接口 |
| **评测框架** | `eval/evaluators/` | 独立于 SUT 实现 |
| **评测数据集** | `eval/datasets/` | smoke / full / regression_bugs |
| **Ground Truth** | `GroundTruth` | 参考答案 + 必需/禁止事实 |
| **指标** | `eval/quality_gate/gate.py` | 定义与阈值分离 |
| **质量门禁** | `QualityGate` | 分层判定 + 诊断信息 |
| **报告** | `eval/reporting/` | 三格式互补 |
| **CI/CD** | `.github/workflows/` | 触发策略 + 阻断 |

**解耦的验证方式**：`eval/providers/mock.py` 可以在没有任何 API 的情况下替换 SUT，评测框架代码零改动。

## 目录结构

```
eval/
├── schemas/
│   ├── dataset.py          Dataset / EvalCase / GroundTruth
│   └── result.py           CaseResult / EvalReport / TokenUsage
├── datasets/
│   ├── __init__.py         注册表 + 组合加载
│   ├── smoke.py            8 条 · CI 快速回归
│   ├── full.py             19 条 · 12 类完整覆盖
│   ├── regression.py       16 条 · 历史缺陷档案
│   └── gold_set.py         30 条 · 人工标注验证集
├── providers/
│   ├── sut.py              Provider 协议 + RAGProvider
│   └── mock.py             故障注入（12 种场景）
├── evaluators/
│   ├── faithfulness.py     声明级幻觉检测
│   ├── correctness.py      正确性/完整性/相关性/拒答/冲突
│   ├── semantic.py         Embedding 语义评测
│   ├── judge.py            LLM-as-a-Judge
│   └── validation.py       评测器可靠性验证
├── quality_gate/
│   └── gate.py             分层门禁 + 诊断
├── reporting/
│   ├── html_report.py      可视化报告
│   └── csv_report.py       明细导出
├── cli/
│   └── main.py             命令行入口
├── runner.py               评测编排
├── baseline.py             基线对比
└── mutation.py             变异测试

configs/
└── quality_gate.yaml       门禁阈值配置

tests/
├── unit/                   Schema 单元测试
├── evaluators/             各 Evaluator 单元测试
├── regression/             缺陷回归测试
└── test_quality_gate.py    旧门禁（保留兼容）
```

## 关键设计决策

### 1. 三态而非两态

```python
Status.PASSED / Status.FAILED / Status.ERROR
```

**原因**：旧实现只有"通过/不通过"，导致 API 失败时 `actual_output` 为空串，而幻觉检测首行 `if not output.strip(): return True` 把空串判为"无幻觉"→ **API 失败被当成通过**。

`error` 状态不计入通过率分母——因为 error 是"评测未完成"，不是"质量表现"。

### 2. 声明级幻觉检测

**能力对比（实测）**：

| 场景 | 旧字符覆盖率 | 新声明级 |
|---|---|---|
| 正确回答 | 无幻觉 | 无幻觉 |
| 多说一层 | 无幻觉 ❌ | **有幻觉** ✅ |
| 说错层 | 无幻觉 ❌ | **有幻觉** ✅ |
| 完全编造 | 无幻觉 ❌ | **有幻觉** ✅ |
| 数字编造 | 无幻觉 ❌ | **有幻觉** ✅ |
| 编造外国人名 | 无幻觉 ❌ | **有幻觉** ✅ |
| 完全跑题 | 无幻觉 ❌ | **有幻觉** ✅ |

**原理**：把回答拆成原子声明，逐条验证其是否被上下文支撑，而非整段算覆盖率。

### 3. 三种评测方法并存

| 方法 | 优势 | 局限 | 定位 |
|---|---|---|---|
| Lexical | 快、零成本、确定性 | 同义改写与实体调换是盲区 | Baseline |
| Semantic | 能识别同义改写 | 粒度粗，无法定位具体声明 | 辅助 |
| LLM Judge | 语义理解能力强 | 有偏差、成本高 | 交叉验证 |

**三者不混算**。`final_score` 只在需要时给出，各方法分数独立记录。

实测分歧（同义改写场景）：Lexical 0.47 / Semantic 0.81 —— **存在结论相反的样本，说明词级方法确实失效**。

### 4. 三层解耦

| 层 | 载体 | 维护方 | 换业务时 |
|---|---|---|---|
| 配置 | `config.yaml` | 测试/开发 | 改配置 |
| 数据 | `knowledge/` `cases.csv` | 业务方 | 替换数据 |
| 代码 | `app/` `eval/` | 开发 | 不改动 |

### 5. 缺陷闭环

```
Bug 发现 → 复现 → 固化用例 → 修复 → 回归测试 → CI 保护
```

16 条历史缺陷全部记录了 `discovered_in` / `root_cause` / `fix_summary`，并按 `layer` 分类（evaluator_bug / method_limit / env_issue）指明应由哪个指标验证。

## 已知局限

| 局限 | 影响 | 改进方向 |
|---|---|---|
| 声明级验证对"对比型回答"仍误报 | 跨上下文整合的句子被判无依据 | 子句拆分后分别匹配 |
| 拉丁词提取对专有名词覆盖有限 | 罕见名词可能漏检 | 引入 Embedding |
| 冲突检测依赖 SVO 语序 | 主谓倒装场景漏检 | 语义级冲突判定 |
| Judge 与被测系统同模型 | 存在自我偏好，未验证 | 交叉验证不同 Judge |
| Gold Set 无独立第三方标注 | 评测器指标存在循环论证 | 他人标注 + 200+ 样本 |
| 检索仍是关键词匹配 | 命中率 68% 是瓶颈 | Embedding 检索 |
| 无 branch protection | CI 失败不能真正阻止合并 | 仓库设置中启用 |

## 扩展点

新增评测器只需实现接口：

```python
def my_evaluator(answer, context, ground_truth) -> dict:
    return {"score": 0.0, "is_passed": False, "reason": ""}
```

新增 SUT 只需实现 Protocol：

```python
class MyProvider:
    name = "my_sut"
    def ask(self, question) -> SUTResponse: ...
    def is_available(self) -> bool: ...
```

两者都无需修改评测框架其他部分。
