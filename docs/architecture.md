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
├── env_loader.py            配置加载（.env + EVAL_* 隔离 + 脱敏自检）
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
├── test_env_loader.py      配置优先级与隔离回归
└── test_quality_gate.py    旧门禁（保留兼容）
```

项目根目录的 `env.example` 是配置模板，复制为 `.env` 后填入真实值。
`.env` 已被 `.gitignore` 排除。

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

### 6. 配置隔离：项目变量名 vs 通用变量名

`OPENAI_API_KEY` / `OPENAI_BASE_URL` 这组名字由 OpenAI 最早定义，
后来被所有兼容 OpenAI 协议的服务商沿用。本机上其他 AI 工具
（某些 CLI 的 Provider 切换器）也会往**系统环境变量**里写同名值，
共用名字必然互相覆写：你切一次供应商，评测就连错服务。

`eval/env_loader.py` 解决这一点：

| 手段 | 说明 |
|---|---|
| 项目专属变量名 | `EVAL_API_KEY` / `EVAL_BASE_URL` / `EVAL_MODEL_NAME`，与外部工具零重叠 |
| 项目内 `.env` | 配置只对本项目生效，不写入系统，无需重开终端 |
| 统一入口 | 所有模块都走 `env_loader`，不再各自 `os.getenv` |
| 自检命令 | `python -m eval config` 显示实际生效的变量名（密钥脱敏） |

取值优先级刻意把「`.env` 里的项目名」排在「环境变量里的通用名」之前：

```
进程环境变量 EVAL_*  >  .env 中 EVAL_*  >  进程环境变量 OPENAI_*  >  .env 中 OPENAI_*
```

如果反过来排，外部工具改一次系统环境变量就能污染本项目，变量改名等于白改。
这条顺序由 `tests/test_env_loader.py::TestPriority` 中的回归用例锁定。

向后兼容 `OPENAI_*` 是有意的：GitHub Actions 上已配置的 secrets
无需改动即可继续工作，CI 因此也在持续验证这条兼容路径。

### 7. CI 配置本身需要静态检查

Phase 13 写下的 `if: ${{ secrets.OPENAI_API_KEY != '' }}` 导致
**整个 workflow 文件被GitHub 作废**——不是某个 job 失败，而是
所有 job 都不会被创建（`jobs` API 返回 `total_count: 0`）。

**同一根因连续发作三次**，每次修法都对但认知不完整：

| # | 写法 | 为什么还错 |
|---|---|---|
| 1 | `if: ${{ secrets.X != '' }}` | job 级 if 不能用表达式包裹 |
| 2 | 注释里写 `${{ secrets.X }}` | **GitHub 解析注释里的表达式**，与代码同等对待 |
| 3 | `if: secrets.X != ''` | **secrets 在 job 级 if 完全不可用，裸表达式也不行** |

第 3 次才拿到真相：GitHub 在**解析文件阶段**就求值 job 级 `if`，
那时 `secrets` 尚未注入。正确做法是把密钥判断移到 **step 级**
（step 级 `if` 可用 `secrets`），本项目改为：

```yaml
- name: 密钥检查
  id: keycheck
  env:
    KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    if [ -z "$KEY" ]; then
      echo "::warning::未配置 secret，跳过真实 API 评测"
      echo "skip=true" >> $GITHUB_OUTPUT
      exit 0
    fi
    echo "skip=false" >> $GITHUB_OUTPUT

- name: 执行质量评测
  if: steps.keycheck.outputs.skip == 'false'   # step 级 if，secrets 可用
```

**为什么前两次没发现**：本地 343 项Python 测试不解析 YAML，全绿也照样漏。
而且错误藏在「整个文件作废」的表现形式下——
看不到任何 job 痕迹，失败只用 1 秒，容易被当成「还在排队」。

`tests/check_workflow.py` 的判据不是「YAML 能否解析」
（这三次错误在 YAML 层面**全都合法**），而是
**表达式引用的上下文在该位置是否可用**：

| 位置 | 可用上下文 |
|---|---|
| job 级 `if` | `github` / `needs` / `inputs` / `vars` / `hashFiles`（**无 secrets**） |
| step 级 | 上述 + `secrets` / `steps` / `env` / `matrix` / `strategy` |
| **注释** | **无** —— 但表达式仍会被解析 |

该检查已加成 CI 的第一个 job（`workflow-lint`），
必须排在最前且无前置依赖——否则 workflow 坏掉时
lint job 自己也跑不了，形成死锁。

**测试设计上的一条原则**：检查器自身的测试必须
把**每一次历史错误**还原回去并断言能被抓到，同时断言合法写法**不会**误报。
一个见什么都报的检查器等于没检查器，所以「防误报」用例与「抓错」用例同等重要。

通用教训：**凡是靠外部工具解释执行的文件（YAML / XML / shell 配置），
Python 测试覆盖不到，必须单独加静态检查；
而静态检查的判据必须是那套工具的真实语义，不是通用语法。**

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
