# LLM 质量评测与质量门禁系统

> 面向 RAG 问答系统的自动化质量评测体系
> 幻觉检测 · 正确性验证 · 质量门禁 · 持续集成

---

## 这是什么

给 RAG 问答系统做**自动化质量检测**的框架。

**核心问题**：AI 的回答是一段自然语言，没法像测按钮那样断言"点一下应该弹这个框"。而它最严重的失败不是"答错"，是"**一本正经地编造**"。

**本系统的作用**：量化"这个 AI 系统答得有多靠谱"→ 不达标时阻断 CI。

---

## 快速开始

```bash
python -m eval list                          # 列出数据集
python -m eval validate --dataset full       # 验证数据集
python -m eval run --dataset smoke --mock    # 离线评测（无需 API，4 秒）
python -m eval run --dataset full            # 完整评测
python -m eval case --id smoke_in_01 --detect "错误回答"  # 看判定过程
```

---

## 项目规模

| 模块 | 规模 |
|---|---|
| 核心代码 `eval/` | 30 文件 7477 行 |
| 测试 `tests/` | 18 文件 3778 行 |
| 技术文档 `docs/` | 1286 行 |
| 数据集 | 73 条（8 + 19 + 16 + 30）|
| 指标 | 12 项 |
| 历史缺陷档案 | 16 条 |

**零第三方依赖**：核心逻辑全部用 Python 标准库（json / urllib / re / csv / statistics）。PyYAML 存在时自动优先使用。

---

## 核心能力

### 1. 声明级幻觉检测

**旧方法（字符覆盖率）的硬盲区**：

| 回答 | 上下文 | 旧判定 | 正确 |
|---|---|---|---|
| 中国的首都是**上海** | 北京是中国的首都 | 无幻觉 ❌ | **有幻觉** |
| 分为基础层、用例层、**报告层** | …数据层 | 无幻觉 ❌ | **有幻觉** |
| 由 **Kent Beck** 提出 | …修改代码后 | 无幻觉 ❌ | **有幻觉** |
| 今天天气晴朗 **25 度** | 等价类划分 | 无幻觉 ❌ | **有幻觉** |

**根因**：字符覆盖率不区分实词的"角色"，"上海"和"北京"在字符层面无法分辨。

**本系统**：拆成原子声明 → 逐条验证 → 叠加四类冲突检测（forbidden 事实 / 数字 / 否定极性 / 拉丁词）→ 跑题检测。

**实测 30 条 Gold Set，零漏报。**

---

### 2. 三类用例 + 12 个类别

| 类型 | 场景 | 期望行为 |
|---|---|---|
| `in_domain` | 知识库内有答案 | 准确作答 |
| `out_domain` | 知识库中**无答案** | **必须拒答** |
| `ambiguous` | 问题不完整 | 要求澄清 |

**域外用例是幻觉检测的核心** —— 幻觉只在模型没资料时才暴露。

扩展至 12 类：多跳、否定、数字、同义改写、上下文冲突、Prompt Injection 等。

---

### 3. 三种评测方法并存

| 方法 | 强项 | 弱项 | 定位 |
|---|---|---|---|
| **Lexical** | 确定性、零成本 | 同义改写与实体调换是盲区 | Baseline |
| **Semantic** | 识别同义改写 | 粒度粗，无法定位声明 | 辅助 |
| **LLM Judge** | 复杂语义理解 | 有偏差、成本高 | 交叉验证 |

**三者不合并成一个分数**，因为失效模式不同。

实测分歧（同义改写）：Lexical 0.474 vs Semantic 0.812 —— **存在结论相反的样本**。

---

### 4. 三态而非两态

```python
Status.PASSED / Status.FAILED / Status.ERROR
```

**解决的真实缺陷**：旧实现中 API 调用失败 → 空输出 → 幻觉检测首行 `if not output.strip(): return True` 判为"无幻觉" → **API 失败被当成通过**。

`error` 不计入通过率分母 —— error 是"评测未完成"，不是"质量表现"。

---

### 5. 分层质量门禁

| 层 | 含义 | 特点 |
|---|---|---|
| `critical` | 能力底线 | 严格（域外拒答 = 100%）|
| `quality` | 质量目标 | 允许小幅不达标 |
| `performance` | 性能与成本 | 按项目调整 |
| `stability` | 稳定性 | 按场景调整 |

**门禁失败必须可诊断**：

```
[answer_correctness] 未达标：0.8500 >= 0.9
失败样本：case-01, case-02, case-03

修复方向：
  answer_correctness: 完善 required_facts 标注；检查被测系统 Prompt
```

---

### 6. 缺陷闭环

```
Bug 发现 → 复现 → 固化用例 → 修复 → 回归测试 → CI 保护
```

**16 条历史缺陷全部有档案**：`discovered_in` / `root_cause` / `fix_summary` / `verify_by`。

| 层 | 数量 |
|---|---|
| `evaluator_bug` | 7 |
| `method_limit` | 5 |
| `env_issue` | 4 |

**9 条标为 `xfail` 而非通过** —— 明确声明"这 9 条不由声明级验证覆盖"，不假装通过。

---

### 7. 离线可运行

```bash
python -m eval run --dataset smoke --mock    # 4 秒，无需 API
```

Mock Provider 提供 12 种故障注入：正确答案、错误答案、幻觉、实体调换、拒答、空回答、超时、API 错误、非法 JSON、上下文冲突等。

**CI 无密钥时明确跳过真实评测，而非假装成功。**

---

## 项目结构

```
eval/
├── schemas/         Dataset / EvalCase / GroundTruth / CaseResult / EvalReport
├── datasets/        smoke(8) / full(19) / regression(16) / gold_set(30)
├── providers/       SUT 抽象 + RAGProvider + MockProvider(12 场景)
├── evaluators/      faithfulness / correctness / semantic / judge / validation
├── quality_gate/    分层门禁 + 诊断
├── reporting/       HTML / JSON / CSV
├── cli/             命令行入口
├── runner.py        评测编排
├── baseline.py      基线对比
└── mutation.py      变异测试

configs/quality_gate.yaml
tests/{unit,evaluators,regression}/
```

详见 [`docs/architecture.md`](docs/architecture.md)。

---

## 运行命令

| 命令 | 说明 |
|---|---|
| `python -m eval list` | 列出可用数据集 |
| `python -m eval validate --dataset full` | 验证数据集 |
| `python -m eval run --dataset smoke` | 执行评测 |
| `python -m eval run --dataset smoke --mock` | 离线评测 |
| `python -m eval run --dataset full --judge --semantic` | 增强评测 |
| `python -m eval run --dataset smoke --save-baseline` | 保存基线 |
| `python -m eval run --dataset smoke --compare-baseline` | 检测退化 |
| `python -m eval report --input reports/eval_results.json` | 重新生成报告 |
| `python -m eval gate --input reports/eval_results.json` | 只跑门禁 |
| `python -m eval case --id smoke_in_01` | 用例详情 |
| `python -m eval.evaluators.validation` | 评测器可靠性验证 |
| `python -m eval.datasets.regression` | 回归集统计 |
| `python -m eval.mutation` | 变异测试 |

**退出码**：`0` 通过 / `1` 门禁失败 / `2` 配置错误 / `3` 数据错误 / `4` 运行错误

---

## CI 策略

| 触发 | 数据集 | 密钥需求 |
|---|---|---|
| PR | smoke | 无（离线测试始终执行）|
| push main | smoke + full | 可选 |
| 定时（每日） | full | 可选 |
| 手动 | 可选 | 可选 |

四个 Job：离线单元测试 / Mock 评测链路 / 评测器自检 / 真实 API 评测。

---

## 已知局限

**这些是真实的未解决问题，不隐藏。**

| 局限 | 影响 |
|---|---|
| **Gold Set 存在循环论证** | 样本与标签均由项目作者构造，指标可信度受限 |
| **Judge 会漏判实体调换** | 实测 score=0.72 但 hallucination=False |
| **Judge 与被测系统同模型** | 自我偏好未验证 |
| 声明级验证对对比型回答误报 | 跨上下文整合的句子易被误判 |
| 冲突检测依赖 SVO 语序 | 主谓倒装场景漏检 |
| 检索仍是关键词匹配 | 命中率 68% 是瓶颈 |
| 未开 branch protection | CI 失败不能真正阻止合并 |

**真正的评测器可靠性验证需要**：他人撰写的答案 + 至少两人独立标注 + 200+ 样本。

详见 [`docs/evaluation.md`](docs/evaluation.md) 第八节。

---

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | 架构、分层、关键设计决策、扩展点 |
| [`docs/metrics.md`](docs/metrics.md) | 12 项指标的定义/算法/阈值/优缺点/已知误差 |
| [`docs/dataset.md`](docs/dataset.md) | 12 类别规范、Ground Truth 五要素、增补原则 |
| [`docs/evaluation.md`](docs/evaluation.md) | 评测方法、为什么这么设计、怎么证明可靠 |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | 常见问题与排查 |

---

## 环境变量

```cmd
setx OPENAI_API_KEY "你的key"
setx OPENAI_BASE_URL "https://open.bigmodel.cn/api/paas/v4"
setx OPENAI_MODEL_NAME "glm-4-flash"

REM 可选：Judge 与 Embedding 独立配置（支持交叉验证）
setx JUDGE_MODEL_NAME "glm-4-plus"
```

> `setx` 只对新开的命令行窗口生效。

---

## 测试

```bash
# 全部（需 API）
python -m pytest tests/ -v

# 仅离线测试（无需 API）
python -m pytest tests/ \
  --ignore=tests/evaluators/test_semantic.py \
  --ignore=tests/evaluators/test_judge.py \
  --ignore=tests/test_quality_gate.py -v
```

---

## 设计原则

1. **不伪造数据** —— 指标不可用时记 `unavailable`，不填 0
2. **不假装通过** —— 未覆盖的缺陷标 `xfail`，不标绿
3. **失败必须可诊断** —— 门禁失败要说清哪、差多少、怎么修
4. **解耦** —— 换 SUT 不改评测框架；换业务不改代码
5. **诚实记录局限** —— 已知问题写进文档，不藏
