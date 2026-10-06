# LLM 质量评测与质量门禁系统

> 面向 RAG 问答系统的自动化质量评测体系
> 幻觉检测 · 正确性验证 · 质量门禁 · 持续集成

一个用于量化「RAG 问答系统答得有多靠谱」的自动化评测框架。核心目标是解决一个具体问题：**大模型最严重的失败不是"答错"，而是"一本正经地编造"（幻觉）**，而传统测试方法无法有效检测这种现象。

---

## 核心能力

### 声明级幻觉检测

多数基于字符覆盖率的实现存在一个根本缺陷：它无法区分实词的「角色」。

| 回答 | 上下文 | 字符覆盖率判定 | 正确判定 |
|---|---|---|---|
| 中国的首都是**上海** | 北京是中国的首都 | 无幻觉 ❌ | **有幻觉** |
| 分为基础层、用例层、**报告层** | ……数据层 | 无幻觉 ❌ | **有幻觉** |
| 由 **Kent Beck** 提出 | ……修改代码后 | 无幻觉 ❌ | **有幻觉** |
| 今天天气晴朗 **25 度** | 等价类划分 | 无幻觉 ❌ | **有幻觉** |

**根因**：「上海」与「北京」在字符层面无法分辨何者正确。

**本项目的做法**：将回答拆解为原子声明 → 逐条验证是否有上下文依据 → 叠加四类冲突检测（禁止事实 / 数字 / 否定极性 / 拉丁词）→ 跑题检测。

**实测**：30 条人工标注样本，零漏报。

---

### 三层评测方法

| 方法 | 强项 | 弱项 | 定位 |
|---|---|---|---|
| **Lexical** | 确定性、零成本 | 同义改写与实体调换是盲区 | 基线 |
| **Semantic** | 能识别同义改写 | 粒度粗，无法定位具体声明 | 辅助 |
| **LLM Judge** | 复杂语义理解 | 存在偏差、成本高 | 交叉验证 |

**三者不合并为单一分数**，因为失效模式不同。

实测分歧（同义改写场景）：Lexical 0.474 vs Semantic 0.812 —— **存在结论相反的样本**，证明单一方法不可信。

---

### 三态而非两态

```python
Status.PASSED / Status.FAILED / Status.ERROR
```

**解决的真实缺陷**：早期实现中，API 调用失败 → 输出为空串 → 幻觉检测首行 `if not output.strip(): return True` 判为「无幻觉」→ **API 失败被当作通过**。

`error` 不计入通过率分母 —— error 意味着「评测未完成」，而非「质量表现」。

---

### 分层质量门禁

| 层 | 含义 | 特点 |
|---|---|---|
| `critical` | 能力底线 | 严格（域外拒答率 = 100%）|
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

### 缺陷闭环

```
Bug 发现 → 复现 → 固化用例 → 修复 → 回归测试 → CI 保护
```

**16 条历史缺陷全部附档案**：`discovered_in` / `root_cause` / `fix_summary` / `verify_by`。

| 缺陷层 | 数量 |
|---|---|
| `evaluator_bug`（评测器自身缺陷）| 7 |
| `method_limit`（方法论局限）| 5 |
| `env_issue`（运行环境问题）| 4 |

其中 **9 条标记为 `xfail` 而非通过** —— 明确声明「这 9 条不由声明级验证覆盖」，不做虚假覆盖。

---

## 修复的 P0 级缺陷

| 缺陷 | 危害 |
|---|---|
| 空输出被判为无幻觉 | **API 失败被当作通过** |
| `expected_output` 不参与判定 | 「域内准确率 100%」是失真指标 |
| 相关性计算方向反了 | 无关回答可得满分 |
| 实体调换检测不出 | 最隐蔽的幻觉类型全部漏检 |
| 拉丁字母被丢弃 | 编造的外国人名检测不出 |
| 数字冲突检测门槛过高 | 高字面重合场景漏报 |
| 跑题未判为幻觉 | 答非所问不判定为失败 |

**验证依据**：修复前 Recall 78.6% → 修复后 100%。

---

## 项目规模

| 模块 | 规模 |
|---|---|
| 核心代码 | 30 文件 7477 行 |
| 测试 | 18 文件 3778 行，278 项测试 |
| 技术文档 | 5 份，1286 行 |
| 评测数据集 | 73 条（4 个层级）|
| 质量指标 | 12 项 |
| 历史缺陷档案 | 16 条 |

**零第三方依赖**：核心逻辑全部基于 Python 标准库（`json` / `urllib` / `re` / `csv` / `statistics`）。若环境中存在 PyYAML 则自动优先使用。

---

## 快速开始

### 环境准备

```cmd
setx OPENAI_API_KEY "你的key"
setx OPENAI_BASE_URL "https://open.bigmodel.cn/api/paas/v4"
setx OPENAI_MODEL_NAME "glm-4-flash"
```

> `setx` 仅对新开的命令行窗口生效。

### 运行

```bash
# 列出可用数据集
python -m eval list

# 验证数据集合法性
python -m eval validate --dataset full

# 离线评测（无需 API，约 4 秒）
python -m eval run --dataset smoke --mock

# 完整评测
python -m eval run --dataset full

# 启用语义与 Judge
python -m eval run --dataset full --semantic --judge

# 查看单条用例 + 演示评测器判定过程
python -m eval case --id smoke_in_01 --detect "测试用的错误回答"
```

### 测试

```bash
# 全部测试（需 API）
python -m pytest tests/ -v

# 仅离线测试（无需 API）
python -m pytest tests/ \
  --ignore=tests/evaluators/test_semantic.py \
  --ignore=tests/evaluators/test_judge.py \
  --ignore=tests/test_quality_gate.py -v
```

---

## 命令一览

| 命令 | 说明 |
|---|---|
| `python -m eval list` | 列出可用数据集 |
| `python -m eval validate --dataset full` | 验证数据集合法性 |
| `python -m eval run --dataset smoke` | 执行评测 |
| `python -m eval run --dataset smoke --mock` | 离线评测（无需 API）|
| `python -m eval run --dataset full --judge --semantic` | 增强评测 |
| `python -m eval run --dataset smoke --save-baseline` | 保存基线 |
| `python -m eval run --dataset smoke --compare-baseline` | 检测指标退化 |
| `python -m eval report --input reports/eval_results.json` | 重新生成报告 |
| `python -m eval gate --input reports/eval_results.json` | 仅执行门禁评估 |
| `python -m eval case --id smoke_in_01` | 查看用例详情 |
| `python -m eval.evaluators.validation` | 评测器可靠性验证 |
| `python -m eval.datasets.regression` | 回归集统计 |
| `python -m eval.mutation` | 变异测试 |

**退出码**：`0` 通过 / `1` 门禁失败 / `2` 配置错误 / `3` 数据错误 / `4` 运行错误

---

## 项目结构

```
eval/
├── schemas/         Dataset / EvalCase / GroundTruth / CaseResult / EvalReport
├── datasets/        smoke(8) / full(19) / regression(16) / gold_set(30)
├── providers/       SUT 抽象 + RAGProvider + MockProvider(12 种故障注入)
├── evaluators/      faithfulness / correctness / semantic / judge / validation
├── quality_gate/    分层门禁 + 失败诊断
├── reporting/       HTML / JSON / CSV
├── cli/             命令行入口
├── runner.py        评测编排
├── baseline.py      基线对比
└── mutation.py      变异测试

configs/quality_gate.yaml
docs/                architecture / metrics / dataset / evaluation / troubleshooting
tests/{unit,evaluators,regression}/
```

---

## CI 策略

| 触发条件 | 数据集 | 密钥需求 |
|---|---|---|
| Pull Request | smoke | 不需要（离线测试始终执行）|
| push 到 main | smoke + full | 可选 |
| 定时（每日）| full | 可选 |
| 手动触发 | 可选 | 可选 |

四个独立 Job：离线单元测试 / Mock 评测链路 / 评测器自检 / 真实 API 评测。

**无 API 密钥时明确跳过真实评测，而非伪装成功。**

---

## 已知局限

以下是真实存在的未解决问题，不做隐藏。

| 局限 | 影响 |
|---|---|
| **Gold Set 存在循环论证** | 样本与标签均由项目作者构造，指标可信度受限 |
| **Judge 会漏判实体调换** | 实测 score=0.72 但 hallucination=False |
| **Judge 与被测系统使用同一模型** | 自我偏好风险未验证 |
| 声明级验证对对比型回答误报 | 跨上下文整合的句子易被误判 |
| 冲突检测依赖 SVO 语序 | 主谓倒装场景漏检 |
| 检索仍为关键词匹配 | 命中率 68% 是当前瓶颈 |
| 未启用 branch protection | CI 失败无法真正阻止合并 |

**真正的评测器可靠性验证需要**：他人撰写的回答样本 + 至少两人独立标注 + 200 条以上样本量。详见 [`docs/evaluation.md`](docs/evaluation.md) 第八节。

---

## 设计原则

1. **不伪造数据** —— 指标不可用时记为 `unavailable`，不填 0
2. **不假装通过** —— 未覆盖的缺陷标记 `xfail`，不做虚假覆盖
3. **失败必须可诊断** —— 门禁失败须说明哪项指标、差多少、如何修复
4. **解耦** —— 更换被测系统不改评测框架；更换业务不改代码
5. **诚实记录局限** —— 已知问题写入文档，不隐藏

---

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | 架构分层、关键设计决策、扩展点 |
| [`docs/metrics.md`](docs/metrics.md) | 12 项指标的定义/算法/阈值/优缺点/已知误差 |
| [`docs/dataset.md`](docs/dataset.md) | 12 个类别规范、Ground Truth 五要素、用例增补原则 |
| [`docs/evaluation.md`](docs/evaluation.md) | 评测方法、设计动机、可靠性验证 |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | 常见问题与排查路径 |

---

## Quick Verification

Just cloned and want to confirm it works? See [`QUICKSTART.md`](QUICKSTART.md) (Chinese) — 5 minutes.

**Fastest check** (no API key required):
```bash
python -m eval run --dataset smoke --mock
```

[中文 README](./README.md)

---

## Quick Verification

Just cloned and want to confirm it works? See [`QUICKSTART.md`](QUICKSTART.md) (Chinese) — 5 minutes.

**Fastest check** (no API key required):
```bash
python -m eval run --dataset smoke --mock
```

[中文 README](./README.md)

---

## License

MIT
