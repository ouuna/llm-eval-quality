# 待办清单 · Prompt

> 用途：把这段话直接发给 AI（Claude / GPT 等），让它接手继续做。
> 前提：AI 能访问到这个项目目录，有修改代码和运行命令的权限。

---

## 我要做的事

请阅读 `/d/llm_eval_project/docs/待办与已知问题.md`（若不存在则读本文件的"背景"部分），
然后**按下面 3 个任务依次处理**。

任务有依赖关系，**不要跳序**。每完成一个就跑一次全量测试确认没引入回归。

---

## 任务一：解决 Gold Set 循环论证（最高优先级）

### 问题现状

`goldset/gold_set.json` 里30 条样本的**题目、答案、标签，以及评测器的检测规则，全部出自同一作者**。

后果：评测器的规则是为检出这批样本的缺陷而设计的，
再用同一批数据验证它，结论天然偏乐观。
现在报告里「Recall 100%」这个数字水分很大。

### 要做的事

**1. 让标注流程能落盘**

`eval/datasets/gold_set.py` 里这两个函数已实现但零调用点：

```
save_gold_set()   # 行471
load_gold_set()   # 行 463
```

把它们接进 `eval/evaluators/validation.py`，
让它支持**对比两个标注文件**并输出差异清单：

```python
#期望的接口
report = compare_annotations("goldset/gold_set.json",
                "goldset/gold_set_second.json")
# 返回：一致数、不一致数、不一致样本的明细（含题目、两个标签、各自理由）
```

**2. 加一致性指标**

在 `eval/evaluators/validation.py` 里实现 Cohen's Kappa
（该文件目前没有任何 IAA 相关代码）：

```python
from eval.evaluators.validation import cohens_kappa
k = cohens_kappa(labels_a, labels_b)
```

要求：
- 空输入、单类别输入要返回 `None` 而非抛异常（原因写在 `reason` 字段）
- 同时给出「观察一致率」作为对照——基线准确率高的场景下，
  Kappa 才有意义（否则 95% 一致率可能只是两个人都随便标true）

**3. 内置数据源的兜底逻辑要改掉**

`eval/datasets/gold_set.py` 里文件不存在时会静默用内置 `SAMPLES`。
**这必须改成警告**——静默兜底会让人误以为验证过了。

**4. 写测试**

- Kappa 的边界：完全一致 = 1.0，完全无关 ≈ 0
- 空输入、单类别、样本数不等
- `compare_annotations` 能正确找出分歧样本

**5. 如实更新文档**

如果暂时找不到第二标注者，在 README 的「已知限制」里明确写：
「Gold Set 为单标注者，评测器指标存在乐观偏差」。

---

## 任务二：提升检索召回率

### 问题现状

`app/rag.py:207` 的 `retrieve()` 是纯关键词打分 + 手写同义词表。

实测召回率：
- `smoke` 集 **50%**
- `coverage` 集 **66.7%**

典型失败案例：
```
问题：ext_sum_01「请总结资料中关于「测试方法」的内容」
召回：「持续集成：代码提交后自动触发构建和测试」
原因：「测试」二字匹配上了，而真正该召回的四个测试方法都没命中
```

召回 50% 意味着一半问题在**检索阶段就失败了**，
后面所有评测器测的都是「模型基于错误上下文作答」，测不出系统真实质量。

### 要做的事

**1. 先诊断，别急着换实现**

现在的 `retrieval_hit_rate` 把「检索没召回」和「召回了但模型没答对」混在一起，
无法判断问题在哪一层。

写一个诊断脚本，输出每条用例的：
- 该召回的片段（来自 Gold Set 的 context 字段）
- 实际召回了什么
- 失败原因分类：**未召回 / 召回但排序低 / 召回完整但答错**

用同一批 Gold Set 跑，产出报告。

**2. 再改检索**

按诊断结果决定策略。候选（按改动量从小到大）：

| 方案 | 改动 | 预期 |
|---|---|---|
| 扩充同义词表 | 小 | 有限 |
| 加二元组（bigram）匹配 | 小 | 中 |
| BM25 替代当前打分 | 中 | 中 |
| Embedding 语义检索 | 大 | 大 |

**注意权衡**：Embedding 会引入第三方依赖，
而本项目目前是**0 第三方依赖**（含 YAML 解析都是自己实现的）。

如果引入 Embedding，需要：
- 在 README 里说明这个取舍
- 保留降级路径——没配 Embedding API 时自动回退到词频检索，
  而不是报错

**3. 用变异测试守住不回归**

改检索后跑 `python -m eval.mutation`，
确认检出率没下降（当前基线 95.33%）。

---

## 任务三：修文档与数据的几处不一致

### 3.1 README 的召回率数字不准

README 第 388 行写「实测召回率约 82%」，
但实测 coverage 集 **66.7%**、smoke 集 **50%**。
82% 那个数字来自更早的测量，检索逻辑后来改过。

改成区间并说明来源，例如
「实测召回率 50%~67%（取决于数据集）」。

### 3.2 三个指标缺算法说明

`eval_results.json` 里 13 个指标中，这三个没有 `definition`：

- `overall_pass_rate`
- `answer_correctness`
- `error_rate`

需求明确要求「每个指标都要有对应的算法说明和局限说明」。
在 `eval/thresholds.py` 补齐这三个的登记。

顺带检查 `caveat`（局限说明）——目前只有 5/13 个有。

### 3.3 补齐薄弱的数据集分布

现有 74 条用例里：
- `multi_hop`（多跳）**仅 1 条**
- `paraphrase` 仅 2 条

而实测表现最差的两类正是：
- 多跳 **50%**
- 对比类 **40%**

单条用例只能说明「这类问题里的一种」，
覆盖面不足时某个细分类目的真实表现完全测不出来。

**优先补 multi_hop 和 context_conflict**，各加 5~8 条。

要求（同 `eval/datasets/coverage.py` 的规范）：
- 每条有明确的 `note` 写清测试意图
- 有 `tags`
- 拒答类不给 `reference_answer`
- 跑 `python -m eval validate --dataset <name>` 必须零问题

---

## 全局约束（所有任务都要遵守）

**1. 每完成一个任务就跑全量测试**
```bash
python -m pytest tests/ -q
```
当前基线：**608 项通过**。不允许下降。

**2. 提交前跑两个自检**
```bash
python -m tests.check_undefined_names   # 0.9 秒
python -m tests.simulate_ci            # 约 30 秒
```

这两个工具存在的理由：今天同一个问题在本地全绿、在 CI 连挂三轮。
前一个防「本地 3.14 宽松、CI 3.11 严格」的差异，
后一个防「本地有 .env、CI 没有」的差异。

**3. 文档里的数字必须绑定测试**

已有 `tests/test_readme_honesty.py` 与 `tests/test_todo_doc.py`，
新增的数字也要照这个模式绑——否则文档会漂移，
而今天已经因为 README 数字与实测不符出过问题。

**4. 不要伪造数据**

需求里明确写了「不要伪造测试数据」。
不要为了让指标好看而调整 Ground Truth 或删用例。

如果指标难看就如实记录，并说明为什么难看。

**5. 修改用 Edit 工具，不要用 shell 字符串替换**

今天在同一文件上用批量替换改了十几次，每次修一处另几处又坏
（中文、缩进、转义在字符串替换里极不可靠）。

正确做法：`git checkout` 回到干净状态 → `Read` 看全貌 →
`Edit` 精确改一处→ 立刻验证。

---

## 背景（AI 若读不到原清单，看这里）

### 项目是什么

LLM / RAG 应用的自动化质量评测框架。
验证对象是**输出质量**，不只是「接口能不能通」。

### 现状数据

| 项 | 数值 |
|---|---|

| 规模 | 89 个 Python 文件 / 24333 行 |
| 测试 | 729 项收集，默认跑 608 项通过（约 30 秒） |
| 评测用例 | 74 条（smoke 8 / full 19 / extended 17 / coverage 30） |
| Gold Set | 30 条，单一标注者 |
| 变异检出率 | 95.33% |
| 运行依赖 | 0 个第三方库 |
| CI workflow | 3 个（test / evaluation / regression） |

### 各数据集实测

| 数据集 | 通过率 | 召回率 | 正确性 | 完整性 | 相关性 |
|---|---|---|---|---|---|
| smoke | 62.5% | 50% | 75% | 75% | 100% |
| coverage | 60.0% | 66.7% | 69.4% | 73.6% | 92.2% |

### 变异测试的真实盲区

| 变异类型 | 检出率 |
|---|---|
| 答案替换 / 追加伪事实 / 数字篡改 | 100% |
| 否定翻转 | 92% |
| 实体调换 | **75%** |
| 因果倒置 | **0%** |
| 时序错乱 | **0%** |

因果倒置与时序错乱是字符级覆盖率算法的先天盲区，
已在测试中如实标注为已知局限。

### 架构要点

```
数据集 → Provider(RAG/HTTP/Mock) → 评测器(幻觉/正确性/拒答)
       → 统计层(分位数/错误率) → 基线对比 → 门禁 → 报告
```

分层解耦：换被测系统只需新增 Provider；
换评测算法不影响门禁与报告。

### 关键命令

```bash
# 跑测试
python -m pytest tests/ -q        # 默认：离线，约 30 秒

# 跑评测（需 API Key）
python -m eval run --dataset smoke
python -m eval list                # 列数据集
python -m eval validate --dataset coverage

# 评测器自验证
python -m eval.evaluators.validation    # Gold Set 反向测量
python -m eval.mutation                 # 变异测试
python -m eval.calibrate                # 阈值校准

# 提交前自检
python -m tests.check_undefined_names
python -m tests.simulate_ci
```

### 三条设计原则（不要违背）

1. **`score=None`（没测到）与 `score=0`（测了是零）必须区分**
   混用会让「全部调用失败」显示成「质量完美」

2. **门禁失败不让 CI 变红**
   被测 RAG 确实有幻觉，评测器拦下它是正确行为。
   让 CI 失败会导致长期红，久而久之没人看，等于没有门禁。
   真正该让 CI 失败的是配置错误、数据错误、运行错误。

3. **低检出率不等于评测器没用**
   它只说明在这一类缺陷上可靠。目标是如实暴露盲区，
   不是刷高数字。所以有测试故意断言「检出率不得为 100%」。

---

## 验收标准

三个任务全部完成后，应满足：

```bash
python -m pytest tests/ -q
# 608 项以上通过，且新增任务对应的测试

python -m eval.mutation
# 检出率不低于 95.33%（改检索不能导致变异检出率下降）

python -m eval run --dataset coverage
# retrieval_hit_rate 显著高于 66.7%

python -m eval.evaluators.validation
# 输出里包含 IAA 指标（或明确说明为何无法计算）

python -m tests.check_undefined_names
python -m tests.simulate_ci
# 两个自检都通过

python -m eval validate --dataset <新增的数据集名>
# 零问题
```

---

## 交付要求

每完成一个任务：
1. 跑全量测试确认无回归
2. 说明**改了什么、依据是什么、还有什么没做**
3. 把新数字更新进 `docs/待办与已知问题.md`

**不要只报告「已完成」** —— 如果某个数字变了
（检出率、召回率、测试数），要明确说变化了多少、为什么变。

如果遇到需要人工判断的事（比如真实的召回失败原因分类），
停下来问，不要自己拍板然后继续跑。