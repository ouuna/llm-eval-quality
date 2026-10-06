# 数据集说明

## 三个层级

| 数据集 | 条数 | 用途 | 触发时机 |
|---|---|---|---|
| `smoke` | 8 | 快速回归，CI 默认 | PR / 每次提交 |
| `full` | 19 | 完整评测，12 类覆盖 | 手动 / 发版前 |
| `regression_bugs` | 16 | **验证评测器自身不复发** | 每次提交 |
| `gold_set` | 30 | **验证评测器可靠性** | 手动 |

**注意**：`regression_bugs` 与 `gold_set` 不属于用例数据集体系——它们验证的是评测器本身，不是被测系统。

---

## 12 个类别

| 类别 | 测试目的 | 典型陷阱 |
|---|---|---|
| `in_domain` | 检索召回与答案正确 | 答漏要点 |
| `out_domain` | **幻觉抑制（核心）** | 热情作答而非拒答 |
| `ambiguous` | 问题不完整时的处理 | 强行作答 |
| `boundary` | 部分覆盖场景 | 把部分当全部 |
| `multi_hop` | 多处信息综合 | 漏掉某处 |
| `negative` | 否定式提问 | 把"不是"答成"是" |
| `numeric` | 数字与时序事实 | 数量编造 |
| `paraphrase` | 同义表述下的检索 | 检索不到 |
| `insufficient_ctx` | 上下文不足时的处理 | 硬答 |
| `context_conflict` | 矛盾上下文的识别 | 盲选一条 |
| `unanswerable` | 本质不可回答 | 硬答 |
| `prompt_injection` | 越权诱导与信息泄露 | 泄露系统提示词 |

---

## Ground Truth 五要素

```python
GroundTruth(
    reference_answer="...",       # 给人看，报告展示
    acceptable_answers=[...],     # 语义等价表述，避免改写误判
    required_facts=[...],         # 正确性判据（必须出现）
    forbidden_facts=[...],        # 幻觉判据（禁止出现）
    evidence=[...],               # 冲突与依据标注
)
```

**为什么不用 `reference_answer` 字符串匹配？**

因为改写会失配：

```
参考：把输入域划分为若干互不相交的子集
回答：把输入域切分为互不相交的子集     ← 一字之差，字符串匹配失败
```

`acceptable_answers` 提供语义等价表述；`required_facts` 提供事实级判据。

---

## 用例字段规范

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | ✅ | 唯一 |
| `category` | ✅ | 12 类之一 |
| `question` | ✅ | 问题 |
| `expected_behavior` | ✅ | `answer` / `refuse` / `clarify` / `partial` |
| `context` | | 显式指定上下文（冲突类必须） |
| `ground_truth` | ✅ | 见上 |
| `difficulty` | | `easy` / `medium` / `hard` |
| `tags` | | 自由标注 |

### 关键设计：`expected_behavior` 显式声明

```python
expected_behavior: answer | refuse | clarify | partial
```

**不从 category 推断。**

旧实现里 `ambiguous` 与 `out_domain` 的判定逻辑完全相同（都是"无幻觉即通过"），因为期望行为是隐式的。现在 `ambiguous` 显式声明 `clarify`，必须有专门逻辑支持。

---

## 数据集校验

```bash
python -m eval validate --dataset full
```

校验项：
- ID 唯一
- `expected_behavior` 与 category 一致（如 out_domain 必须 refuse）
- `answer` 类必须有参考答案
- `refuse` 类不应有参考答案
- `required_facts` 与 `forbidden_facts` 不得矛盾
- `answer` 类仅有 `reference_answer` 会被提示（判定依据不足）

**不合规在评测开始前即失败**，而不是跑到一半才炸。

---

## 增补用例的原则

**不要为了增加数量生成无测试价值的样本。**

优先构造**具有代表性的 failure case**：

| 优先级 | 类型 | 举例 |
|---|---|---|
| 1 | 已发现的真实缺陷 | 实体调换、数字编造 |
| 2 | 已知薄弱环节 | 同义改写、上下文冲突 |
| 3 | 边界情况 | 极短回答、极长回答 |
| 4 | 常规正常 | 典型正确回答 |

---

## Gold Set 的特殊地位

### 它验证的是评测器，不是被测系统

```bash
python -m eval.evaluators.validation
```

输出混淆矩阵、Precision/Recall/F1/FPR/FNR，以及漏报与误报的模式归类。

### 必须人工复核

```python
reviewed = True   # 人工确认过的标签才计入统计
```

**未复核的样本不参与验证。** 这是设计意图——自动生成的标注不能当自己的真值。

### 当前局限

30 条样本由项目作者构造并标注，**存在循环论证**：

```
我写答案 → 我写 forbidden_facts → 评测器命中 → 我标"有幻觉"
```

**真实的验证需要**：
1. 他人撰写的答案
2. 至少两人独立标注
3. 200+ 样本

这是本项目最重要的未完成项。

---

## 回归数据集的定位

`regression_bugs` 记录**开发过程中发现的每一个缺陷**：

```python
RegressionCase(
    id="REG-P6-001",
    phase="Phase 6 验证",
    layer="evaluator_bug",      # evaluator_bug / method_limit / env_issue
    severity="P0",
    root_cause="...",
    fix_summary="...",
    verify_by="faithfulness",   # 该缺陷由哪个指标验证
)
```

**`verify_by` 字段很关键**——它防止"用错误的指标验证缺陷"：

| verify_by | 含义 |
|---|---|
| `faithfulness` | 声明级验证 |
| `correctness` | 正确性/完整性 |
| `relevance` | 相关性 |
| `mechanism` | 机制专项测试（如 CRLF 切分） |
| `manual` | **需人工核验，不假装通过** |

---

## 文件位置

```
eval/datasets/
├── __init__.py        注册表 + 组合加载（smoke+full）
├── smoke.py           快速回归集
├── full.py            完整评测集
├── regression.py      缺陷档案
└── gold_set.py        人工标注验证集

configs/
└── quality_gate.yaml  门禁阈值

goldset/               Gold Set 标注文件（CSV/JSON）
```

---

## 组合数据集

```bash
python -m eval run --dataset smoke+full      # 合并去重后执行
```

ID 重复的用例会被自动去重。
