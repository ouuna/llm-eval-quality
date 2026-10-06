# 快速验证清单

> clone 后照着这份清单走，约 5 分钟确认项目可用。

---

## 前置：确认环境

打开 **cmd**（不是 PowerShell），执行：

```cmd
python --version
```

需要 Python 3.9 或更高。

如果提示 `'python' 不是内部或外部命令`，说明Python 没装或没加 PATH。

---

## 第 1 步：clone 项目

```cmd
cd /d D:\
git clone https://github.com/ouuna/llm-eval-quality.git
cd llm-eval-quality
```

> 仓库是公开的，不需要登录也能 clone。

---

## 第 2 步：安装依赖

```cmd
pip install pytest
```

**为什么只有 pytest？** 因为项目核心代码**零第三方依赖**，只用 Python 标准库。

只有语义评测和 Judge 需要额外依赖，不装也不影响主流程。

---

## 第 3 步：离线自测（不需要 API key）

```cmd
python -m eval list
```

**应该看到**：

```
smoke     8 条    CI 快速回归集
full      19 条   完整评测集，12 个类别
regression_bugs   16 条历史缺陷回归集
```

---

## 第 4 步：跑离线评测（4 秒）

```cmd
python -m eval run --dataset smoke --mock
```

**这是最快的验证方式** —— 不需要 API key，不花钱。

**你会看到**（输出可能略有差异）：
```
Cases: 8
Passed: 2
Failed: 6
Quality Gate: FAILED
  - [refusal_accuracy] 未达标：0.6250 >= 0.95
  - [answer_correctness] 未达标：0.3750 >= 0.9
```

**为什么 FAILED？** Mock 固定返回"正确回答"场景，但 smoke 里有 3 条域外用例（期望拒答），Mock 答了 → 门禁检出 `refusal_accuracy` 不达标。

**这是正确行为。** 如果这里显示 PASSED，反而说明门禁坏了。

---

## 第 5 步：看报告

```cmd
start reports\report.html
```

浏览器会打开报告，里面有：
- 12 个指标卡
- 门禁结果与失败原因
- 缺陷 Pareto 分布
- 失败用例详情（含模型回答与判定理由）

---

## 第 6 步：跑离线测试

```cmd
python -m pytest tests/ --ignore=tests/evaluators/test_semantic.py --ignore=tests/evaluators/test_judge.py --ignore=tests/test_quality_gate.py -q
```

**应该看到 180+ 项通过**，其中：
```
184 passed, 1 skipped, 9 xfailed in 5.26s
```

> 完整测试（含需API 的语义与 Judge 评测）是 278 项。
> 离线子集不含这三个文件，所以是 184 项。

> `xfailed` 是正常的 —— 那是明确标记「这 9 条不由声明级验证覆盖」，不做虚假覆盖。

---

## 第 7 步（可选）：看评测器可靠性验证

```cmd
python -m eval.evaluators.validation
```

**输出**：
- 混淆矩阵（TP/FN/FP/TN）
- Precision / Recall / F1 / FPR / FNR
- 漏报与误报的模式归类

**注意**：报告会说明这是循环论证（样本与标签由同一作者构造），这是已知局限。

---

## 第 8 步（可选）：看回归集

```cmd
python -m eval.datasets.regression
```

**会列出 16 条历史缺陷**，含发现阶段、根因、修复方式。

---

## 第 9 步（需要 API key）：真实评测

### 配置环境变量

```cmd
setx OPENAI_API_KEY "你的key"
setx OPENAI_BASE_URL "https://open.bigmodel.cn/api/paas/v4"
setx OPENAI_MODEL_NAME "glm-4-flash"
```

> **必须重开命令行窗口**，`setx` 只对新窗口生效。

### 验证配置

```cmd
python -c "import os; print('Key已配置:', os.getenv('OPENAI_API_KEY') is not None)"
```

### 执行

```cmd
python -m eval run --dataset smoke
```

**第一次会跑 1-2 分钟**（22 次API 调用）。

---

## 常见问题

### 报错 `No module named pytest`

**原因**：pytest 装在另一个 Python 环境。

**解决**：显式指定解释器路径
```cmd
D:\python\python.exe -m pytest
```

检查有多个 Python：
```cmd
where python
```

---

### 报错 `CONFIGURATION ERROR`

**原因**：环境变量未生效。

**解决**：
1. 确认执行过 `setx`
2. **重开命令行窗口**
3. 用 `--mock` 模式先验证项目本身可跑

---

### 想确认「项目是否真的能跑」

**只跑第 4 步**：

```cmd
python -m eval run --dataset smoke --mock
```

**能输出报告 = 项目正常**，不需要任何配置。

---

## 验证清单

完成以下 4 步即表示 clone 后的项目完全可用：

- [ ] `python -m eval list` 输出数据集列表
- [ ] `python -m eval run --dataset smoke --mock` 生成报告
- [ ] `start reports\report.html` 能打开报告
- [ ] `python -m pytest tests/ --ignore=...` 200+ 项通过

**这 4 步全部成功 =无需任何 API 配置，项目即可正常使用。**
