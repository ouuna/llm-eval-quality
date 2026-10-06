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

**应该看到 250 项左右通过**，其中：
```
253 passed, 1 skipped, 9 xfailed in 7.02s
```

> 完整测试（含需 API 的语义与 Judge 评测）是 347 项。
> 离线子集不含 `test_semantic.py` / `test_judge.py` / `test_quality_gate.py`
> 这三个文件，所以是 253 项。

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

### 方式A：项目内配置文件（推荐）

在项目根目录执行：

```cmd
copy env.example .env
```

然后用记事本打开 `.env`，把 `EVAL_API_KEY=` 后面填上你的 key，保存。

**为什么推荐这种方式**：`.env` 只对本项目生效，不会污染系统环境变量。
如果你电脑上还装了别的 AI 工具（某些 Provider 切换类工具也在用
`OPENAI_API_KEY` 这组名字），用系统环境变量就会互相覆盖 ——
你在那边切一次供应商，这边跑评测就会连错服务。

`.env` 已在 `.gitignore` 里，不会被提交到仓库，密钥不会泄露。

### 方式 B：系统环境变量

```cmd
setx EVAL_API_KEY "你的key"
setx EVAL_BASE_URL "https://open.bigmodel.cn/api/paas/v4"
setx EVAL_MODEL_NAME "glm-4-flash"
```

> `setx` 只对新开的命令行窗口生效，改完要重开窗口。

> 如果你之前已经用 `setx OPENAI_API_KEY` 配过了，程序依然能读到，
> 但建议改成 `EVAL_*`，理由见上面方式 A 的说明。

### 验证配置

```cmd
python -m eval config
```

会输出当前生效的配置（密钥脱敏显示）：

```
==================================================================
当前 API 配置
==================================================================
  配置文件      D:\llm_eval_project\.env  （已存在）
  生效的变量名  EVAL_API_KEY
  API Key       4573********r4
  接口地址      https://open.bigmodel.cn/api/paas/v4
  模型名        glm-4-flash
  Judge 模型    glm-4-flash
==================================================================
```

重点看「生效的变量名」那一行：
- 显示 `EVAL_API_KEY` → 走的是项目配置文件，与外部工具互不干扰
- 显示 `OPENAI_API_KEY` → 仍在读系统环境变量，建议迁移到 `.env`

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
