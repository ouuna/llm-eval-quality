# 常见问题

## 环境与配置

### 报 `CONFIGURATION ERROR：缺少环境变量`

**原因**：`EVAL_API_KEY` 未配置（接口地址与模型名有默认值，不配置也能跑）。

**解决（推荐）**：用项目内配置文件，只对本项目生效：

```cmd
copy env.example .env
```

打开 `.env`，填入 `EVAL_API_KEY`，保存后直接重跑命令即可，不用重开终端。

**验证配置是否生效**：
```cmd
python -m eval config
```

也可以用系统环境变量：
```cmd
setx EVAL_API_KEY "你的key"
```

> ⚠️ `setx` **只对新开的命令行窗口生效**。

---

### 电脑上还装了别的 AI 工具，本项目配置被它改掉了

**现象**：本来跑得好好的，突然报连接错误，或者
`python -m eval config` 显示的接口地址变成了别家的。

**原因**：`OPENAI_API_KEY` / `OPENAI_BASE_URL` 这组变量名是OpenAI 最早定义的，
后来被所有兼容 OpenAI 协议的服务商沿用。本机上其他工具
（某些 AI CLI 的 Provider 切换工具）也会往**系统环境变量**里写同名的值，
两边互相覆写。

**解决**：改用项目专属变量名 + 项目内 `.env`，彻底隔离。

```cmd
copy env.example .env
```

在 `.env` 里配置 `EVAL_API_KEY` / `EVAL_BASE_URL` / `EVAL_MODEL_NAME`。
这套变量名只有本项目在用，不与任何外部工具重叠。

**确认隔离是否生效**：
```cmd
python -m eval config
```

- 「生效的变量名」显示 `EVAL_API_KEY` → 已隔离，外部工具改不动你的配置
- 显示 `OPENAI_API_KEY` → 还在读系统环境变量，需要迁移

**验证方法**：在 CC Switch 或同类工具里切换一次供应商，然后重跑上面的命令，
显示的接口地址应该完全不变。

---

### 变量配了但读不到

逐条排查：

```cmd
python -m eval config
```

| 输出里看到 | 说明 | 怎么办 |
|---|---|---|
| 配置文件后显示「不存在」 | 没有 `.env` 文件 | 执行 `copy env.example .env` |
| 生效的变量名是 `OPENAI_API_KEY` | 读的是系统环境变量，不是 `.env` | 检查 `.env` 里的变量名有没有拼错 |
| 缺失项非空 | 该变量在`.env` 和环境变量里都没有 | 确认这一行的 `=` 后面填了值，不是空着 |

常见低级错误：
- `.env` 保存成了 `env.txt` 或 `env.example.txt` → 不会被识别
- 变量名写成 `EVAL_APIKEY`（少了下划线）→ 不会生效
- `=` 号两边有空格 → 变量名会被带上尾随空格，识别不到

---

### `No module named pytest`

**原因**：pytest 装在另一个 Python 环境。

**解决**：用装了 pytest 的那个解释器：
```cmd
D:\python\python.exe -m pytest
```

检查 `where python` 看有多个 Python 时尤其注意。

---

### 脚本报 `SyntaxError: invalid syntax` 但代码看起来没问题

**原因**：从聊天/网页复制代码时混入了**零宽字符**或**全角标点**。

典型症状：注释里含全角冒号 `：` 时，Python 3.14 报语法错误。

**解决**：
```python
import ast, io
try:
    ast.parse(io.open("file.py", encoding="utf-8").read())
except SyntaxError as e:
    print(f"行 {e.lineno}: {e.msg}")
```

**预防**：用 VS Code / PyCharm 编辑，不用记事本。

---

### 知识库读取后只有 1 条文档，检索恒为空

**原因**：文件是 CRLF 换行（Windows 默认），而切分逻辑按 `\n\n`。

**现象**：整份文件被当成一条超大文档，与任何关键词匹配分数都低 → 全被过滤。

**解决**：切分前统一换行符。
```python
content = content.replace("\r\n", "\n").replace("\r", "\n")
```

**预防**：已加 `.gitattributes` 强制 `eol=lf`。

---

## 评测相关

### 指标全为 0 / 全为 unavailable

**检查**：

1. 环境变量是否生效（重开终端）
2. `OPENAI_MODEL_NAME` 是否拼写正确
3. 单独测试 API：
```bash
python -c "import os; print(os.getenv('OPENAI_API_KEY') is not None)"
```

---

### 幻觉率异常高

**可能原因**：

| 原因 | 排查方式 |
|---|---|
| 知识库为空 | 看检索片段是否为 0 |
| 模型不遵循 Prompt | 检查 Prompt 是否被改过 |
| 阈值过低 | 看 `config.yaml` 的 `hallucination_threshold` |

**注意**：`forbidden_facts` 未标注时，实体调换类幻觉会漏检（不是误报，是漏报）。

---

### 域外拒答率低

**可能原因**：

1. Prompt 中的拒答约束被移除 → 用 A/B 实验验证：
```bash
python tests/ab_experiment.py
```
2. 知识库意外包含了域外内容 → 检查知识库覆盖范围

---

### Judge 分数波动大

**实测**：`temperature=0` 时 spread=0.000，稳定。

若仍波动，检查：
1. `temperature` 是否真的为 0
2. 是否所有 Judge 调用走同一模型
3. 网络是否导致重试（重试可能返回不同结果）

**已知局限**：Judge 与被测系统同模型时存在自我偏好，未验证。需用 `JUDGE_MODEL_NAME` 换模型交叉验证。

---

## CI 相关

### Actions 显示 "SKIPPED"

**原因**：无 `OPENAI_API_KEY` secret。

**这是预期行为**——无密钥时明确跳过，而非假装成功。

**离线测试仍会执行**：`unit-tests` 和 `offline-eval` 不需要密钥。

---

### 门禁失败了但我不知道改哪里

门禁输出包含修复方向：

```
[answer_correctness] 未达标：0.8500 >= 0.9
失败样本：case-01, case-02, case-03

修复方向：
  answer_correctness: 完善 required_facts 标注；检查被测系统 Prompt
```

**跑 `python -m eval case --id case-01` 可看单条详情。**

---

### 报告想重新生成但不想重跑评测

```bash
python -m eval report --input reports/eval_results.json
```

调整报告模板后无需重新调用 API。

---

## 数据集相关

### `validate` 报 "仅有 reference_answer 不足以程序化判定正确性"

**原因**：只有参考答案，没有 `required_facts`。

**解决**：补充 `required_facts`，程序才能做事实级判定：

```python
ground_truth=GroundTruth(
    reference_answer="...",
    required_facts=["北京是中国的首都"],
    forbidden_facts=["北京是中国最大的城市"],
)
```

---

### 域外用例填了 `expected_output` 报校验错误

**这是设计如此**。域外用例期望系统拒答，不应有标准答案。

```python
# 正确
ground_truth=GroundTruth(reference_answer=None)

# 错误：填了参考答案
ground_truth=GroundTruth(reference_answer="参考答案...")
```

---

## 理解指标

### 为什么相关性从 0.47 变成 0.51

0.47 是**旧实现**（方向反了），测的是"答案有没有抄问题里的字"。
0.51 是**新实现**，用 `required_facts` 覆盖率。

**新旧不可直接比较**——定义变了。

---

### 域内准确率 100% 但相关性只有 0.5，矛盾吗

**不矛盾**，但**值得警惕**。

如果两者严重不自洽，说明至少有一个指标定义错误。
**指标自洽性检查是发现指标失准的有效手段。**

---

### 语义相似度 0.79 但判定为有幻觉，矛盾吗

**不矛盾**。语义相似只说明"话题相关"，不能说明"每个事实都有依据"。

```
"多出一层业务层"  语义 0.773   ← 话题高度相关
"完全正确答案"    语义 0.794   ← 但差得不多
```

**语义层面区分度不够，真正的能力在声明级验证。**

---

## 开发

### 如何新增一个 Evaluator

实现返回 dict 的函数即可，无需修改框架：

```python
def my_evaluator(answer, context, ground_truth) -> dict:
    return {"score": 0.0, "is_passed": False, "reason": ""}
```

在 `eval/runner.py` 的 `evaluate_case()` 中接入。

---

### 如何新增一个 SUT

实现 Protocol：

```python
class MyProvider:
    name = "my_sut"
    def ask(self, question) -> SUTResponse: ...
    def is_available(self) -> bool: ...
```

然后 `register_provider("my_sut", MyProvider)`。

---

### 如何在无 API 环境开发

```bash
python -m eval run --dataset smoke --mock
```

Mock Provider 提供 12 种故障场景，含超时、API 错误、非法 JSON。

---

### 哪些测试需要 API

| 测试 | 需要 API |
|---|---|
| `tests/unit/` | ❌ |
| `tests/evaluators/test_faithfulness.py` | ❌ |
| `tests/evaluators/test_correctness.py` | ❌ |
| `tests/evaluators/test_mock_provider.py` | ❌ |
| `tests/evaluators/test_quality_gate.py` | ❌ |
| `tests/regression/` | ❌ |
| `tests/evaluators/test_semantic.py` | ✅ |
| `tests/evaluators/test_judge.py` | ✅ |
| `tests/test_quality_gate.py`（旧） | ✅ |

**无 API 也能跑大部分测试。**
