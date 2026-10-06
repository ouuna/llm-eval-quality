# 敏感信息处理说明

本项目对外公开前需完成以下检查。**任何一项缺失都可能导致密钥泄露或个人信息暴露。**

---

## 一、密钥处理

### 代码中的密钥读取方式

全部通过环境变量读取，代码中不出现任何密钥字面量：

```python
API_KEY = os.getenv("OPENAI_API_KEY")
```

### 提交前必查

```bash
# 检查是否误提交密钥
grep -rn "OPENAI_API_KEY" --include="*.py" . | grep -v getenv
grep -rnE "[A-Za-z0-9]{32}\.[A-Za-z0-9]{40,}" --include="*.py" .
```

两条命令应无输出。

### 需要提交的文件

| 文件 | 是否提交 | 说明 |
|---|---|---|
| `env.example` | ✅ 提交 | 仅含字段名与示例，不含真实值 |
| `.env` | ❌ 禁止 | 已在 .gitignore 中排除 |
| `.github/workflows/eval.yml` | ✅ 提交 | 密钥走 Secrets，不硬编码 |

### CI 密钥配置

GitHub 仓库 → Settings → Secrets and variables → Actions → New repository secret：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL_NAME`

**不要把密钥写在 workflow 文件里。** 本项目已使用 `${{ secrets.XXX }}` 方式引用。

---

## 二、路径与身份信息

### 代码中不应出现

- 个人用户名（如 Windows 用户名、Linux 用户名）
- 绝对路径（如 `C:\Users\xxx\...`、`D:\...`）
- 姓名、学号、身份证号、手机号、宿舍等
- 邮箱地址（个人邮箱）

本项目通过 `PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))` 动态推导路径，不含硬编码个人路径。

### 检查命令

```bash
grep -rniE "C:\\\\Users|D:\\\\|在学|student|身份证" --include="*.py" --include="*.yaml" --include="*.md" .
```

---

## 三、报告文件

`reports/` 目录下的 JSON / CSV 报告包含模型回答原文。**理论上可能包含真实业务内容**，公开项目建议：

- 已在 `.gitignore` 中排除 `reports/*.json`、`reports/*.csv`
- 如需展示报告效果，保留脱敏后的 `report.html`（已确认不含密钥）

---

## 四、提交前最终检查清单

```bash
# 1. 确认 .gitignore 生效
git status --short | grep -E "\.env|reports/.*\.(json|csv)" && echo "存在待提交敏感文件" || echo "OK"

# 2. 确认无密钥字面量
git diff --cached | grep -iE "api[_-]?key\s*=\s*[\"'][^\"']{8,}" && echo "发现硬编码密钥" || echo "OK"

# 3. 确认报告未入库
git ls-files | grep -E "reports/.*\.(json|csv)" && echo "报告已入库" || echo "OK"
```

三条均输出 OK 后方可提交。

---

## 五、如果密钥已误提交

1. 立即到服务商后台**吊销并重新生成密钥**
2. 用 `git filter-repo` 或 BFG 清理历史
3. 强制推送 `git push --force`
4. 再次确认密钥已失效

**注意：仅删除当前版本的密钥文件是不够的，git 历史中仍可检出。**
