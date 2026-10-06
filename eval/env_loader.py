"""
统一配置加载器（零第三方依赖）
--------------------------------
为什么需要这个文件
------------------
大模型 SDK 生态普遍使用 `OPENAI_API_KEY` / `OPENAI_BASE_URL` 这类环境变量名，
因为它们最早由 OpenAI 定义，后来被所有兼容 OpenAI 协议的服务商沿用。

问题在于：一些本机工具（如 Claude Code Provider 切换类工具）也把
**自己的**配置写进同名的用户级环境变量。当用户在这个电脑上同时使用
本项目和这类工具时，两边会互相覆写同一个变量，导致项目突然连错服务
或者直接报缺配置。

本项目的解决方式
----------------
1. 提供**项目专属变量名** `EVAL_*`，与外部工具的变量名完全隔离；
2. 支持项目根目录的 `.env` 文件，配置只对本项目生效，不写入系统；
3. 仍然兼容旧的 `OPENAI_*` 变量名，保证 GitHub Actions 等已有配置
   无需修改即可继续工作。

取值优先级（从高到低）
----------------------
    进程环境变量中的 EVAL_*  >  .env 文件中的 EVAL_*
      >  进程环境变量中的 OPENAI_*  >  .env 文件中的 OPENAI_*

即：显式设置的永远赢过文件里的；项目专属名赢过通用名。
"""

import os

# ---------------------------
# 变量名定义
# ---------------------------

# 项目专属名（推荐使用，与任何外部工具不冲突）
ENV_API_KEY = "EVAL_API_KEY"
ENV_BASE_URL = "EVAL_BASE_URL"
ENV_MODEL_NAME = "EVAL_MODEL_NAME"
ENV_JUDGE_MODEL = "EVAL_JUDGE_MODEL_NAME"

# 通用名（保留兼容：CI 的 secrets、既有文档都还在用）
LEGACY_API_KEY = "OPENAI_API_KEY"
LEGACY_BASE_URL = "OPENAI_BASE_URL"
LEGACY_MODEL_NAME = "OPENAI_MODEL_NAME"
LEGACY_JUDGE_MODEL = "JUDGE_MODEL_NAME"

# Judge 模型未单独指定时的兜底顺序（详见 get_judge_model_name）
JUDGE_FALLBACK_PAIRS = (
    (ENV_JUDGE_MODEL, LEGACY_JUDGE_MODEL),
    (ENV_MODEL_NAME, LEGACY_MODEL_NAME),
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")

# 供 CLI / 报错信息使用的配置项清单：(项目名, 兼容名, 中文说明, 是否必填)
# 只有 API Key 是硬性必填：接口地址与模型名都有内置默认值，
# 缺失时框架能用默认值继续工作，不该把这种情况拦成配置错误。
REQUIRED_ITEMS = (
    (ENV_API_KEY, LEGACY_API_KEY, "API Key", True),
    (ENV_BASE_URL, LEGACY_BASE_URL, "接口地址", False),
    (ENV_MODEL_NAME, LEGACY_MODEL_NAME, "模型名", False),
)

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL_NAME = "glm-4-flash"


# ---------------------------
# .env 解析
# ---------------------------
def parse_env_text(text):
    """
    解析 .env 文本为dict。

    支持：
        KEY=VALUE
        KEY="VALUE"# 带引号，末尾注释被忽略
        KEY=VALUE      # 行尾注释（值中不含 # 时）
        export KEY=VALUE
        # 整行注释、空行

    不做变量插值（${...}），保持零依赖且行为可预测。
    重复出现的键，后者覆盖前者。
    """
    result = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export "):].lstrip()

        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        # 去掉引号包裹；引号内的 # 不视为注释
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            hash_pos = value.find(" #")
            if hash_pos != -1:
                value = value[:hash_pos].strip()

        result[key] = value

    return result


def load_env_file(path=None):
    """
    读取 .env 文件。文件不存在时返回空 dict，不报错。

    读取失败（如编码异常）同样降级为空 dict——配置问题不应该让
    整个评测框架无法启动，真正的缺失会在后续的可用性检查里报出来。
    """
    target = path or ENV_FILE
    try:
        with open(target, "r", encoding="utf-8") as f:
            return parse_env_text(f.read())
    except (OSError, UnicodeDecodeError):
        return {}


# ---------------------------
# 取值
# ---------------------------
def _lookup(primary_name, legacy_name):
    """
    按隔离优先级查找单个配置项。

    顺序固定为：
        1. 进程环境变量中的项目名（EVAL_*）
        2. .env 文件中的项目名（EVAL_*）
        3. 进程环境变量中的通用名（OPENAI_*）
        4. .env 文件中的通用名（OPENAI_*）

    关键点在于第2 条排在第 3 条之前。
    如果让通用名的环境变量优先于项目名的 .env 文件，
    那么外部工具一旦改了系统环境变量，本项目仍会被污染——
    变量改名就白改了。项目自己写的配置必须比外部的更优先。
    """
    value = os.environ.get(primary_name)
    if value:
        return value, primary_name

    file_values = load_env_file()

    value = file_values.get(primary_name)
    if value:
        return value, primary_name

    if legacy_name:
        value = os.environ.get(legacy_name)
        if value:
            return value, legacy_name

        value = file_values.get(legacy_name)
        if value:
            return value, legacy_name

    return None, primary_name


def get(primary_name, legacy_name=None, default=None):
    """
    按隔离优先级取配置值。

    命中项目专属名时返回对应值；未命中才回退到通用兼容名；
    都没有则返回 default。
    """
    value, _ = _lookup(primary_name, legacy_name)
    return value if value else default


def get_api_key():
    return get(ENV_API_KEY, LEGACY_API_KEY)


def get_base_url(default=None):
    return get(ENV_BASE_URL, LEGACY_BASE_URL, default or DEFAULT_BASE_URL)


def get_model_name(default=None):
    return get(ENV_MODEL_NAME, LEGACY_MODEL_NAME, default or DEFAULT_MODEL_NAME)


def get_judge_model_name():
    """Judge 模型名：未单独配置时退回被测模型名。"""
    for primary, legacy in (("EVAL_JUDGE_MODEL_NAME", LEGACY_JUDGE_MODEL),
                (ENV_MODEL_NAME, LEGACY_MODEL_NAME)):
        value = get(primary, legacy)
        if value:
            return value
    return None


def api_key_env_name():
    """
    返回实际生效的 API Key 变量名。

    用途：向用户报错时要告诉他该配哪个变量，而不是笼统地列三个。
    """
    _, name = _lookup(ENV_API_KEY, LEGACY_API_KEY)
    return name


# ---------------------------
# 校验与诊断
# ---------------------------
def missing_required():
    """
    返回缺失的必填配置项，形如 [("EVAL_API_KEY", "API Key"), ...]。

    注意：base_url 与 model_name 缺失时不计入，
    因为两者有可用默认值，框架会用默认值继续工作。
    """
    missing = []

    for primary, legacy, label, required in REQUIRED_ITEMS:
        if required and not _lookup(primary, legacy)[0]:
            missing.append((primary, label))

    return missing


def check_env():
    """
    校验 API Key 是否配置。缺失时抛 EnvironmentError。

    报错信息直接给出可复制的操作步骤，而不是让用户自己猜该设哪个变量。
    """
    if not missing_required():
        return True

    lines = ["缺少必需的 API 配置：", ""]
    for name, label in missing_required():
        lines.append(f"  · {name}（{label}）")
    lines += [
        "",
        "推荐做法（只对本项目生效，不影响系统其他工具）：",
        f"  1. 复制 {os.path.join(PROJECT_ROOT, 'env.example')} 为 {ENV_FILE}",
        f"  2. 打开 {ENV_FILE}，填入上面列出的变量并保存",
        "",
        "临时用法（只对当前窗口生效，关闭窗口即失效）：",
        '  set EVAL_API_KEY "你的key"',
        f'  set {ENV_BASE_URL} "{DEFAULT_BASE_URL}"',
        f'  set {ENV_MODEL_NAME} "{DEFAULT_MODEL_NAME}"',
    ]

    raise EnvironmentError("\n".join(lines))


def describe():
    """
    返回脱敏后的配置摘要，用于 CLI 展示与排障。

    API Key 只显示前 4 位与末 2 位，中间固定用星号占位——
    排障时需要确认"是不是同一个 key"，但不应该把完整密钥打到屏幕上。
    """
    key = get_api_key()

    if key:
        masked = (f"{key[:4]}{'*' * 8}{key[-2:]}" if len(key) > 10 else "***")
    else:
        masked = "(未配置)"

    return {
        "env_file": ENV_FILE,
        "env_file_exists": os.path.exists(ENV_FILE),
        "api_key": masked,
        "api_key_env": api_key_env_name(),
        "base_url": get_base_url(),
        "model_name": get_model_name(),
        "judge_model_name": get_judge_model_name(),
        "missing": [n for n, _ in missing_required()],
    }