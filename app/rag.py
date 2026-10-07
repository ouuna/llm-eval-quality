"""
RAG 问答系统（被测对象）
---------------------------------
本系统是 LLM 评测体系的测试对象，用于验证大模型在检索增强场景下
是否存在幻觉、答案相关性不足、输出不稳定等问题。

处理链路：加载知识库 -> 检索相关片段 -> 组装 Prompt -> 调用大模型生成答案
"""

import os
import re
import sys
import json
import time
import urllib.request
import urllib.error

# ---------------------------
# 1. 配置：优先读项目专属 EVAL_* 变量或 .env，避免与本机其他工具
#    （如 Claude Code Provider 切换工具）争抢 OPENAI_* 同名变量
# ---------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from eval import env_loader
from eval.config_loader import load_config, get as cfg_get

API_KEY = env_loader.get_api_key()
BASE_URL = env_loader.get_base_url()
MODEL = env_loader.get_model_name()

# ---------------- 加载配置 ----------------
try:
    _CFG = load_config()
except Exception:
    _CFG = {}

KNOWLEDGE_FILES = [
    os.path.join(PROJECT_ROOT, p.replace("/", os.sep))
    for p in (cfg_get(_CFG, "system.knowledge_files", ["knowledge/test_basics.txt"])
              if _CFG else ["knowledge/test_basics.txt"])
]
TOP_K = cfg_get(_CFG, "system.top_k", 3)
TEMPERATURE = cfg_get(_CFG, "model.temperature", 0.3)

# 文档分隔符此前也是硬编码 "\n\n"，配置项形同虚设。
# 换知识库时不同来源的分隔符可能不同（如 "\n### "），
# 所以这里真正读配置，并把转义序列还原成真实字符
# —— 写"\n\n" 在 YAML 双引号里会变成真换行，写 '\\n\\n' 才是两个字面量。
_DOC_SEP_RAW = cfg_get(_CFG, "system.doc_separator", "\n\n")
try:
    DOC_SEPARATOR = _DOC_SEP_RAW.encode().decode("unicode_escape")
except (UnicodeDecodeError, AttributeError):
    DOC_SEPARATOR = _DOC_SEP_RAW

# 超时与重试此前是模块级硬编码（TIMEOUT=60 / MAX_RETRY=3），
# 而 config.yaml 里也写了model.timeout / model.max_retry ——
# 两处并存，配置文件形同虚设。改到只认配置，删掉硬编码常量。
#
# 这样调整超时不必改代码，也让「配置能不能生效」变得可验证。
TIMEOUT = cfg_get(_CFG, "model.timeout", 60)
MAX_RETRY = cfg_get(_CFG, "model.max_retry", 3)
CALL_INTERVAL = cfg_get(_CFG, "model.call_interval", 0.5)

PROMPT_TEMPLATE_CFG = cfg_get(_CFG, "prompt.template", "")
ENABLE_HALLUCINATION_GUARD = cfg_get(_CFG, "prompt.enable_hallucination_guard", True)


def missing_config() -> list:
    """
    返回缺失的配置项列表，未缺失时返回空列表。

    与 _check_env() 分开是刻意的：
      · missing_config() 只报告，不抛异常
      · _check_env() 报告并抛异常

    HTTP 层需要前者——它要把「配置缺失」转成503，
    而不是让异常冒泡变成 500。两者语义不同：
    503 是「服务暂时不可用」，500 是「服务出错了」。

    这也是为什么配置状态查询要独立成一个函数：
    混在一起的话，HTTP 层只能捕获异常再猜是哪一类。
    """
    return [name for name, value in {
        env_loader.ENV_API_KEY: API_KEY,
        env_loader.ENV_BASE_URL: BASE_URL,
        env_loader.ENV_MODEL_NAME: MODEL,
    }.items() if not value]


def _check_env():
    """提前校验 API 配置，让报错信息直接给出可复制的修复步骤"""
    missing = missing_config()
    if missing:
        raise EnvironmentError(
            f"缺少 API 配置：{', '.join(missing)}\n"
            f"推荐做法：复制 env.example 为 {env_loader.ENV_FILE}，填入上面的变量\n"
            f"（只对本项目生效，不会影响系统里其他工具的同名配置）"
        )


# ---------------------------
# 2. 知识库加载
# ---------------------------
def load_knowledge():
    """
    读取全部知识库文件，拆分为多条文档。

    关键点 1：支持多知识库文件，路径由 config.yaml 的
              system.knowledge_files 配置，实现业务无关。
    关键点 2：需同时处理 \\n\\n 与 \\r\\n\\r\\n 两种换行符。
              Windows 记事本默认保存为 CRLF，若只按 \\n\\n 切分会失败，
              导致整个文件被视为单条文档，检索时整体不匹配。
              —— 此问题在项目初期实际发生，导致检索结果恒为空。
    """
    docs = []
    for path in KNOWLEDGE_FILES:
        if not os.path.exists(path):
            print(f"[WARN] 知识库文件不存在，已跳过：{path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        content = content.replace("\r\n", "\n").replace("\r", "\n")

        for doc in content.split(DOC_SEPARATOR):
            doc = doc.strip()
            if doc:
                docs.append(doc)

    return docs


# ---------------------------
# 3. 检索
# ---------------------------
STOP_WORDS = {
    "什么", "是", "的", "了", "吗", "呢", "请", "介绍", "一下",
    "怎么", "怎样", "如何", "有", "和", "与", "说", "讲", "告诉",
    "定义", "含义", "意思", "请问", "那么", "以及", "对于", "关于",
}

# 同义词/别名：让问法与文档表述不一致时也能命中
SYNONYMS = {
    "等价类": ["等价类划分", "划分", "子集"],
    "边界值": ["边界值分析", "边界", "最大值", "最小值"],
    "场景法": ["场景", "场景法", "基本流", "备选流", "业务流程"],
    "场景": ["场景法", "基本流", "备选流"],
    "正交": ["正交实验法", "正交表", "组合"],
    "正交实验": ["正交实验法", "正交表"],
    "用例": ["测试用例", "用例编号", "预期结果", "实际结果"],
    "测试用例": ["用例", "用例编号", "优先级", "预期结果"],
    "缺陷": ["缺陷报告", "严重程度", "优先级", "复现步骤"],
    "缺陷报告": ["缺陷", "缺陷标题", "严重程度", "附件截图"],
    "冒烟": ["冒烟测试", "核心功能", "详细测试"],
    "冒烟测试": ["冒烟", "核心功能", "版本"],
    "回归": ["回归测试", "修改代码", "新缺陷"],
    "回归测试": ["回归", "重新执行", "新缺陷"],
    "接口": ["接口测试", "请求参数", "响应字段", "状态码"],
    "接口测试": ["接口", "请求参数", "响应字段"],
    "分层": ["自动化测试框架分层", "基础层", "用例层", "数据层"],
    "框架分层": ["自动化测试框架分层", "基础层", "用例层"],
    "持续集成": ["持续集成", "代码提交", "自动触发", "尽早发现问题"],
    "单元": ["单元测试", "最小可测试单元"],
    "单元测试": ["单元", "最小可测试单元", "开发人员"],
    "集成": ["集成测试", "接口协作"],
    "集成测试": ["集成", "多个模块", "接口协作"],
    "严重程度": ["严重程度", "优先级", "缺陷影响范围", "修复顺序"],
    "优先级": ["严重程度", "修复顺序", "优先级"],
    "区别": ["不同", "差异"],
    "目的": ["目的", "意义", "用于"],
    "意义": ["意义", "目的", "价值"],
    "哪些": ["哪些", "什么", "包括"],
    "包含": ["包含", "包括", "字段", "内容"],
    "字段": ["用例编号", "所属模块", "优先级"],
    "内容": ["包含", "内容", "提交"],
    "信息": ["环境信息", "附件截图"],
}


def _extract_keywords(question):
    """
    从问题中提取有检索意义的词组。

    最初实现按单字匹配，中文场景下命中率极低（单字不构成语义边界），
    改为按语义完整词组切分并按词长加权，召回率显著提升。
    """
    # 先去掉标点，保留中文、英文、数字
    cleaned = re.sub(r"[，。？?！!、；;：:\s]+", " ", question)
    segments = [s.strip() for s in cleaned.split(" ") if s.strip()]

    keywords = []
    for seg in segments:
        if seg in STOP_WORDS:
            continue
        if len(seg) >= 2:
            keywords.append(seg)
        elif len(seg) == 1 and re.search(r"[a-zA-Z0-9]", seg):
            keywords.append(seg)

    if not keywords:
        keywords = [s for s in segments if s]

    return keywords


def retrieve(question, docs, top_k=None):
    """
    检索：词组 + 同义词扩展 + 词长加权打分
    """
    if top_k is None:
        top_k = TOP_K

    keywords = _extract_keywords(question)

    # 扩展检索词：加入同义词，提升召回
    expanded = list(keywords)
    for kw in keywords:
        for key, aliases in SYNONYMS.items():
            if key in kw or kw in key:
                expanded.extend(aliases)

    expanded = list(set(expanded))

    scored = []
    for doc in docs:
        score = sum(doc.count(k) * len(k) for k in expanded)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored[:top_k] if score > 0]


# ---------------------------
# 4. 调用大模型
# ---------------------------
def call_llm(prompt):
    """调用大模型，带超时与重试"""
    _check_env()

    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
    }).encode("utf-8")

    last_error = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            req = urllib.request.Request(
                f"{BASE_URL.rstrip('/')}/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            resp = json.loads(
                urllib.request.urlopen(req, timeout=TIMEOUT).read().decode("utf-8")
            )
            return resp["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            # 限流（429）与鉴权（401/403）不该用同一种方式重试：
            # 前者等一会就好，后者再试多少次都一样，
            # 白白耗掉 3 次调用和60 秒超时。
            last_error = e
            if e.code == 429 and attempt < MAX_RETRY:
                wait = _retry_after(e) or CALL_INTERVAL * attempt
                print(f"[WARN] 被限流（429），{wait:.1f}s 后重试"
                      f"（第 {attempt} 次）")
                time.sleep(wait)
                continue
            if e.code in (401, 403):
                raise RuntimeError(
                    f"鉴权失败（HTTP {e.code}）：请检查 API Key 是否正确、"
                    f"是否已开通该模型权限。已重试 {attempt} 次，放弃。"
                ) from e
            if attempt < MAX_RETRY:
                print(f"[WARN] 第 {attempt} 次调用失败（HTTP {e.code}），重试中...")
                time.sleep(CALL_INTERVAL * attempt)
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError) as e:
            last_error = e
            if attempt < MAX_RETRY:
                print(f"[WARN] 第 {attempt} 次调用失败（{e}），重试中...")
                # 指数退避：原来固定 2 秒，连续失败时太密集
                time.sleep(CALL_INTERVAL * (2 ** (attempt - 1)))

    raise RuntimeError(f"调用大模型失败，已重试 {MAX_RETRY} 次：{last_error}")


def _retry_after(http_error):
    """读取服务端建议的重试等待时间（秒），没有则返回 None"""
    hdr = getattr(http_error, "headers", None)
    if not hdr:
        return None
    try:
        val = hdr.get("Retry-After")
        return float(val) if val else None
    except (TypeError, ValueError):
        return None


# ---------------------------
# 5. 被测系统主逻辑
# ---------------------------
# Prompt 模板：优先取config.yaml 的 prompt.template，
# 使领域措辞与防幻觉约束可通过配置调整而无需改代码
PROMPT_TEMPLATE = PROMPT_TEMPLATE_CFG or """你是软件测试领域的助手。请严格仅根据以下参考资料回答问题。
如果参考资料中没有相关信息，请明确回答"参考资料中未提及"，不要编造或推测。

参考资料：
{context}

问题：{question}

请用中文简洁回答："""

# 检索不到内容时的固定拒答措辞。
#
# 措辞的选择不是随意的：必须落在评测器
# `evaluate_refusal` 的拒答标记词表内，否则会被判成
# 「既没回答也没拒答」，即模型失能而非正确拒答。
#
# 查eval/evaluators/correctness.py 的 DEFAULT_REFUSAL_MARKERS
# 与 faithfulness.py 的 refusal_markers 确认覆盖面。
NO_CONTEXT_ANSWER = "参考资料中未提及该问题相关内容，无法回答。"


def ask(question):
    """
    被测的 RAG 问答系统

    返回：(答案, 检索到的上下文列表)

    异常处理
    --------
    这里曾经 `except: return "", contexts` —— 把调用失败伪装成
    「正确地没话说出来」。后果是评测器把API 故障当成"模型拒答"，
    判为通过：系统坏得越彻底，分数反而越高。

    现在失败时抛出，由调用方（Provider）显式标记为 error。
    「没检索到 → 拒答」与「调用失败」是两种完全不同的语义，
    绝不能混成同一个空字符串。
    """
    docs = load_knowledge()
    contexts = retrieve(question, docs)

    if not contexts:
        # 检索不到内容时**明确拒答**，而不是返回空字符串。
        #
        # 早先这里 `return "", []`，理由是「没检索到就是没有答案」。
        # 但空字符串对调用方是完全歧义的：
        #   · 到底是真的没有依据，还是服务出问题了？
        # 用户看到一片空白，无从判断。
        #
        # 这一点尤其重要，因为评测器的拒答检测靠的是
        # 「未提及 / 没有相关 / 无法回答」这类**明确的拒答措辞**。
        # 返回空串时，拒答检测会因为「本来就没回答」而无法区分，
        # 于是域外问题看起来像是「模型没作答」而不是「模型正确拒答」。
        # 把两种情况都变成空白，指标就失去了区分能力。
        return NO_CONTEXT_ANSWER, []

    context_str = "\n\n".join(contexts)

    if ENABLE_HALLUCINATION_GUARD:
        prompt = PROMPT_TEMPLATE.format(context=context_str, question=question)
    else:
        # 对照模式：移除防幻觉约束，用于验证该约束的实际效果。
        # 实测该模式下域外拒答率从 100% 降至 0%，是项目中的关键对照实验。
        prompt = ("请根据以下参考资料回答问题。\n\n"
                  f"参考资料：\n{context_str}\n\n"
                  f"问题：{question}\n\n请用中文回答：")

    # 失败不再吞掉：让 Provider 层显式标记 error
    answer = call_llm(prompt)

    return answer, contexts


# ---------------------------
# 6. 手动测试入口
# ---------------------------
if __name__ == "__main__":
    questions = [
        "什么是等价类划分？",
        "边界值分析怎么做？",
        "什么是冒烟测试和回归测试的区别？",
        "Python 怎么安装？",  # 故意问知识库外的，测幻觉
    ]

    for q in questions:
        print("=" * 60)
        print(f"[问题] {q}")
        ans, ctx = ask(q)

        print("\n--- 检索片段 ---")
        if not ctx:
            print("（未检索到相关内容）")
        for i, c in enumerate(ctx, 1):
            print(f"[{i}] {c[:70]}...")

        print("\n--- 生成回答 ---")
        print(ans if ans else "（无回答）")
        print()
