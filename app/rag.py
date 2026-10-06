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

API_KEY = env_loader.get_api_key()
BASE_URL = env_loader.get_base_url()
MODEL = env_loader.get_model_name()
TIMEOUT = 60
MAX_RETRY = 3

sys.path.insert(0, os.path.join(PROJECT_ROOT, "tests"))

# ---------------- 加载配置 ----------------
try:
    from config_loader import load_config, get as cfg_get
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
PROMPT_TEMPLATE_CFG = cfg_get(_CFG, "prompt.template", "")
ENABLE_HALLUCINATION_GUARD = cfg_get(_CFG, "prompt.enable_hallucination_guard", True)


def _check_env():
    """提前校验 API 配置，让报错信息直接给出可复制的修复步骤"""
    missing = [n for n, v in {
        env_loader.ENV_API_KEY: API_KEY,
        env_loader.ENV_BASE_URL: BASE_URL,
        env_loader.ENV_MODEL_NAME: MODEL,
    }.items() if not v]

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

        for doc in content.split("\n\n"):
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
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError) as e:
            last_error = e
            if attempt < MAX_RETRY:
                print(f"[WARN] 第 {attempt} 次调用失败（{e}），重试中...")
                time.sleep(2)

    raise RuntimeError(f"调用大模型失败，已重试 {MAX_RETRY} 次：{last_error}")


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


def ask(question):
    """
    被测的 RAG 问答系统

    返回：(答案, 检索到的上下文列表)
    """
    docs = load_knowledge()
    contexts = retrieve(question, docs)

    if not contexts:
        return "", []

    context_str = "\n\n".join(contexts)

    if ENABLE_HALLUCINATION_GUARD:
        prompt = PROMPT_TEMPLATE.format(context=context_str, question=question)
    else:
        # 对照模式：移除防幻觉约束，用于验证该约束的实际效果。
        # 实测该模式下域外拒答率从 100% 降至 0%，是项目中的关键对照实验。
        prompt = ("请根据以下参考资料回答问题。\n\n"
                  f"参考资料：\n{context_str}\n\n"
                  f"问题：{question}\n\n请用中文回答：")

    try:
        answer = call_llm(prompt)
    except Exception as e:
        print(f"[ERROR] 生成回答失败：{e}")
        return "", contexts

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
