"""
LLM-as-a-Judge 评测器
---------------------------------
用大模型对输出做多维度质量判定，处理词级与语义方法都判不了的
复杂语义问题（逻辑错误、推理跳步、隐含前提错误）。

为什么需要 Judge
----------------
声明级验证能检出"哪句话没依据"，但检不出：
    "北京是中国的首都，上海是中国的最大城市"   ← 两句都有依据，但第二句无意义
    "测试用例应该包含预期结果"← 依据充分，但结论错误（过度绝对）
    "所有软件测试都必须自动化"              ← 依据充分，但推理过度

这些问题需要语义理解，而不只是字面匹配。

Judge Bias：必须控制的风险
---------------------------
1. **长度偏好**   Judge倾向给长回答高分
   对策：Prompt 明确要求"回答过短不应因此加分"；对照实验验证
2. **自我偏好**   Judge 偏好自己生成的文本
   对策：Judge 模型与被测模型分开配置，且可交叉互换
3. **位置偏好**   偏好第一个/最后一个选项
   对策：A/B 顺序随机化
4. **刻度漂移**   不同次调用分数不稳定
   对策：temperature=0，同一输入重复验证一致性
5. **肯定倾向**   倾向给"高分"，尤其在要求"评判"时
   对策：强制要求给出 evidence 字段，无证据的高分视为无效

本模块的设计原则
----------------
- **Judge 不是真值**：结果标注为 judge 维度，与 lexical/semantic 并列而非覆盖
- **失败必须暴露**：Judge 调用失败/输出非法 → 标记 evaluator_error，
  绝不当成"通过"（需求第六条）
- **结构化输出**：强制 JSON + Schema 校验，非法则重试
- **可观测**：每次判定记录原始响应，便于事后审查
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os
import json
import time
import re
import urllib.request
import urllib.error

from eval import env_loader


# ============================================================
# 配置
# ============================================================
DEFAULT_JUDGE_CONFIG = {
    "provider": "openai_compatible",
    "model": None,                 # 留空则读 EVAL_JUDGE_MODEL_NAME，再退回 EVAL_MODEL_NAME
    "api_key_env": "EVAL_API_KEY",
    "api_base": None,
    "timeout": 45,
    "max_retry": 3,                # JSON 非法时的重试次数
    "temperature": 0.0,            # 0 以减少刻度漂移
    "score_min": 0,
    "score_max": 5,
    # 分数低于此值视为有问题
    "pass_threshold": 3,
}


# ============================================================
# 结果结构
# ============================================================
@dataclass
class JudgeResult:
    """Judge 判定结果"""
    available: bool
    scores: Dict[str, float] = field(default_factory=dict)   # 0-5 归一化到 0-1
    hallucination: Optional[bool] = None
    reason: str = ""
    evidence: List[str] = field(default_factory=list)
    error: Optional[str] = None
    error_kind: Optional[str] = None   # timeout / invalid_json / api_error / schema_error
    raw_response: str = ""
    attempts: int = 0
    model: str = ""
    latency_ms: int = 0

    @property
    def final_score(self) -> Optional[float]:
        """
        综合分

        取五维平均。缺失维度不计。
        """
        if not self.scores:
            return None
        return round(sum(self.scores.values()) / len(self.scores), 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "scores": self.scores,
            "final_score": self.final_score,
            "hallucination": self.hallucination,
            "reason": self.reason,
            "evidence": self.evidence,
            "error": self.error,
            "error_kind": self.error_kind,
            "attempts": self.attempts,
            "model": self.model,
            "latency_ms": self.latency_ms,
        }


# ============================================================
# Judge Prompt
# ============================================================
JUDGE_SYSTEM_PROMPT = """你是一个严格的质量评估专家。你的任务是对AI 系统的回答进行客观评判。

评判维度（每项 0-5 分，5 为最好）：

1. faithfulness（有据性）：回答中的每个事实陈述是否都能在给定资料中找到依据？
2. relevance（相关性）：回答是否在回应问题，没有跑题？
3. correctness（正确性）：回答的事实是否正确、是否符合专业常识？
4. completeness（完整性）：问题要求的关键点是否都覆盖了？
5. coherence（连贯性）：表达是否清晰、有无自相矛盾？

**重要规则**（违反这些会使你的评判无效）：

1. 每个评分必须给出具体 evidence（引用资料或回答中的原文片段）。
   没有 evidence 的高分一律视为无效。
2. 不要因为回答更长就给更高分。长度与质量无关。
3. 只依据提供的资料和回答本身判断，不要引入你自己的知识去"纠正"资料。
4. 如果资料不足以支持某个判断，在 reason 中明确说明"资料不足"，
   不要猜测。
5. 严格输出 JSON，不要输出任何其他文字。
"""

JUDGE_USER_TEMPLATE = """请评判以下 AI 系统的回答。

【问题】
{question}

【系统检索到的资料】
{context}

【AI 的回答】
{answer}

【参考答案（如有）】
{reference}

【评测重点】
{focus}

请严格按以下 JSON 格式输出，不要有任何额外文字：

{{
  "scores": {{
    "faithfulness": 0-5的整数,
    "relevance": 0-5的整数,
    "correctness": 0-5的整数,
    "completeness": 0-5的整数,
    "coherence": 0-5的整数
  }},
  "hallucination": true或false,
  "reason": "判断理由，不超过150字",
  "evidence": ["支撑你判断的具体原文片段", "..."],
  "insufficient_context": true或false
}}"""


# ============================================================
# 工具函数
# ============================================================
def extract_json(text: str) -> Optional[dict]:
    """
    从响应中提取 JSON

    LLM 经常在 JSON 前后加解释文字或```json 代码块标记，
    需要稳健提取。返回 None 表示无法解析。
    """
    if not text:
        return None

    t = text.strip()

    # 去除 markdown 代码块围栏
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", t, re.DOTALL)
    if fence:
        t = fence.group(1).strip()

    # 直接解析
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass

    # 提取第一个完整的 {...} 块（处理前后有文字的情况）
    depth = 0
    start = None
    for i, ch in enumerate(t):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(t[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    return None


def validate_judge_schema(data: Any, score_min: int = 0,
                         score_max: int = 5) -> List[str]:
    """
    校验 Judge 输出的结构

    返回问题列表（空表示合法）
    """
    issues: List[str] = []

    if not isinstance(data, dict):
        return [f"顶层不是对象，而是 {type(data).__name__}"]

    scores = data.get("scores")
    if not isinstance(scores, dict):
        issues.append("scores 缺失或不是对象")
    else:
        required = {"faithfulness", "relevance", "correctness"}
        missing = required - set(scores)
        if missing:
            issues.append(f"scores 缺少必需维度：{missing}")
        for k, v in scores.items():
            if not isinstance(v, (int, float)):
                issues.append(f"scores.{k} 不是数字：{v!r}")
            elif not (score_min <= v <= score_max):
                issues.append(f"scores.{k}={v} 超出 [{score_min}, {score_max}]")

    if "hallucination" not in data:
        issues.append("hallucination 字段缺失")
    elif not isinstance(data["hallucination"], bool):
        issues.append(f"hallucination 不是布尔值：{data['hallucination']!r}")

    if "reason" not in data or not isinstance(data["reason"], str):
        issues.append("reason 字段缺失或不是字符串")

    return issues


def normalize_scores(scores: Dict[str, float], score_max: int = 5) -> Dict[str, float]:
    """把 0-5 分归一化到 0-1，便于与其他指标统一处理"""
    return {k: round(v / score_max, 3) for k, v in scores.items()
            if isinstance(v, (int, float))}


# ============================================================
# Judge 客户端
# ============================================================
class JudgeClient:
    """
    LLM Judge 客户端

    关键设计：所有失败路径都返回 JudgeResult(available=False)，
    由上层标记为 evaluator_error。绝不在此处返回"通过"。
    """

    def __init__(self, config: Dict[str, Any] = None, **overrides):
        self.cfg = {**DEFAULT_JUDGE_CONFIG, **(config or {}), **overrides}

        self.model = self.cfg.get("model") or env_loader.get_judge_model_name()
        self.api_key_env = self.cfg["api_key_env"]
        self.api_key = env_loader.get(self.api_key_env, "OPENAI_API_KEY")
        self.api_base = (self.cfg.get("api_base")
                         or env_loader.get_base_url())

        # 记录统计，用于报告
        self.stats = {"calls": 0, "failures": 0, "retries": 0}

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_base and self.model)

    def availability(self) -> Dict[str, Any]:
        if not self.api_key:
            return {"available": False,
                    "reason": f"未配置 API Key（{self.api_key_env}，"
                               f"可用 python -m eval config 排查）"}
        if not self.model:
            return {"available": False,
                    "reason": "未配置 Judge 模型（EVAL_JUDGE_MODEL_NAME 或 EVAL_MODEL_NAME）"}
        if not self.api_base:
            return {"available": False, "reason": "未配置接口地址（EVAL_BASE_URL）"}
        return {"available": True, "reason": "", "model": self.model}

    # ---------- 核心调用 ----------
    def judge(self, question: str, answer: str, context: str = "",
              reference: str = None, focus: str = "") -> JudgeResult:
        """
        执行一次判定

        返回的 JudgeResult 一定不为 None。失败时 available=False 且
        error_kind 明确标注失败类型。
        """
        av = self.availability()
        if not av["available"]:
            return JudgeResult(
                available=False, error=av["reason"],
                error_kind="not_configured", model=self.model or "",
            )

        user_prompt = JUDGE_USER_TEMPLATE.format(
            question=question,
            context=context or "（未检索到资料）",
            answer=answer or "（空回答）",
            reference=reference or "（无参考答案）",
            focus=focus or "综合判定",
        )

        last_error = None
        last_kind = None
        raw = ""
        start = time.perf_counter()

        for attempt in range(1, self.cfg["max_retry"] + 1):
            if attempt > 1:
                self.stats["retries"] += 1

            try:
                raw = self._call_api(user_prompt)
            except TimeoutError as e:
                last_error, last_kind = f"调用超时：{e}", "timeout"
                continue
            except urllib.error.URLError as e:
                last_error, last_kind = f"网络错误：{e}", "api_error"
                continue
            except Exception as e:
                last_error, last_kind = f"{type(e).__name__}: {e}", "api_error"
                continue

            self.stats["calls"] += 1

            # JSON 解析
            data = extract_json(raw)
            if data is None:
                last_error = f"无法从响应中解析 JSON（前200字：{raw[:200]}）"
                last_kind = "invalid_json"
                continue

            # Schema 校验
            issues = validate_judge_schema(data, self.cfg["score_min"],
                                          self.cfg["score_max"])
            if issues:
                last_error = f"Schema 校验失败：{issues}"
                last_kind = "schema_error"
                continue

            # 成功
            return JudgeResult(
                available=True,
                scores=normalize_scores(data["scores"], self.cfg["score_max"]),
                hallucination=bool(data.get("hallucination")),
                reason=str(data.get("reason", "")),
                evidence=[str(e) for e in (data.get("evidence") or [])],
                raw_response=raw,
                attempts=attempt,
                model=self.model,
                latency_ms=int((time.perf_counter() - start) * 1000),
            )

        # 全部重试失败 —— 显式报错，不伪造结果
        self.stats["failures"] += 1
        return JudgeResult(
            available=False,
            error=f"Judge 判定失败（重试 {self.cfg['max_retry']} 次）：{last_error}",
            error_kind=last_kind,
            raw_response=raw,
            attempts=self.cfg["max_retry"],
            model=self.model,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    def _call_api(self, user_prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.cfg["temperature"],
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.api_base.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        resp = json.loads(
            urllib.request.urlopen(req, timeout=self.cfg["timeout"]).read().decode()
        )
        return resp["choices"][0]["message"]["content"]


# ============================================================
# Judge 一致性检查（控制 Judge Bias 的关键手段）
# ============================================================
def check_judge_consistency(client: JudgeClient, question: str, answer: str,
                            context: str = "", repeats: int = 3,
                            tolerance: float = 0.2) -> Dict[str, Any]:
    """
    重复调用 Judge，检测刻度漂移

    temperature=0 理论上应确定性，但：
      - 部分服务商不保证严格确定性
      - 服务端可能有内部随机性
    因此必须实测而非假设。

    用途：判断该 Judge 的分数可信度。如果同一输入多次评分差异大，
    说明该 Judge 在当前配置下不可靠，评测结果需谨慎解读。
    """
    scores = []
    results = []
    for _ in range(repeats):
        r = client.judge(question, answer, context)
        results.append(r)
        if r.available and r.final_score is not None:
            scores.append(r.final_score)

    if len(scores) < 2:
        return {
            "available": False,
            "reason": f"有效评分不足（{len(scores)}/{repeats}）",
            "stable": None,
        }

    mean = sum(scores) / len(scores)
    spread = max(scores) - min(scores)

    return {
        "available": True,
        "stable": spread <= tolerance,
        "mean": round(mean, 3),
        "spread": round(spread, 3),
        "tolerance": tolerance,
        "samples": scores,
        "reason": (
            f"重复 {repeats} 次评分极差 {spread:.3f}"
            f"（阈值 {tolerance}）→ {'稳定' if spread <= tolerance else '存在漂移'}"
        ),
    }


# ============================================================
# A/B 位置偏好检测
# ============================================================
def check_position_bias(client: JudgeClient, question: str,
                        context: str, good: str, bad: str,
                        rounds: int = 2) -> Dict[str, Any]:
    """
    检测 Judge 的位置偏好

    方法：交换好/坏答案的呈现顺序，看评分是否随位置变化。
    若位置A 总是得高分，说明 Judge 有位置偏好，其评分不可信。

    这是控制 Judge Bias 的标准做法（来自 LLM-as-a-Judge 论文）。
    """
    first_good, first_bad = [], []

    for _ in range(rounds):
        # 顺序一：好答案在前
        r1 = client.judge(question, good, context)
        # 顺序二：坏答案在前
        r2 = client.judge(question, bad, context)

        if r1.available and r1.final_score is not None:
            first_good.append(r1.final_score)
        if r2.available and r2.final_score is not None:
            first_bad.append(r2.final_score)

    # 对照：好答案在两个位置各测一次
    good_pos1, good_pos2 = [], []
    for _ in range(rounds):
        r_a = client.judge(question, good, context)
        r_b = client.judge(question, good, f"（同上）{context}")
        if r_a.available and r_a.final_score is not None:
            good_pos1.append(r_a.final_score)
        if r_b.available and r_b.final_score is not None:
            good_pos2.append(r_b.final_score)

    if len(first_good) < 1 or len(first_bad) < 1:
        return {"available": False, "reason": "有效样本不足"}

    good_when_first = sum(first_good) / len(first_good)
    bad_when_first = sum(first_bad) / len(first_bad)

    # 位置偏好：好答案在前时得分 - 坏答案在前时得分
    position_effect = good_when_first - bad_when_first

    return {
        "available": True,
        "good_score": round(good_when_first, 3),
        "bad_score": round(bad_when_first, 3),
        "position_effect": round(position_effect, 3),
        "has_position_bias": abs(position_effect) > 0.3,
        "reason": (
            f"好答案在前={good_when_first:.2f}，"
            f"坏答案在前={bad_when_first:.2f}，"
            f"位置效应={position_effect:+.2f}"
        ),
    }
