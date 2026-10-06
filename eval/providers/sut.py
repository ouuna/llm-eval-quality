"""
被测系统（SUT）抽象与 Provider
---------------------------------
把"被测对象"抽象为接口，使评测框架与具体实现解耦。

改造动机（修复审计问题 P0-7）
------------------------------
旧实现直接 `from app.rag import ask`，导致：
    - 无法替换 SUT（换成 LangChain、换成 REST API 调用都不行）
    - SUT 签名一变，评测层立即崩溃
    - 稳定性检测内部调 ask()，与外部调用路径耦合

新设计
------
    Provider（协议）
        ├── RAGProvider      本地 RAG 系统
        ├── MockProvider     故障注入用（Phase 8）
        └── 未来：LangChainProvider / HTTPProvider

所有 Provider 统一返回 SUTResponse，携带：
    - answer        模型回答
    - contexts      检索到的上下文
    - latency_ms    调用耗时
    - token_usage   token 与成本（不可用时为 None）
    - raw           原始响应，便于追溯

关于命名
--------
用Provider 而非 SUT 类名，因为 SUT（System Under Test）是"被测对象"
的泛称，而 Provider 强调"数据来源"。一个 Provider 可能是 SUT 本身，
也可能是 SUT 的远程调用入口。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol
import time
import json
import os

from eval.schemas.result import TokenUsage


# ============================================================
# 统一响应结构
# ============================================================
@dataclass
class SUTResponse:
    """
    被测系统统一响应

    设计要点：
      - latency_ms 与 token_usage 是一等公民，而非事后统计
        （需求第十二节：性能评测）
      - error 字段显式存在，调用失败时 status 由上层判定为 error
        而不是伪装成"空回答"
    """
    answer: str
    contexts: List[str] = field(default_factory=list)
    latency_ms: int = 0
    token_usage: TokenUsage = field(default_factory=TokenUsage.unavailable)
    raw: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def context_str(self) -> str:
        return "\n\n".join(self.contexts)


class Provider(Protocol):
    """
    被测系统提供者协议

    实现者只需提供 ask()，其余评测逻辑由框架负责。
    """

    name: str

    def ask(self, question: str) -> SUTResponse:
        """执行一次问答，返回统一响应"""
        ...

    def is_available(self) -> bool:
        """自检：配置是否完整、API 是否可达"""
        ...


# ============================================================
# 异常类型
# ============================================================
class ProviderError(Exception):
    """Provider 调用失败"""
    pass


class ProviderNotConfigured(ProviderError):
    """配置缺失（区别于调用失败）"""
    pass


# ============================================================
# 本地 RAG Provider
# ============================================================
class RAGProvider:
    """
    包装现有 app.rag.RAG 系统

    注意：这里用 import 而非继承，因为 app.rag 是被测系统而非
    评测框架的一部分，不应被框架"接管"。
    """

    name = "rag_local"

    def __init__(self, module_path: str = "app.rag"):
        self._module_path = module_path
        self._mod = None

    def _load(self):
        if self._mod is None:
            import importlib
            try:
                self._mod = importlib.import_module(self._module_path)
            except Exception as e:
                raise ProviderNotConfigured(
                    f"无法加载被测模块 {self._module_path}：{e}"
                ) from e
        return self._mod

    def is_available(self) -> bool:
        try:
            self._load()
            return True
        except Exception:
            return False

    def ask(self, question: str) -> SUTResponse:
        mod = self._load()
        start = time.perf_counter()

        try:
            answer, contexts = mod.ask(question)
        except Exception as e:
            return SUTResponse(
                answer="",
                latency_ms=int((time.perf_counter() - start) * 1000),
                error=f"{type(e).__name__}: {e}",
            )

        latency = int((time.perf_counter() - start) * 1000)

        # 尝试获取 token 用量（若被测系统暴露了接口）
        token_usage = TokenUsage.unavailable()
        getter = getattr(mod, "get_last_usage", None)
        if callable(getter):
            try:
                token_usage = TokenUsage.from_api(getter())
            except Exception:
                pass

        return SUTResponse(
            answer=answer or "",
            contexts=list(contexts or []),
            latency_ms=latency,
            token_usage=token_usage,
        )


# ============================================================
# 注册表
# ============================================================
_REGISTRY: Dict[str, type] = {
    "rag": RAGProvider,
}


def get_provider(name: str, **kwargs):
    """
    按名称获取 Provider

    参数
    ----
    name: rag | mock
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"未知 Provider：{name}，可用：{list(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def register_provider(name: str, cls: type):
    """注册自定义 Provider（便于扩展到 LangChain / HTTP 等）"""
    _REGISTRY[name] = cls
    return cls


def available_providers() -> List[str]:
    return list(_REGISTRY)
