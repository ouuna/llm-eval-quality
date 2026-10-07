"""
HTTP Provider：通过 HTTP 调用被测系统
--------------------------------
为什么需要它
------------
在加 HTTP 服务层之前，Provider 只有两种：

  · RAGProvider   —— 直接 import 本地函数
  · MockProvider  —— 故障注入

真实项目里被测系统往往是**独立部署的服务**，
测试框架只能通过 HTTP 访问它。
没有这个 Provider，「接口测试」和「端到端评测」就接不起来：
接口测试用的是自己造的数据，评测用的是本地函数调用，
两者验证的不是同一条链路。

有了它，才能真正回答这个问题：
    隔着网络之后，评测结果还一致吗？

这一层会引入本地调用不存在的风险：
  · 网络错误、超时
  · 响应格式不符
  · 序列化开销
  · 服务端真实返回的空 context

这些都要能被显式区分，而不是统统变成「空回答」。
"""

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from eval.providers.sut import ProviderError, SUTResponse
from eval.schemas.result import TokenUsage

# 网络层可能抛出的异常全集。
#
# 为什么必须列全
# ------------
# `http.client` 会抛 ConnectionResetError（服务拒绝连接时）、
# RemoteDisconnected（服务端提前关闭）、
# ConnectionRefusedError 等，它们**不是** URLError 的子类
# （URLError 只是包装了部分情况），也不都是 TimeoutError。
#
# 漏掉任何一个，结果就是：网络出问题时评测进程直接崩，
# 而真正原因被一个看不懂的栈盖住。
import http.client
import socket

NETWORK_ERRORS = (
    urllib.error.URLError,
    TimeoutError,
    socket.timeout,
    socket.gaierror,
    ConnectionError,
    http.client.HTTPException,
    OSError,
)


class HTTPProvider:
    """
    通过 HTTP 调用被测系统。

    遵循 Provider 协议：ask(question) -> SUTResponse
    """

    name = "http_sut"

    def __init__(self, base_url: str = "http://127.0.0.1:8765",
                 timeout: float = 60.0,
                 path: str = "/ask",
                 rag_module=None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.path = path
        # 可选：直接注入被测模块，绕过网络（用于对照实验）
        self._direct = rag_module

    # ---- Provider 协议 ----
    def is_available(self) -> bool:
        """健康检查：服务是否活着。任何异常都视为不可用"""
        try:
            req = urllib.request.Request(f"{self.base_url}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            # 这里用宽泛的 except 是刻意的：
            # 探针的语义是「能不能用」，任何失败都等于不能用。
            # 细分类别对调用方没有价值。
            return False

    def ask(self, question: str) -> SUTResponse:
        start = time.perf_counter()

        if self._direct is not None:
            return self._ask_direct(question, start)

        payload = json.dumps(
            {"question": question}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{self.path}",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            # 服务端明确返回了错误状态。
            #
            # 这里要区分 4xx 与 5xx：
            #   4xx → 请求本身有问题（调用方该改）
            #   5xx → 服务端故障（该查服务）
            # 混为一谈会让排障时先怀疑自己。
            body = _safe_json(e.read())
            code = _error_code(body)
            return SUTResponse(
                answer="",
                latency_ms=int((time.perf_counter() - start) * 1000),
                error=f"HTTP {e.code} {code or ''}".strip(),
                raw={"status": e.code, "body": body},
            )
        except NETWORK_ERRORS as e:
            # 网络层失败：连不上、超时、连接被重置等。
            #
            # 早期这里只捕获 URLError 与 TimeoutError，
            # 结果服务不可达时抛出的 ConnectionResetError 直接穿透，
            # **整个评测进程崩掉**——
            # 表现是「跑测试时报了个看不懂的网络错误」，
            # 而真正原因（服务没起来）完全被掩盖。
            #
            # 关键：任何连接问题都应转成 error 状态返回，
            # 而不是抛异常。Provider 的约定是「永不抛，只标记」。
            return SUTResponse(
                answer="",
                latency_ms=int((time.perf_counter() - start) * 1000),
                error=f"{type(e).__name__}: {e}",
            )
        except OSError as e:
            # 兜底：socket 层还有其它 OSError
            # （如 ConnectionAbortedError、BrokenPipeError 的其它子类）
            return SUTResponse(
                answer="",
                latency_ms=int((time.perf_counter() - start) * 1000),
                error=f"{type(e).__name__}: {e}",
            )

        latency = int((time.perf_counter() - start) * 1000)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            # 响应不是 JSON：可能是代理返回了错误页、网关超时页等
            return SUTResponse(
                answer="",
                latency_ms=latency,
                error=f"响应不是合法 JSON：{e.msg}",
                raw={"body": raw[:500]},
            )

        if not isinstance(data, dict) or "answer" not in data:
            return SUTResponse(
                answer="",
                latency_ms=latency,
                error=("响应缺少 answer 字段"
                       f"（实际字段：{list(data)[:6] if isinstance(data, dict) else type(data).__name__}）"),
                raw=data,
            )

        contexts = data.get("contexts") or []
        if not isinstance(contexts, list):
            return SUTResponse(
                answer="",
                latency_ms=latency,
                error=f"contexts 应为列表，实际为 {type(contexts).__name__}",
                raw=data,
            )

        return SUTResponse(
            answer=data.get("answer") or "",
            contexts=[str(c) for c in contexts],
            latency_ms=latency,
            token_usage=TokenUsage.unavailable(),
            raw=data,
        )

    def _ask_direct(self, question, start) -> SUTResponse:
        """
        不走网络，直接调用被测模块。

        用途是**对照实验**：
        同一批用例分别用「本地直调」与「HTTP 调用」跑，
        如果指标差异明显，说明链路上有额外损耗
        （序列化、超时配置、字段丢失等）。
        """
        try:
            answer, contexts = self._direct.ask(question)
        except Exception as e:
            return SUTResponse(
                answer="",
                latency_ms=int((time.perf_counter() - start) * 1000),
                error=f"{type(e).__name__}: {e}",
            )
        return SUTResponse(
            answer=answer or "",
            contexts=list(contexts or []),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )


def _safe_json(raw: bytes):
    """错误响应体可能不是 JSON，不能因此再抛异常"""
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return raw.decode("utf-8", "replace")[:300]


def _error_code(body):
    if isinstance(body, dict):
        return (body.get("error") or {}).get("code")
    return None
