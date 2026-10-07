"""
API 测试共享 fixture
-------------------
把服务启动、请求发送、响应解析收敛到一处，
让各个测试文件只关心「测什么」，不操心「怎么发请求」。

为什么不写在conftest.py 里
--------------------------
tests/conftest.py 是全局 fixture，作用域覆盖整个测试树。
而 HTTP 服务只在 tests/api/ 下需要——
放全局会让不相关的测试也看到这些 fixture，
反而干扰阅读。放在子目录的 conftest 里作用域刚好。
"""

import json
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

import pytest

from app.server import RAGRequestHandler, make_server


class ApiResponse:
    """把 HTTP 响应包一层，让断言写起来更清楚"""

    def __init__(self, status: int, body: Any, headers: Dict[str, str],
                 elapsed_ms: float):
        self.status = status
        self.body = body
        self.headers = headers
        self.elapsed_ms = elapsed_ms

    @property
    def is_success(self) -> bool:
        return 200 <= self.status < 300

    @property
    def error_code(self) -> Optional[str]:
        """错误响应里的业务错误码"""
        if isinstance(self.body, dict) and "error" in self.body:
            return self.body["error"].get("code")
        return None

    @property
    def error_message(self) -> Optional[str]:
        if isinstance(self.body, dict) and "error" in self.body:
            return self.body["error"].get("message")
        return None

    def __repr__(self) -> str:
        return (f"<ApiResponse {self.status} "
                f"{round(self.elapsed_ms)}ms {self.body!r}>")


class ApiClient:
    """极简 HTTP 客户端（零依赖）"""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def request(self, method: str, path: str,
                data: Any = None,
                raw_body: Optional[str] = None,
                content_type: Optional[str] = "application/json",
                timeout: float = 30.0) -> ApiResponse:
        import time

        if raw_body is not None:
            body = raw_body.encode("utf-8")
        elif data is not None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        else:
            body = None

        headers = {}
        if content_type:
            headers["Content-Type"] = content_type

        req = urllib.request.Request(
            f"{self.base_url}{path}", data=body, headers=headers,
            method=method)

        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read().decode("utf-8")
                status = resp.status
                resp_headers = dict(resp.headers)
        except urllib.error.HTTPError as e:
            payload = e.read().decode("utf-8")
            status = e.code
            resp_headers = dict(e.headers)
        elapsed = (time.perf_counter() - start) * 1000

        try:
            parsed = json.loads(payload) if payload else None
        except json.JSONDecodeError:
            # 响应不是 JSON 本身就是缺陷，保留原文供断言
            parsed = payload

        return ApiResponse(status, parsed, resp_headers, elapsed)

    def get(self, path: str, **kw) -> ApiResponse:
        return self.request("GET", path, **kw)

    def post(self, path: str, data: Any = None, **kw) -> ApiResponse:
        return self.request("POST", path, data=data, **kw)


# ============================================================
# 离线服务：不依赖 API Key
# ============================================================
@pytest.fixture(scope="module")
def api_server():
    """
    启动一个后台 HTTP 服务，整个模块共用。

    刻意用 port=0 让系统分配端口：
    固定端口会在并行跑测试时冲突，
    而端口冲突的报错与真实缺陷长得一样，很难分辨。
    """
    server = make_server(port=0, verbose=False)
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield ApiClient(f"http://127.0.0.1:{port}")

    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.fixture
def fault_injection():
    """
    故障注入上下文管理器。

    用法：
        with fault_injection("internal_error"):
            resp = client.post("/ask", {"question": "x"})
    """
    class _Injector:
        def __init__(self):
            self._prev = None

        def set(self, mode):
            self._prev = RAGRequestHandler._inject.get("mode")
            RAGRequestHandler._inject["mode"] = mode

        def clear(self):
            RAGRequestHandler._inject["mode"] = None

        def __call__(self, mode):
            return _Context(self, mode)

    class _Context:
        def __init__(self, inj, mode):
            self._inj = inj
            self._mode = mode

        def __enter__(self):
            self._inj.set(self._mode)
            return self

        def __exit__(self, *exc):
            self._inj.clear()
            return False

    inj = _Injector()
    yield inj
    inj.clear()


# ============================================================
# 需要真实 API 的服务
# ============================================================
@pytest.fixture(scope="module")
def live_api_server():
    """
    启动接入真实 LLM 的服务。

    没有 API 配置时跳过——不是失败。
    「没配key」和「代码有bug」是两种完全不同的事，
    混在一起会让 CI 红灯，看起来像坏了。
    """
    from eval import env_loader

    if env_loader.missing_required():
        pytest.skip("未配置 API Key，跳过真实 API 测试")

    try:
        from app import rag
    except Exception as e:
        pytest.skip(f"被测系统加载失败：{e}")

    if rag.missing_config():
        pytest.skip("被测系统 API 配置不完整")

    server = make_server(port=0, rag_module=rag, verbose=False)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield ApiClient(f"http://127.0.0.1:{port}")

    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
