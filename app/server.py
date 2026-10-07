"""
被测系统的 HTTP 服务层
--------------------------------
为什么需要这一层
----------------
原先SUT 只是一个本地函数 `app.rag.ask()`，
于是「接口测试」这件事**根本没有作用对象**：

  · 没有状态码可断言
  · 没有响应 schema 可校验
  · 没有 Content-Type 可检查
  · 超时行为无法测（函数调用要么返回要么抛异常）
  · 5xx 错误无法模拟

这些恰恰是测试开发岗位最核心的能力。
一个只会调本地函数的测试框架，撑不起「接口测试」这个说法。

所以给 SUT 套一层标准 HTTP 接口：
业务逻辑仍是 `app.rag.ask()`（不重复实现），
这层只负责 HTTP 语义——参数校验、状态码、响应格式。

零第三方依赖
------------
用标准库 `http.server`，不引入 Flask/FastAPI。
理由与项目其他地方一致：依赖越少，别人clone 下来就能跑，
且读得懂每一行在做什么。

端点
----
    GET  /health          健康检查，不触发 LLM 调用
    POST /ask             问答主接口
    POST /inject_fault    故障注入（仅 Mock 模式），供 API 测试用

错误约定
--------
刻意用不同状态码区分不同失败原因，测试才有意义：

    400  请求本身有问题（不是合法 JSON、缺字段、question 为空）
    413  question 超过长度限制
    415  Content-Type 不是 application/json
    500  内部错误（LLM 调用失败等）
    503  依赖不可用（未配置 API Key）

把「请求错误」和「服务错误」分开很重要：
前者是调用方的问题，后者是服务的问题，
混成500 会让排障时先怀疑自己。
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# 配置
# ============================================================
# 请求体大小上限（字节）。超过返回 413。
#
# 这个值必须有上限：没有上限的服务，一个超大 body
# 就能把内存吃光。真实服务都会设，这里也不例外。
MAX_BODY_BYTES = 64 * 1024          # 64KB

# 拒绝超大 body 时，最多丢弃多少字节。
#
# 客户端声称的 Content-Length 未必等于它实际发出的量
# （超时、断连、恶意构造都可能），无脑 read 会一直阻塞。
# 设个上限，丢弃不完就直接关连接——宁可让客户端重试，
# 也不能让服务端线程卡死。
DRAIN_MAX_BYTES = 1024 * 1024       # 1MB

# question 字段长度上限（字符）。超过返回 400。
MAX_QUESTION_CHARS = 2000


class ApiError(Exception):
    """带状态码的业务异常"""

    def __init__(self, status: int, code: str, message: str,
                 detail: Any = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.detail = detail

    def to_dict(self) -> Dict[str, Any]:
        d = {"error": {"code": self.code, "message": self.message}}
        if self.detail is not None:
            d["error"]["detail"] = self.detail
        return d


# ============================================================
# 请求处理
# ============================================================
def _validate_request(body: Any) -> str:
    """
    校验请求体，返回 question。

    每种失败都给出独立的 code，
    因为调用方需要区分「我发错了」和「服务坏了」。
    """
    if not isinstance(body, dict):
        raise ApiError(400, "body_not_object",
                       "请求体必须是 JSON 对象",
                       f"实际类型：{type(body).__name__}")

    if "question" not in body:
        raise ApiError(400, "missing_field",
                       "缺少必填字段 question",
                       {"required": ["question"]})

    question = body["question"]

    if question is None:
        raise ApiError(400, "null_question",
                       "question 不能为 null")

    if not isinstance(question, str):
        raise ApiError(400, "wrong_type",
                       f"question 必须是字符串，实际为 "
                       f"{type(question).__name__}")

    if not question.strip():
        raise ApiError(400, "empty_question",
                       "question 不能为空或只含空白字符")

    if len(question) > MAX_QUESTION_CHARS:
        raise ApiError(400, "question_too_long",
                       f"question 长度 {len(question)} 超过上限 "
                       f"{MAX_QUESTION_CHARS}",
                       {"length": len(question),
                        "max": MAX_QUESTION_CHARS})

    return question


def handle_ask(body: Any, rag_module=None) -> Dict[str, Any]:
    """
    处理 /ask 请求（与 HTTP 解耦，便于直接单测）。

    参数
    ----
    body       已解析的 JSON
    rag_module 被测模块，None 时自动加载 app.rag

    返回
    ----
    成功：{"answer":..., "contexts": [...], "latency_ms":...}
    失败：抛 ApiError
    """
    question = _validate_request(body)

    if rag_module is None:
        try:
            from app import rag as rag_module
        except Exception as e:
            raise ApiError(503, "sut_unavailable",
                           f"被测系统加载失败：{e}")

    # 配置缺失要在调用前拦下——
    # 否则会变成一次耗时的API 调用后才报错。
    # 用missing_config() 而不是捕获 _check_env() 的异常：
    # 两者语义不同，503 表示「暂时不可用」，
    # 混成 500 会让调用方误以为是自己请求的问题。
    if hasattr(rag_module, "missing_config"):
        env_missing = rag_module.missing_config()
        if env_missing:
            raise ApiError(503, "not_configured",
                           f"API 配置缺失：{', '.join(env_missing)}")

    start = time.perf_counter()
    try:
        answer, contexts = rag_module.ask(question)
    except Exception as e:
        # 内部错误统一 500，并把异常类型放进去。
        # 刻意不返回堆栈：生产服务不该把内部路径暴露出去。
        raise ApiError(500, "internal_error",
                       f"生成回答失败：{type(e).__name__}: {e}")

    latency = int((time.perf_counter() - start) * 1000)

    return {
        "answer": answer or "",
        "contexts": list(contexts or []),
        "latency_ms": latency,
    }


# ============================================================
# HTTP 处理器
# ============================================================
class RAGRequestHandler(BaseHTTPRequestHandler):
    """
    HTTP 请求处理器。

    刻意保持轻量：所有业务逻辑在 handle_ask 里，
    这里只做 HTTP 语义的翻译（状态码、Header、JSON 编码）。
    这样业务逻辑可以脱离 HTTP 直接测试。
    """

    # 由 serve() 注入
    rag_module: Any = None
    verbose: bool = False
    # 故障注入开关，供 API 测试构造 5xx
    _inject: Dict[str, Any] = {"mode": None}

    # 默认不打印访问日志，测试时会很吵
    def log_message(self, fmt, *args):
        if self.verbose:
            super().log_message(fmt, *args)

    # ---- 响应工具 ----
    def _send(self, status: int, payload: Dict[str, Any],
              close: bool = False):
        """
        发送 JSON 响应。

        close
        ----
        拒绝一个请求后必须显式关闭连接。

        原因：客户端已经把整个 body 发过来了，
        服务端却只回了个错误就结束响应、继续读下一行——
        socket 上残留的 body 字节会被当成下一个请求的请求行，
        于是客户端看到的是「连接被重置」而不是那个本该返回的 400。

        这个 bug 的表现极具迷惑性：
        单发一次永远正常，批量跑时随机失败，
        而且失败点出现在**下一个**测试上，与真正的原因毫无关联。
        """
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        # 显式声明不缓存：性能测试要测的是真实处理耗时，
        # 缓存会让第二次请求的延迟虚低。
        self.send_header("Cache-Control", "no-store")
        if close:
            # 告诉客户端「别再往这个连接上发东西了」
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(raw)

    def _drain_pending_body(self):
        """丢弃请求头里声明但未读取的 body（未知路径时用）"""
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (ValueError, TypeError):
            return
        if 0 < length <= DRAIN_MAX_BYTES:
            self._drain(length)

    def _drain(self, length: int):
        """
        丢弃已经到达的请求体，让连接回到干净状态。

        分块读取且设上限保护：客户端声称的 Content-Length 可能
        远大于实际发送量（或者压根不发），无脑read 会一直阻塞。
        """
        remaining = min(length, DRAIN_MAX_BYTES)
        while remaining > 0:
            chunk = self.rfile.read(min(8192, remaining))
            if not chunk:
                break
            remaining -= len(chunk)

    def _read_body(self) -> Any:
        """读取并解析请求体，所有失败都归为 400"""
        ctype = self.headers.get("Content-Type", "")
        if "application/json" not in ctype.lower():
            # body 同样要读走：客户端已经写进来了，
            # 不读走会污染连接上的下一个请求。见_drain 的说明。
            try:
                length = int(self.headers.get("Content-Length", 0))
                if 0 < length <= DRAIN_MAX_BYTES:
                    self._drain(length)
            except ValueError:
                pass
            raise ApiError(415, "unsupported_media_type",
                           f"Content-Type 必须是 application/json，"
                           f"实际为 {ctype or '（未提供）'}")

        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            raise ApiError(400, "invalid_content_length",
                           "Content-Length 不是合法数字")

        if length > MAX_BODY_BYTES:
            # 先把 body 读走再拒绝。
            #
            # 这是 keep-alive 下的经典陷阱：
            # 客户端已经把整个 body 写进了 socket，
            # 服务端若直接回错并关闭连接，客户端可能还在写，
            # 就会收到 RST——它看到的是「连接被重置」，
            # 而不是一个本该返回的 400。
            #
            # 更糟的是客户端的连接池会把这个坏连接放回去复用，
            # 于是**下一次**无关的请求也跟着失败。
            # 表现出来就是「单发正常、批量跑随机挂」，
            # 而且挂的是哪个测试完全随机，极难定位。
            self._drain(length)
            raise ApiError(400, "body_too_large",
                           f"请求体 {length} 字节超过上限 "
                           f"{MAX_BODY_BYTES}",
                           {"max_bytes": MAX_BODY_BYTES})

        if length == 0:
            raise ApiError(400, "empty_body", "请求体为空")

        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError:
            raise ApiError(400, "invalid_encoding",
                           "请求体不是合法的 UTF-8")
        except json.JSONDecodeError as e:
            raise ApiError(400, "invalid_json",
                           f"请求体不是合法 JSON：{e.msg}（行 {e.lineno} 列 {e.colno}）")

    # ---- 路由 ----
    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"

        if path == "/health":
            # 健康检查刻意不触发 LLM 调用——
            # 否则探针会消耗配额，且LLM 挂了会导致「服务本身正常」被误判为故障。
            self._send(200, {
                "status": "ok",
                "service": "rag-qa",
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            return

        self._send(404, {"error": {"code": "not_found",
                                    "message": f"未知路径 {path}"}},
                   close=True)

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"

        try:
            if path == "/ask":
                body = self._read_body()

                # 故障注入：仅测试用，让 API 测试能稳定复现 5xx
                mode = self._inject.get("mode")
                if mode == "internal_error":
                    raise ApiError(500, "injected_error",
                                   "人为注入的内部错误（测试用）")
                if mode == "crash":
                    #模拟未捕获异常，验证兜底逻辑
                    raise RuntimeError("人为注入的未捕获异常（测试用）")

                result = handle_ask(body, self.rag_module)
                self._send(200, result)
                return

            # 未知路径：body 完全没读，连接上必然有残留。
            # 不关连接的话，残留字节会被当成下一个请求的请求行，
            # 客户端下一次请求就会收到「连接被重置」。
            #
            # 这个问题在测试里的表现极具迷惑性：
            # 本用例单跑必过，全量跑偶发失败，
            # 而且失败的可能是**另一个**测试——真凶在这里。
            self._drain_pending_body()
            self._send(404, {"error": {"code": "not_found",
                                        "message": f"未知路径 {path}"}},
                       close=True)

        except ApiError as e:
            # 错误一律关闭连接：请求体可能已被读入一部分，
            # 留在连接上的字节会污染下一个请求。
            self._send(e.status, e.to_dict(), close=True)
        except Exception as e:
            # 兜底：任何未预期异常都不能让连接直接断掉，
            # 否则客户端只会看到"连接重置"，拿不到任何诊断信息。
            self._send(500, {
                "error": {"code": "unhandled_error",
                          "message": f"{type(e).__name__}: {e}"}},
                close=True)


# ============================================================
# 启动
# ============================================================
def make_server(port: int = 0, rag_module=None,
                verbose: bool = False) -> HTTPServer:
    """
    创建 HTTP 服务（不启动）。

    port=0 表示由系统分配空闲端口——
    性能测试与并行测试需要同时跑多个实例，
    写死端口会冲突。

    刻意不启用 ThreadingHTTPServer：
    默认的单线程模型能真实反映「串行处理能力」，
    而性能测试里的并发需要能控制并发度。
    多线程版本留给压测时按需开启。
    """
    handler = type("BoundHandler", (RAGRequestHandler,), {
        "rag_module": rag_module,
        "verbose": verbose,
    })
    return HTTPServer(("127.0.0.1", port), handler)


def serve(port: int = 8765, verbose: bool = True):
    """启动服务并阻塞（手动运行用）"""
    try:
        from app import rag
    except Exception as e:
        print(f"[ERROR] 无法加载被测系统：{e}")
        raise

    server = make_server(port, rag, verbose)
    host, actual = server.server_address[0], server.server_address[1]

    print("=" * 62)
    print("RAG 被测系统 HTTP 服务")
    print("=" * 62)
    print(f"  地址       http://{host}:{actual}")
    print(f"  健康检查   GET  /health")
    print(f"  问答接口   POST /ask")
    print(f"  请求体上限 {MAX_BODY_BYTES} 字节")
    print(f"  提问上限   {MAX_QUESTION_CHARS} 字符")
    print()
    print("  试一下：")
    print(f'curl -s -X POST http://{host}:{actual}/ask \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"question":"什么是等价类划分？"}\'')
    print()
    print("  Ctrl+C 停止")
    print("=" * 62)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    serve(port=int(sys.argv[1]) if len(sys.argv) > 1 else 8765)
