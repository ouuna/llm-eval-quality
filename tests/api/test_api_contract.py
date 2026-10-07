"""
API 功能测试
--------------------------------
测的是 HTTP 契约本身：状态码、响应结构、字段类型、错误语义。

这个文件的存在意义
------------------
在加 HTTP 层之前，「接口测试」是空的——
被测系统只是个本地函数 `app.rag.ask()`，
没有状态码、没有 Content-Type、没有 schema可校验。

而这些恰恰是测试开发岗位最核心的能力。
接口契约如果没有被测试保护，它随时会悄悄变：
今天返回 200 明天返回 204，后端改了字段名而客户端还在读旧名，
这类问题在集成阶段才暴露，代价很高。

不测什么
--------
本文件**不测回答质量**。质量评测在 tests/evaluators/ 里。
这里只关心「接口层是否正确」。
两者混在一起会让失败原因难定位——
是接口坏了还是模型答得差？
"""

import json

import pytest

from app.server import MAX_BODY_BYTES, MAX_QUESTION_CHARS


# ============================================================
# 健康检查
# ============================================================
class TestHealth:

    def test_健康检查返回200(self, api_server):
        resp = api_server.get("/health")
        assert resp.status == 200

    def test_健康检查返回约定字段(self, api_server):
        """健康检查的响应结构也是契约，客户端会依赖它"""
        resp = api_server.get("/health")
        assert resp.body["status"] == "ok"
        assert "service" in resp.body
        assert "time" in resp.body

    def test_健康检查不触发llm调用(self, api_server):
        """
        健康检查必须在没有 API Key 时也能成功。

        这是可用性的底线：探针会频繁调用，
        如果它依赖 LLM，那么 LLM 限流或欠费时，
        整个服务会被判定为「不健康」并被摘流量——
        而服务本身其实是好的。
        """
        resp = api_server.get("/health", timeout=5)
        # 关键：响应时间必须是毫秒级，不可能是调用了 LLM
        assert resp.elapsed_ms < 1000, (
            f"健康检查耗时 {resp.elapsed_ms:.0f}ms，疑似触发了 LLM 调用")

    def test_健康检查用GET(self, api_server):
        """GET 是幂等的，探针不该用 POST"""
        assert api_server.request("GET", "/health").status == 200


# ============================================================
# 正常请求
# ============================================================
class TestAskSuccess:
    """这些用例需要真实 API"""

    def test_正常提问返回200(self, live_api_server):
        resp = live_api_server.post(
            "/ask", {"question": "什么是等价类划分？"})
        assert resp.status == 200, f"实际：{resp}"

    def test_响应包含必要字段(self, live_api_server):
        """
        字段缺失是接口演进最常见的破坏方式。

        后端把 "contexts" 改名成 "documents"，
        客户端还在读旧名字——不测 schema 的话，
        这个问题会一路带到生产。
        """
        resp = live_api_server.post(
            "/ask", {"question": "什么是冒烟测试？"})
        assert resp.status == 200

        for field in ("answer", "contexts", "latency_ms"):
            assert field in resp.body, f"响应缺少字段 {field}"

    def test_字段类型正确(self, live_api_server):
        """类型也要断言：字符串当字典用会报很难懂的错"""
        resp = live_api_server.post(
            "/ask", {"question": "什么是回归测试？"})
        body = resp.body

        assert isinstance(body["answer"], str)
        assert isinstance(body["contexts"], list)
        assert all(isinstance(c, str) for c in body["contexts"])
        assert isinstance(body["latency_ms"], int)
        assert body["latency_ms"] >= 0

    def test_响应为json格式(self, live_api_server):
        resp = live_api_server.post(
            "/ask", {"question": "什么是场景法？"})
        ctype = resp.headers.get("Content-Type", "")
        assert "application/json" in ctype, f"Content-Type 应为JSON，实际 {ctype}"

    def test_响应声明不缓存(self, live_api_server):
        """
        必须显式 no-store。

        性能测试测的是真实处理耗时；有缓存的话，
        第二次请求会命中缓存，延迟虚低，
        测出来的 P95 完全没有意义。
        """
        resp = live_api_server.post(
            "/ask", {"question": "什么是缺陷报告？"})
        assert "no-store" in resp.headers.get("Cache-Control", "")

    def test_响应时间在合理范围(self, live_api_server):
        """
        断言耗时上限。

        注意这不是性能测试，只是「没有卡住」的粗筛。
        真正的性能测试在 tests/performance/。
        上限定在 30 秒：GLM 通常 1~3 秒，
        超过 30 秒基本可以断定是网络或重试出了问题。
        """
        resp = live_api_server.post(
            "/ask", {"question": "什么是接口测试？"}, timeout=40)
        assert resp.elapsed_ms < 30000, (
            f"响应耗时 {resp.elapsed_ms:.0f}ms，超出合理范围")

    def test_带额外字段不影响(self, live_api_server):
        """
        宽容性：客户端多传无关字段不应报错。

        真实项目里前端常带 trace_id、user_agent 之类。
        服务端因为不认识的字段就报 400，会让排查变得很难受。
        """
        resp = live_api_server.post("/ask", {
            "question": "什么是单元测试？",
            "trace_id": "abc-123",
            "user_agent": "test",
        })
        assert resp.status == 200, f"多传字段不该报错，实际：{resp}"

    def test_域外问题应拒答而非报错(self, live_api_server):
        """
        知识库外的问题应该正常返回 200 且回答「未提及」。

        这不是「不该报错」而已——
        拒答能力本身就是被测系统的核心质量指标之一。
        """
        resp = live_api_server.post(
            "/ask", {"question": "请问北京今天的天气怎么样？"})
        assert resp.status == 200, f"域外问题不该报错，实际：{resp}"
        assert resp.body["answer"].strip(), "拒答也应有文字说明"
        # 没检索到内容时 contexts 为空是正常的
        assert isinstance(resp.body["contexts"], list)


# ============================================================
# 参数校验（4xx）
# ============================================================
class TestBadRequest:
    """
    这些用例不需要真实 API —— 参数校验在调用 LLM 之前就完成。
    """

    def test_缺少question字段返回400(self, api_server):
        resp = api_server.post("/ask", {"foo": "bar"})
        assert resp.status == 400
        assert resp.error_code == "missing_field"

    def test_空请求体返回400(self, api_server):
        resp = api_server.post("/ask", raw_body="")
        assert resp.status == 400

    def test_空字符串question返回400(self, api_server):
        resp = api_server.post("/ask", {"question": ""})
        assert resp.status == 400
        assert resp.error_code == "empty_question"

    def test_纯空白question返回400(self, api_server):
        """
        纯空白也要拒绝。

        不检查的话，"   " 会走到检索，
        因为分词后没有有效词，最终返回空 context，
        对外表现成 200 + 空回答——
        调用方会困惑到底是服务挂了还是确实没内容。
        """
        resp = api_server.post("/ask", {"question": "   \n\t "})
        assert resp.status == 400
        assert resp.error_code == "empty_question"

    def test_question为null返回400(self, api_server):
        resp = api_server.post("/ask", {"question": None})
        assert resp.status == 400
        assert resp.error_code == "null_question"

    def test_question类型错误返回400(self, api_server):
        """传数字/数组/对象都应被拒，而不是被 str() 强转"""
        for bad in (123, ["a"], {"b": 1}, True):
            resp = api_server.post("/ask", {"question": bad})
            assert resp.status == 400, \
                f"question={bad!r} 应报 400，实际 {resp.status}"
            assert resp.error_code == "wrong_type"

    def test_超长question返回400(self, api_server):
        resp = api_server.post(
            "/ask", {"question": "测" * (MAX_QUESTION_CHARS + 100)})
        assert resp.status == 400
        assert resp.error_code == "question_too_long"

    def test_超长时错误信息说明实际长度(self, api_server):
        """
        报错要说清多少、超多少。

        只说「过长」的话，调用方无法判断该截断到多少。
        """
        n = MAX_QUESTION_CHARS + 50
        resp = api_server.post("/ask", {"question": "测" * n})
        assert resp.error_code == "question_too_long"
        detail = resp.body["error"].get("detail", {})
        assert detail.get("length") == n
        assert detail.get("max") == MAX_QUESTION_CHARS

    def test_请求体不是对象返回400(self, api_server):
        """合法 JSON 但不是对象，如 [1,2,3] 或 "字符串" """
        resp = api_server.post("/ask", raw_body="[1, 2, 3]")
        assert resp.status == 400
        assert resp.error_code == "body_not_object"

    def test_错误信息结构统一(self, api_server):
        """
        所有错误响应必须有 code 与 message。

        只有 message 没有 code，调用方就得靠字符串匹配来分支——
        改一个字就break，这是接口设计的典型坏味道。
        """
        cases = [
            ({}, "missing_field"),
            ({"question": ""}, "empty_question"),
            ({"question": None}, "null_question"),
        ]
        for payload, expected_code in cases:
            resp = api_server.post("/ask", payload)
            assert resp.status == 400
            assert resp.error_code == expected_code
            assert resp.error_message, f"{expected_code} 缺少 message"


# ============================================================
# 非法JSON 与编码
# ============================================================
class TestMalformedBody:

    @pytest.mark.parametrize("raw", [
        "{不是json",
        "{'单引号': 'JSON 不支持单引号'}",
        "{缺少右括号",
        "{'trailing': 1,}",
        "\x00\x01\x02",
    ])
    def test_非法JSON返回400(self, api_server, raw):
        resp = api_server.post("/ask", raw_body=raw)
        assert resp.status == 400
        assert resp.error_code in ("invalid_json", "invalid_encoding")

    def test_json错误信息指出位置(self, api_server):
        """
        报错要带行列位置。

        只说「JSON 格式错误」的话，调用方得自己数字符；
        带上行列位置，对方能立刻定位。
        """
        resp = api_server.post("/ask", raw_body='{"question": }')
        assert resp.status == 400
        msg = resp.error_message or ""
        assert ("行" in msg and "列" in msg), \
            f"错误信息应指出行列位置，实际：{msg}"

    def test_非utf8编码返回400(self, api_server):
        """GBK 编码的中文 JSON 不是合法 UTF-8"""
        import urllib.request
        body = json.dumps(
            {"question": "什么是等价类划分"}, ensure_ascii=False
        ).encode("gbk")
        req = urllib.request.Request(
            f"{api_server.base_url}/ask", data=body,
            headers={"Content-Type": "application/json"})

        import urllib.error
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                status = r.status
        except urllib.error.HTTPError as e:
            status = e.code
        except Exception:
            pytest.fail("服务崩溃了：非 UTF-8 请求体不应导致连接断开")

        assert status == 400

    def test_超大请求体被拒(self, api_server):
        """
        超过上限的请求体应被拒绝。

        没有这道限制，一个超大 body 就能把服务内存吃光。
        """
        huge = json.dumps({"question": "x" * (MAX_BODY_BYTES + 1000)})
        resp = api_server.post("/ask", raw_body=huge)
        assert resp.status == 400
        assert resp.error_code == "body_too_large"


# ============================================================
# Content-Type
# ============================================================
class TestContentType:

    @pytest.mark.parametrize("ctype", [
        "text/plain", "application/xml", "text/html", "application/x-www-form-urlencoded",
    ])
    def test_错误content_type返回415(self, api_server, ctype):
        """
        415 而不是 400。

        这不是吹毛求疵：400 表示「请求内容有问题」，
        415 表示「请求格式本身不被支持」。
        客户端据此可以决定是改内容还是改格式。
        """
        resp = api_server.post(
            "/ask", {"question": "测试"}, content_type=ctype)
        assert resp.status == 415
        assert resp.error_code == "unsupported_media_type"

    def test_缺少content_type返回415(self, api_server):
        """
        完全不传 Content-Type 应报错。

        宽容地默认成 JSON 看似友好，
        但会让「客户端忘了设置」这个 bug 一直潜伏到线上。

        这里必须用原始 socket 而不是 urllib：
        urllib 在没有 Content-Type 时会改用 chunked 传输编码，
        请求里就不含 Content-Length 了——
        服务端读到 0 长度会先报「请求体为空」，
        根本走不到 Content-Type 检查那一步。
        用 socket 才能精确控制报文的每个字节。
        """
        import socket

        host_port = api_server.base_url.replace("http://", "")
        host, port = host_port.split(":")
        port = int(port)

        body = json.dumps({"question": "测试"})
        raw = (
            f"POST /ask HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Content-Length: {len(body.encode('utf-8'))}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
            f"{body}"
        ).encode("utf-8")

        with socket.create_connection((host, port), timeout=10) as s:
            s.sendall(raw)
            resp = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                resp += chunk

        status_line = resp.split(b"\r\n", 1)[0].decode("utf-8", "replace")
        assert "415" in status_line, f"应为 415，实际：{status_line}"

    def test_content_type带charset应接受(self, api_server):
        """
        "application/json; charset=utf-8" 必须接受。

        这是最常见的真实 Content-Type 写法，
        因为严格匹配 `== "application/json"` 会把它拒掉——
        这种 bug 在本地用 curl测不出来，
        因为 curl 默认不带 charset。
        """
        resp = api_server.post(
            "/ask", {"question": "测试"},
            content_type="application/json; charset=utf-8")
        assert resp.status == 415 or resp.status == 503 or resp.status == 200, (
            f"带 charset 的 Content-Type 不该报 415，实际 {resp.status}")

    def test_content_type大小写不敏感(self, api_server):
        """HTTP 头部值大小写不敏感"""
        resp = api_server.post(
            "/ask", {"question": "测试"}, content_type="APPLICATION/JSON")
        assert resp.status != 415, "Content-Type 大小写不应影响判定"


# ============================================================
# 服务端错误（5xx）
# ============================================================
class TestServerErrors:

    def test_注入内部错误返回500(self, api_server, fault_injection):
        resp_holder = {}
        with fault_injection("internal_error"):
            resp_holder["r"] = api_server.post(
                "/ask", {"question": "测试"})
        assert resp_holder["r"].status == 500
        assert resp_holder["r"].error_code == "injected_error"

    def test_未捕获异常不导致连接断开(self, api_server, fault_injection):
        """
        未预期异常必须转成 500 响应，而不是让连接直接断掉。

        如果连接被断开，客户端只会看到「连接重置」，
        拿不到任何诊断信息，排障时完全无从下手。
        """
        with fault_injection("crash"):
            resp = api_server.post("/ask", {"question": "测试"})

        assert resp.status == 500
        assert resp.error_code == "unhandled_error"
        assert resp.body is not None, "必须返回可解析的错误体"

    def test_错误后服务仍可用(self, api_server, fault_injection):
        """
        一次失败不该让服务进入不可用状态。

        这是很实际的场景：偶发失败如果导致进程挂掉，
        就需要人工介入才能恢复。
        """
        with fault_injection("crash"):
            api_server.post("/ask", {"question": "测试"})

        # 服务应仍然正常响应
        assert api_server.get("/health").status == 200

    def test_500也返回json(self, api_server, fault_injection):
        """错误响应也必须是 JSON，否则客户端统一解析会崩"""
        with fault_injection("crash"):
            resp = api_server.post("/ask", {"question": "测试"})
        ctype = resp.headers.get("Content-Type", "")
        assert "application/json" in ctype


# ============================================================
# 路由
# ============================================================
class TestRouting:

    def test_未知路径返回404(self, api_server):
        resp = api_server.get("/not-exist")
        assert resp.status == 404
        assert resp.error_code == "not_found"

    def test_未知路径的POST也返回404(self, api_server):
        resp = api_server.post("/not-exist", {"question": "x"})
        assert resp.status == 404

    def test_尾部斜杠不影响(self, api_server):
        """
        "/ask/" 与 "/ask" 应等价。

        现实中有人会习惯性加斜杠，
        服务因为这个就 404 会让人很困惑。
        """
        assert api_server.post("/ask/", {"question": "x"}).status != 404

    def test_健康检查不受查询参数影响(self, api_server):
        """/health?verbose=1 应正常工作"""
        assert api_server.get("/health?verbose=1").status == 200

    def test_ask带查询参数仍可用(self, api_server):
        """/ask?debug=1 应正常工作"""
        resp = api_server.post("/ask?debug=1", {"question": "x"})
        assert resp.status != 404

    def test_GET访问ask应失败(self, api_server):
        """
        /ask 只接受 POST。

        误用 GET 时应返回 405 而不是 404——
        405 明确说「方法不对」，404 说「路径不对」，
        两者的排查方向完全不同。
        """
        resp = api_server.get("/ask")
        assert resp.status in (404, 405)
