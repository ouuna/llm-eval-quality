"""
集成测试 fixture
-----------------
集成测试与单元测试的区别：**跨层**。

单元测试验证「一个函数对不对」，
集成测试验证「这些部件拼起来能不能跑通」。

本项目的链路有三层：
    HTTP 服务 → Provider → Evaluator → 报告/门禁

任何一层单独测都可能是绿的，拼起来却不通。
比如：
  · Provider 正常返回，但 HTTP 层把字段名改了→ 评测全错
  · Evaluator 正常，但 runner 传错了参数 → 指标算错
  · 报告能生成，但门禁拿到的指标名对不上

这些只有跑通全链路才能发现。
"""

import threading

import pytest

from app.server import make_server


@pytest.fixture(scope="module")
def sut_server():
    """
    启动接入被测系统的 HTTP 服务。

    与 tests/api/conftest.py 里的 api_server 不同：
    那个不注入 rag_module（测的是错误路径，无需真实依赖），
    这个必须接入真实被测系统，否则测的就不是端到端链路了。
    """
    from app import rag

    server = make_server(port=0, rag_module=rag, verbose=False)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}"

    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def http_sut_provider(sut_server):
    """
    通过 HTTP 调用被测系统的 Provider。

    这个 Provider 是本项目里**唯一**走完整网络链路的实现，
    因此它产出的延迟数据才是真实的端到端延迟。
    """
    from eval.providers.http_provider import HTTPProvider

    return HTTPProvider(base_url=sut_server, timeout=60)
