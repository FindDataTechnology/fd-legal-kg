"""McpBaselineClient：对本地 stub JSON-RPC 服务器验证握手顺序与文本块抽取。

仅回环地址（127.0.0.1）上的 stdlib http.server，不触外网。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fd_legal_kg.evalrunner import McpBaselineClient, McpError

TOOL_TEXTS = ['{"hits": [{"title": "民法典"}]}', '{"hits": []}']


def _make_handler(recorder: list[dict], sse: bool, tool_error: bool):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            message = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
            recorder.append({
                'method': message.get('method'),
                'id': message.get('id', '<absent>'),
                'session': self.headers.get('MCP-Session-Id'),
            })
            if message['method'] == 'initialize':
                self._respond(
                    {'jsonrpc': '2.0', 'id': message['id'],
                     'result': {'protocolVersion': '2025-03-26', 'capabilities': {},
                                'serverInfo': {'name': 'stub', 'version': '0'}}},
                    extra_headers={'MCP-Session-Id': 'stub-session-1'},
                )
            elif message['method'] == 'notifications/initialized':
                self._respond(None)  # 通知：202 空体
            elif message['method'] == 'tools/call':
                if tool_error:
                    self._respond({'jsonrpc': '2.0', 'id': message['id'],
                                   'error': {'code': -32000, 'message': 'boom'}})
                    return
                self._respond({
                    'jsonrpc': '2.0', 'id': message['id'],
                    'result': {'content': [
                        {'type': 'text', 'text': TOOL_TEXTS[0]},
                        {'type': 'resource', 'resource': {'uri': 'law://1'}},
                        {'type': 'text', 'text': TOOL_TEXTS[1]},
                    ]},
                })
            else:
                self._respond({'jsonrpc': '2.0', 'id': message.get('id'),
                               'error': {'code': -32601, 'message': 'method not found'}})

        def _respond(self, payload, extra_headers=None):
            if payload is None:
                body = b''
                content_type = 'application/json'
            elif sse:
                body = f'data: {json.dumps(payload)}\n\n'.encode()
                content_type = 'text/event-stream'
            else:
                body = json.dumps(payload).encode()
                content_type = 'application/json'
            self.send_response(202 if payload is None else 200)
            self.send_header('Content-Type', content_type)
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


@pytest.fixture
def rpc_stub():
    servers: list[ThreadingHTTPServer] = []

    def _start(sse: bool = False, tool_error: bool = False) -> tuple[str, list[dict]]:
        recorder: list[dict] = []
        server = ThreadingHTTPServer(('127.0.0.1', 0), _make_handler(recorder, sse, tool_error))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f'http://127.0.0.1:{server.server_port}/mcp', recorder

    yield _start
    for server in servers:
        server.shutdown()
        server.server_close()


async def test_handshake_order_session_echo_and_text_extraction(rpc_stub):
    url, recorder = rpc_stub()
    async with McpBaselineClient(url) as client:
        texts = await client.call('law_semantic_search', {'query': '民法典'})
        await client.call('guide_cases_search', {'query': '跳单'})
    # initialize 在前、通知居中（无 id）、tools/call 在后；initialize 只发生一次
    assert [m['method'] for m in recorder] == [
        'initialize', 'notifications/initialized', 'tools/call', 'tools/call',
    ]
    assert recorder[1]['id'] == '<absent>'
    # MCP-Session-Id 从 initialize 响应头取得并在后续请求回显
    assert recorder[0]['session'] is None
    assert recorder[2]['session'] == 'stub-session-1'
    assert recorder[3]['session'] == 'stub-session-1'
    # 只抽文本块，resource 块被忽略
    assert texts == TOOL_TEXTS


async def test_sse_framed_response_supported(rpc_stub):
    url, recorder = rpc_stub(sse=True)
    client = McpBaselineClient(url)
    texts = await client.call('law_semantic_search', {'query': '民法典'})
    await client.aclose()
    assert texts == TOOL_TEXTS
    assert len(recorder) == 3


async def test_jsonrpc_error_raises(rpc_stub):
    url, _ = rpc_stub(tool_error=True)
    client = McpBaselineClient(url)
    with pytest.raises(McpError, match='boom'):
        await client.call('law_semantic_search', {'query': '民法典'})
    await client.aclose()
