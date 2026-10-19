"""双口径评测跑分器：同一标注题集分别取证「混合检索基线」与「KG 车道」。

口径（OpenSpec change lex-temporal-kg-lane tasks 5.2 / design.md）：
  - 基线 = fd-legal-search-mcp 现役工具的只读调用（JSON-RPC 2.0 over HTTP POST，
    initialize -> notifications/initialized -> tools/call，不依赖 mcp sdk）；
  - KG = fd_legal_kg.query 的检索/游走面（由调用方注入 kg_query_fn，离线测试用 fake）；
  - 本模块只负责「取证 + 落对照表」：verdict 不自动打分，留 None 由人工按每题
    verdict_rule 裁定后回填（spec 要求人工评阅）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel

QUESTION_KINDS: tuple[str, ...] = ('temporal', 'multihop')

DEFAULT_BASELINE_MCP_URL = 'http://cheap6.mesh.finddata.cn:31884/mcp'
MCP_PROTOCOL_VERSION = '2025-03-26'


def baseline_mcp_url() -> str:
    """基线 MCP 端点：LEXKG_BASELINE_MCP_URL 覆盖，默认现役 fd-legal-search-mcp。"""
    return os.environ.get('LEXKG_BASELINE_MCP_URL', '').strip() or DEFAULT_BASELINE_MCP_URL


class EvalQuestion(BaseModel):
    """金标题：id + 类别（时点法/多跳）+ 查询 + 标注口径 + 判定依据。

    额外字段（如草稿里的 baseline_note）静默忽略，草稿文件可直接喂给 load_questions。
    """

    model_config = {'extra': 'ignore'}

    id: str
    kind: Literal['temporal', 'multihop']
    query: str
    expectation: str
    verdict_rule: str


def load_questions(path: str | Path) -> list[EvalQuestion]:
    """从 {"questions": [...]} JSON 文件加载题集（非法 kind 由 pydantic 拒绝）。"""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    items = data.get('questions') if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError(f'{path}: expected a top-level "questions" list')
    return [EvalQuestion.model_validate(item) for item in items]


# ---------------------------------------------------------------------------
# 基线车道：fd-legal-search-mcp 的最小 JSON-RPC 2.0 客户端
# ---------------------------------------------------------------------------

class McpError(RuntimeError):
    """JSON-RPC 错误响应或传输层失败。"""


class McpBaselineClient:
    """fd-legal-search-mcp 客户端（httpx 实现，无 mcp sdk 依赖）。

    握手顺序：initialize（等响应，从响应头记录 MCP-Session-Id）->
    notifications/initialized（无 id 的通知，服务器通常回 202 空体）-> tools/call。
    响应体兼容纯 JSON 与 SSE 分帧两种 Content-Type（streamable HTTP 服务器可能以
    text/event-stream 回 JSON-RPC 消息）；后续请求必须回显 MCP-Session-Id。
    """

    def __init__(self, base_url: str, timeout: float = 60.0, trust_env: bool = False):
        self.base_url = base_url
        self._timeout = timeout
        self._trust_env = trust_env
        self._client: httpx.AsyncClient | None = None
        self._next_id = 1
        self._session_id: str | None = None
        self._initialized = False

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # trust_env 默认关：端点是内网 mesh / 回环，httpx 会经 urllib 吸收 macOS
            # 系统代理（如本机 clash 127.0.0.1:7892），把回环请求劫持到代理上。
            self._client = httpx.AsyncClient(timeout=self._timeout,
                                             trust_env=self._trust_env)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> 'McpBaselineClient':
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def _post(self, payload: dict[str, Any], expect_id: int | None) -> dict[str, Any] | None:
        headers = {
            'Accept': 'application/json, text/event-stream',
            'Content-Type': 'application/json',
        }
        if self._session_id:
            headers['MCP-Session-Id'] = self._session_id
        resp = await (await self._ensure_client()).post(
            self.base_url, json=payload, headers=headers
        )
        session_id = resp.headers.get('mcp-session-id')
        if session_id:
            self._session_id = session_id
        if resp.status_code >= 400:
            raise McpError(f'HTTP {resp.status_code}: {resp.text[:200]}')
        if not resp.content:
            return None  # 通知的合法回包（202 空体）
        if 'text/event-stream' in resp.headers.get('content-type', ''):
            for line in resp.text.splitlines():
                if not line.startswith('data:'):
                    continue
                try:
                    msg = json.loads(line[len('data:'):].strip())
                except json.JSONDecodeError:
                    continue
                if expect_id is None or msg.get('id') == expect_id:
                    return msg
            raise McpError(f'no JSON-RPC message with id={expect_id} in SSE response')
        return json.loads(resp.text)

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        request_id = self._next_id
        self._next_id += 1
        msg = await self._post(
            {'jsonrpc': '2.0', 'id': request_id, 'method': method, 'params': params},
            expect_id=request_id,
        )
        if msg is None or 'error' in msg:
            error = (msg or {}).get('error')
            raise McpError(f'{method}: {error or "empty response"}')
        return msg.get('result')

    async def _notify(self, method: str) -> None:
        await self._post({'jsonrpc': '2.0', 'method': method}, expect_id=None)

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        await self._request('initialize', {
            'protocolVersion': MCP_PROTOCOL_VERSION,
            'capabilities': {},
            'clientInfo': {'name': 'fd-legal-kg-evalrunner', 'version': '0.1.0'},
        })
        await self._notify('notifications/initialized')
        self._initialized = True

    async def call(self, tool: str, arguments: dict[str, Any]) -> list[str]:
        """调用一个 MCP 工具，返回 result.content 里全部文本块的 text。"""
        await self._ensure_initialized()
        result = await self._request('tools/call', {'name': tool, 'arguments': arguments})
        if not isinstance(result, dict):
            raise McpError(f'{tool}: malformed tools/call result')
        if result.get('isError'):
            raise McpError(f'{tool}: tool-level error: {result.get("content")}')
        texts: list[str] = []
        for block in result.get('content', []):
            if (isinstance(block, dict) and block.get('type') == 'text'
                    and isinstance(block.get('text'), str)):
                texts.append(block['text'])
        return texts


# ---------------------------------------------------------------------------
# 双口径跑分
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """单题对照行：两车道的原始证据；verdict 留空待人工按 verdict_rule 回填。"""

    question: EvalQuestion
    baseline_evidence: list[str] = field(default_factory=list)
    kg_evidence: list[dict[str, Any]] = field(default_factory=list)
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    verdict: str | None = None

    def to_dict(self) -> dict[str, Any]:
        q = self.question
        return {
            'id': q.id, 'kind': q.kind, 'query': q.query,
            'expectation': q.expectation, 'verdict_rule': q.verdict_rule,
            'baseline_evidence': list(self.baseline_evidence),
            'kg_evidence': list(self.kg_evidence),
            'ts': self.ts.isoformat(),
            'verdict': self.verdict,
        }


def _is_error_payload(item: Any) -> bool:
    return isinstance(item, dict) and set(item) == {'error'}


def _baseline_hit(result: EvalResult) -> bool:
    """基线命中 = 至少一条非空证据且不是 {"error": ...} 条目。"""
    for text in result.baseline_evidence:
        if not (text or '').strip():
            continue
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return True
        if not _is_error_payload(parsed):
            return True
    return False


def _kg_hit(result: EvalResult) -> bool:
    return any(not _is_error_payload(item) for item in result.kg_evidence)


@dataclass
class EvalReport:
    """逐题对照表：JSONL 落档 + 汇总计数（命中=该车道取证非空且非 error）。"""

    results: list[EvalResult] = field(default_factory=list)

    def to_jsonl(self, path: str | Path) -> None:
        lines = [json.dumps(r.to_dict(), ensure_ascii=False) for r in self.results]
        body = '\n'.join(lines) + ('\n' if lines else '')
        Path(path).write_text(body, encoding='utf-8')

    def summary(self) -> dict[str, Any]:
        def _stats(results: list[EvalResult]) -> dict[str, int]:
            return {
                'total': len(results),
                'baseline_hits': sum(1 for r in results if _baseline_hit(r)),
                'kg_hits': sum(1 for r in results if _kg_hit(r)),
            }

        by_kind = {kind: _stats([r for r in self.results if r.question.kind == kind])
                   for kind in QUESTION_KINDS}
        overall = _stats(self.results)
        return {'total': overall['total'],
                'baseline_hits': overall['baseline_hits'],
                'kg_hits': overall['kg_hits'],
                'by_kind': by_kind}


class BaselineClient(Protocol):
    async def call(self, tool: str, arguments: dict[str, Any]) -> list[str]: ...


KgQueryFn = Callable[[str, EvalQuestion], Awaitable[list[dict[str, Any]]]]

# 基线只读调用现役工具：时点题走混合检索；多跳题叠加指导案例检索
BASELINE_TOOLS: dict[str, tuple[str, ...]] = {
    'temporal': ('law_semantic_search',),
    'multihop': ('law_semantic_search', 'guide_cases_search'),
}

# 工具 -> 其 query 形参名（实测 fd-legal-search-mcp 4.1.1 签名）
BASELINE_ARG_NAMES: dict[str, str] = {
    'law_semantic_search': 'query',
    'law_search': 'title_query',
    'guide_cases_search': 'title_query',
    'wenshu_search': 'title_query',
}


@dataclass
class DualEvalRunner:
    """逐题双口径取证：单车道故障只污染该题该车道的证据，不影响跑批。"""

    baseline: BaselineClient
    kg_query_fn: KgQueryFn | None = None
    baseline_tools: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(BASELINE_TOOLS)
    )

    async def run(self, questions: Sequence[EvalQuestion], *,
                  baseline: bool = True, kg: bool = True) -> EvalReport:
        if kg and self.kg_query_fn is None:
            raise ValueError('kg_query_fn is required when the kg caliber is enabled')
        report = EvalReport()
        for question in questions:
            ts = datetime.now(timezone.utc)
            baseline_evidence = await self._run_baseline(question) if baseline else []
            kg_evidence = await self._run_kg(question) if kg else []
            report.results.append(EvalResult(
                question=question,
                baseline_evidence=baseline_evidence,
                kg_evidence=kg_evidence,
                ts=ts,
            ))
        return report

    async def _run_baseline(self, question: EvalQuestion) -> list[str]:
        tools = self.baseline_tools.get(question.kind, ('law_semantic_search',))
        evidence: list[str] = []
        for tool in tools:
            # 每个工具的参数名不同：law_semantic_search 收 query，
            # guide_cases_search / law_search 收 title_query。统一传 query 会让后者
            # 直接报 missing_argument，把「参数传错」伪装成「基线答不出」。
            arguments = {BASELINE_ARG_NAMES.get(tool, 'query'): question.query}
            try:
                evidence.extend(await self.baseline.call(tool, arguments))
            except Exception as exc:  # 单个工具故障只污染它自己的证据
                evidence.append(
                    json.dumps({'error': f'{tool}: {type(exc).__name__}: {exc}'},
                               ensure_ascii=False)
                )
        return evidence

    async def _run_kg(self, question: EvalQuestion) -> list[dict[str, Any]]:
        assert self.kg_query_fn is not None
        try:
            return list(await self.kg_query_fn(question.query, question))
        except Exception as exc:  # 同上：KG 车道故障逐题落证
            return [{'error': f'{type(exc).__name__}: {exc}'}]


def graph_kg_query_fn(kg: Any, limit: int = 10, max_hops: int = 2) -> KgQueryFn:
    """KG 车道默认适配（live）：search_facts 取证；多跳题再从首条事实主体 walk 补链。

    Evidence 条目即 FactView.to_dict()（含 fact / valid_at / provenance 等字段）。
    """
    from .query import LegalKGQuery

    assert isinstance(kg, LegalKGQuery)

    async def _fn(query: str, question: EvalQuestion) -> list[dict[str, Any]]:
        views = list(await kg.search_facts(query, limit=limit))
        if question.kind == 'multihop' and views:
            views += await kg.walk(views[0].subject, max_hops=max_hops)
        seen: set[tuple[str, str, str]] = set()
        evidence: list[dict[str, Any]] = []
        for view in views:
            key = (view.fact, view.subject, view.object)
            if key in seen:
                continue
            seen.add(key)
            evidence.append(view.to_dict())
        return evidence

    return _fn


def build_live_kg_query_fn(settings: Any, episodes_path: str | None = None) -> KgQueryFn:
    """组装 live KG 车道（需要 LEXKG_* 全量配置与可达的 FalkorDB）。

    episodes_path 给出语料快照时同时接上 provenance 懒解析：事实边只存 episode
    uuid，没有语料索引就无法还原成「出自哪份材料哪一段」。
    """
    from . import config as _config
    from .pipeline import build_graphiti
    from .query import GraphitiQueryBackend, LegalKGQuery
    from .supplier import corpus_index_for, load_episodes_jsonl

    assert isinstance(settings, _config.Settings)
    graphiti = build_graphiti(settings)
    corpus = corpus_index_for(load_episodes_jsonl(episodes_path)) if episodes_path else None
    return graph_kg_query_fn(
        LegalKGQuery(GraphitiQueryBackend(graphiti), corpus_index=corpus)
    )


# ---------------------------------------------------------------------------
# CLI: python -m fd_legal_kg.evalrunner --questions X --out Y [filters]
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog='python -m fd_legal_kg.evalrunner',
        description='双口径评测取证：混合检索基线（fd-legal-search-mcp）vs KG 车道',
    )
    ap.add_argument('--questions', required=True, help='题集 JSON（{"questions": [...]}）')
    ap.add_argument('--out', required=True, help='对照表输出 JSONL 路径')
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 题（0=全部）')
    ap.add_argument('--only-kind', choices=QUESTION_KINDS, default='', help='只跑该类别')
    ap.add_argument('--episodes', default='',
                    help='语料快照 jsonl（给出则接上 provenance 回指）')
    caliber = ap.add_mutually_exclusive_group()
    caliber.add_argument('--baseline-only', action='store_true', help='只跑基线车道')
    caliber.add_argument('--kg-only', action='store_true', help='只跑 KG 车道')
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    questions = load_questions(args.questions)
    if args.only_kind:
        questions = [q for q in questions if q.kind == args.only_kind]
    if args.limit:
        questions = questions[: args.limit]

    async def _run() -> EvalReport:
        async with McpBaselineClient(baseline_mcp_url()) as client:
            kg_fn: KgQueryFn | None = None
            if not args.baseline_only:
                from . import config
                settings = config.from_env()
                settings.apply_runtime_env()
                kg_fn = build_live_kg_query_fn(settings, episodes_path=args.episodes or None)
            runner = DualEvalRunner(client, kg_fn)
            return await runner.run(
                questions,
                baseline=not args.kg_only,
                kg=not args.baseline_only,
            )

    report = asyncio.run(_run())
    report.to_jsonl(args.out)
    print(json.dumps(report.summary(), ensure_ascii=False, indent=2))
    print(f'wrote {args.out} ({len(report.results)} questions)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
