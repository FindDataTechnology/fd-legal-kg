"""law_db 只读供应：指导性案例全文 + 法规条文级切片 -> Episode。

零写入保证（belt and braces）：
  1) 连接层：psycopg 连接串带 options='-c default_transaction_read_only=on'，
     服务端拒绝任何写入；
  2) 语句层：ReadOnlyGuard 对每条即将发送的 SQL 做静态断言，命中
     INSERT/UPDATE/DELETE/DDL/MERGE/COPY 等直接抛 ReadOnlyViolation。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

PSYCOPG_OPTIONS = '-c default_transaction_read_only=on'

# ---------------------------------------------------------------------------
# SQL 常量块（2026-10-18 已对 live law_db 核对）：
#   guide_cases(source_id PK, case_no, doc_type, title, listed_at, publish_at,
#               source, source_url, fetched_at, content_hash, keyword, gist,
#               facts, result, reasoning, statutes, body, reference_status,
#               reference_note, loaded_at)
#   laws(id PK, src_db, src_id, source_id, category, title, publish, expiry,
#        type, status, file_path, content, content_status, quality_flag,
#        source_note, created_at)      —— 无 law_articles 表，条文在 laws.content 内
#   law_chunks_v2(chunk_id, law_id, chunk_no, chunk_text, char_start, char_end,
#                category, title, publish, embedded, created_at)
# 法规条级切片由本模块在读取侧做（第X条正则 + char span），不依赖 chunk 表。
# ---------------------------------------------------------------------------
GUIDE_CASE_SQL = """
SELECT
    gc.source_id     AS stable_id,
    gc.case_no       AS case_no,
    gc.title         AS doc_title,
    gc.publish_at    AS publish_at,
    gc.listed_at     AS listed_at,
    coalesce(gc.gist, '')    AS gist,
    coalesce(gc.facts, '')   AS facts,
    coalesce(gc.result, '')  AS result,
    coalesce(gc.reasoning, '') AS reasoning,
    coalesce(gc.statutes, '') AS statutes,
    coalesce(gc.body, '')    AS body
FROM guide_cases gc
ORDER BY gc.case_no NULLS LAST, gc.source_id
LIMIT %(limit)s
"""

# 试点选取的四部基本法（id 为 law_db 实测值；cap 为条级切片上限，合计约 207 条）
PILOT_STATUTES: list[tuple[int, str, int]] = [
    (18726, '中华人民共和国刑法', 55),
    (69695, '中华人民共和国民法典', 65),
    (18708, '中华人民共和国公司法', 50),
    (18833, '中华人民共和国民事诉讼法', 37),
]

STATUTE_SQL = """
SELECT
    l.id       AS stable_id,
    l.title    AS doc_title,
    l.publish  AS publish,
    l.expiry   AS expiry,
    l.status   AS status,
    l.content  AS content
FROM laws l
WHERE l.id = %(law_id)s AND l.content_status = 'ok'
"""

AMENDMENT_SQL = """
SELECT
    l.id       AS stable_id,
    l.title    AS doc_title,
    l.publish  AS publish,
    l.content  AS content
FROM laws l
WHERE l.title ~ '^中华人民共和国刑法修正案'
  AND l.content_status = 'ok'
ORDER BY l.publish
"""

INSPECT_SQL = """
SELECT table_name, column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name IN ('laws', 'guide_cases', 'law_chunks_v2')
ORDER BY table_name, ordinal_position
"""

_ARTICLE_RE = re.compile(r'第[零一二三四五六七八九十百千]+条')


def split_articles(content: str) -> list[tuple[str, int, int, str]]:
    """laws.content -> [(条标, char_start, char_end, 条文文本)]，连续切片、去噪声短段。"""
    marks = [(m.group(0), m.start()) for m in _ARTICLE_RE.finditer(content)]
    out: list[tuple[str, int, int, str]] = []
    for i, (label, start) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else len(content)
        text = content[start:end].strip()
        if len(text) < 20:
            continue
        out.append((label, start, end, text))
    return out

_WRITE_KEYWORDS = re.compile(
    r'\b(insert\s+into|update\b|delete\s+from|delete\b|merge\s+into|merge\b|'
    r'create\s+(table|index|view|database|schema)|drop\b|alter\b|truncate\b|'
    r'grant\b|revoke\b|copy\b|set\b|vacuum\b|reindex\b|comment\s+on)\b',
    re.IGNORECASE,
)


class ReadOnlyViolation(RuntimeError):
    """试图向只读供应连接发送写语句。"""


def assert_readonly(sql: str) -> None:
    """静态断言 SQL 为只读。先剥掉字符串字面量与注释，避免误伤 SELECT ... 'insert'。"""
    stripped = re.sub(r"'(?:[^']|'')*'", "''", sql)
    stripped = re.sub(r'--[^\n]*', ' ', stripped)
    stripped = re.sub(r'/\*.*?\*/', ' ', stripped, flags=re.DOTALL)
    match = _WRITE_KEYWORDS.search(stripped)
    if match:
        raise ReadOnlyViolation(
            f'law_db supplier is read-only; write statement rejected: {match.group(0)!r}'
        )


@dataclass
class Episode:
    episode_id: str
    kind: str  # 'guide_case' | 'statute_article'
    content: str
    provenance: dict[str, Any] = field(default_factory=dict)
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    group_id: str = 'pilot'

    def __post_init__(self) -> None:
        if self.kind not in ('guide_case', 'statute_article', 'statute_amendment'):
            raise ValueError(
                f'episode kind must be guide_case|statute_article|statute_amendment, got {self.kind!r}'
            )
        if self.provenance.get('source_db') != 'law_db':
            self.provenance['source_db'] = 'law_db'


def build_connect_kwargs(dsn: str) -> dict[str, Any]:
    """psycopg AsyncConnection.connect 的关键字参数（含只读 options，可断言）。"""
    return {
        'conninfo': dsn,
        'options': PSYCOPG_OPTIONS,
        'autocommit': True,
    }


async def connect(settings: Any) -> 'ReadOnlyConnection':
    import psycopg

    dsn = settings.require_law_database_url()
    kwargs = build_connect_kwargs(dsn)
    raw = await psycopg.AsyncConnection.connect(**kwargs)
    return ReadOnlyConnection(raw)


class ReadOnlyConnection:
    """psycopg AsyncConnection 的只读守卫包装。所有 execute 先过 assert_readonly。"""

    def __init__(self, inner: Any):
        self._inner = inner

    async def fetch_episodes(
        self, kinds: tuple[str, ...] = ('guide_case', 'statute_article', 'statute_amendment'),
        limit: int = 100000,
    ) -> list[Episode]:
        """试点全集供给：guide_cases 全量 + 四部基本法条级 + 刑法修正案全文。

        limit 仅作用于 guide_cases（试点 293 全量，远小于默认值）。
        """
        episodes: list[Episode] = []
        async with self._inner.cursor() as cur:
            if 'guide_case' in kinds:
                assert_readonly(GUIDE_CASE_SQL)
                await cur.execute(GUIDE_CASE_SQL, {'limit': limit})
                for row in await cur.fetchall():
                    episodes.append(guide_case_row_to_episode(row))
            if 'statute_article' in kinds:
                for law_id, _title, cap in PILOT_STATUTES:
                    assert_readonly(STATUTE_SQL)
                    await cur.execute(STATUTE_SQL, {'law_id': law_id})
                    row = await cur.fetchone()
                    if row is None:
                        continue
                    episodes.extend(statute_row_to_episodes(row, cap))
            if 'statute_amendment' in kinds:
                assert_readonly(AMENDMENT_SQL)
                await cur.execute(AMENDMENT_SQL)
                for row in await cur.fetchall():
                    episodes.append(amendment_row_to_episode(row))
        return episodes

    async def inspect_schema(self) -> list[tuple]:
        assert_readonly(INSPECT_SQL)
        async with self._inner.cursor() as cur:
            await cur.execute(INSPECT_SQL)
            return list(await cur.fetchall())

    async def close(self) -> None:
        await self._inner.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _as_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def guide_case_row_to_episode(row: Any) -> Episode:
    """guide_cases 行 -> Episode。行形状见 GUIDE_CASE_SQL（已 live 核对）。

    正文按【标题/裁判要点/基本案情/裁判结果/裁判理由/相关法条/正文】分节拼装——
    分节标题兼作抽取上下文提示（实测抽取质量受益）。
    """
    (stable_id, _case_no, doc_title, publish_at, listed_at,
     gist, facts, result, reasoning, statutes, body) = row
    parts = [f'【标题】{doc_title}']
    if gist:
        parts.append(f'【裁判要点】{gist}')
    if facts:
        parts.append(f'【基本案情】{facts}')
    if result:
        parts.append(f'【裁判结果】{result}')
    if reasoning:
        parts.append(f'【裁判理由】{reasoning}')
    if statutes:
        parts.append(f'【相关法条】{statutes}')
    parts.append(f'【正文】{body}')
    as_of = _as_utc(publish_at) or _as_utc(listed_at)
    return Episode(
        episode_id=f'gc:{stable_id}',
        kind='guide_case',
        content='\n'.join(parts),
        provenance={
            'source_db': 'law_db',
            'stable_id': f'guide_cases:{stable_id}',
            'as_of': as_of or datetime.now(timezone.utc),
            'doc_title': doc_title,
        },
        valid_at=as_of,
    )


def statute_row_to_episodes(row: Any, cap: int) -> list[Episode]:
    """laws 行（单部法规全文）-> 条级 Episode 列表（带 char span 与版本口径）。"""
    stable_id, doc_title, publish, expiry, status, content = row
    publish_dt = _as_utc(publish)
    episodes: list[Episode] = []
    for label, start, end, text in split_articles(content or '')[:cap]:
        episodes.append(Episode(
            episode_id=f'law:{stable_id}:{label}',
            kind='statute_article',
            content=f'《{doc_title}》（发布 {publish}）{text}',
            provenance={
                'source_db': 'law_db',
                'stable_id': f'laws:{stable_id}',
                'as_of': publish_dt or datetime.now(timezone.utc),
                'doc_title': doc_title,
                'span': {'char_start': start, 'char_end': end, 'unit': 'laws.content'},
            },
            valid_at=publish_dt,
        ))
    return episodes


def amendment_row_to_episode(row: Any) -> Episode:
    """刑法修正案行 -> 整文 Episode（供失效/演化语义验证）。"""
    stable_id, doc_title, publish, content = row
    publish_dt = _as_utc(publish)
    return Episode(
        episode_id=f'law:{stable_id}:full',
        kind='statute_amendment',
        content=f'《{doc_title}》（发布 {publish}）\n{content}',
        provenance={
            'source_db': 'law_db',
            'stable_id': f'laws:{stable_id}',
            'as_of': publish_dt or datetime.now(timezone.utc),
            'doc_title': doc_title,
        },
        valid_at=publish_dt,
    )


def load_episodes_jsonl(path: str) -> list[Episode]:
    """试点语料快照 -> Episode 列表（与 fetch_episodes 同构；跑批用快照保证可复现）。"""
    episodes: list[Episode] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prov = d.get('provenance') or {}
            as_of = _as_utc(prov.get('as_of'))
            episodes.append(Episode(
                episode_id=d['episode_id'],
                kind=d['kind'],
                content=d['content'],
                provenance={
                    'source_db': 'law_db',
                    'stable_id': prov.get('stable_id', ''),
                    'as_of': as_of or datetime.now(timezone.utc),
                    'doc_title': prov.get('doc_title', ''),
                    'span': prov.get('span'),
                },
                valid_at=_as_utc(d.get('valid_at')),
            ))
    return episodes


async def _inspect_cli() -> int:
    from . import config

    settings = config.from_env()
    settings.apply_runtime_env()
    conn = await connect(settings)
    try:
        rows = await conn.inspect_schema()
    finally:
        await conn.close()
    if not rows:
        print('no columns found for laws/guide_cases/law_articles in public schema')
        return 1
    current_table = None
    for table_name, column_name, data_type, is_nullable in rows:
        if table_name != current_table:
            print(f'\n== {table_name} ==')
            current_table = table_name
        print(f'  {column_name:<32} {data_type:<24} nullable={is_nullable}')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='fd_legal_kg.supplier')
    parser.add_argument('--inspect', action='store_true',
                        help='连接真实 law_db 并打印 laws/guide_cases/law_articles 的列（核对 SQL 常量块用）')
    args = parser.parse_args(argv)
    if not args.inspect:
        parser.print_help()
        return 2
    return asyncio.run(_inspect_cli())


if __name__ == '__main__':
    raise SystemExit(main())


def corpus_index_for(episodes: list[Episode]) -> dict[str, dict[str, Any]]:
    """episode_id -> 可回指原文的溯源三元组（source_db / stable_id / as_of / span）。

    查询面用它把图里的事实边还原成「出自哪份材料的哪一段」：
    边只记 episode uuid，uuid 经图内 Episodic.name 映射回 episode_id，再到这里取溯源。
    """
    index: dict[str, dict[str, Any]] = {}
    for ep in episodes:
        prov = ep.provenance or {}
        as_of = prov.get('as_of')
        index[ep.episode_id] = {
            'episode_id': ep.episode_id,
            'kind': ep.kind,
            'source_db': prov.get('source_db', 'law_db'),
            'stable_id': prov.get('stable_id', ''),
            'doc_title': prov.get('doc_title', ''),
            'as_of': as_of.isoformat() if hasattr(as_of, 'isoformat') else str(as_of or ''),
            'span': prov.get('span'),
        }
    return index
