"""法规名归并审计报告：读图统计「条号分裂」造成的聚合损失，并给出归并后增益。

只读，不改图。用法（zihan）:
    /home/ubuntu/fd-legal-kg-venv/bin/python scripts/statute_merge_report.py [--graph pilot]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import redis  # noqa: E402

from fd_legal_kg.statute_name import group_by_statute, parse_statute_name  # noqa: E402

ARTICLE_HINT = re.compile(r'第[一二三四五六七八九十百千零〇\d]+条')


def _rows(r, cypher: str):
    raw = r.execute_command('GRAPH.QUERY', 'pilot', cypher)
    return raw[1] if len(raw) > 1 else []


def _esc(s: str) -> str:
    return s.replace('\\', '\\\\').replace("'", "\\'")


def citing_entities(r, names: list[str]) -> set[str]:
    out: set[str] = set()
    for nm in names:
        for (u,) in _rows(r, f"MATCH (a:Entity)-[:RELATES_TO]->(b:Entity:Statute "
                             f"{{name:'{_esc(nm)}'}}) RETURN DISTINCT a.uuid AS u"):
            out.add(u)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=6889)
    ap.add_argument('--graph', default='pilot')
    ap.add_argument('--top', type=int, default=10)
    a = ap.parse_args()

    r = redis.Redis(host=a.host, port=a.port, decode_responses=True)
    names = [row[0] for row in _rows(r, 'MATCH (n:Entity:Statute) RETURN n.name AS name')]
    groups = group_by_statute(names)

    split = [n for n in names if ARTICLE_HINT.search(n)]
    interp = [n for n in names if parse_statute_name(n).kind == 'interpretation']
    print(f'Statute 节点 {len(names)}')
    print(f'  带条号（可归并）  {len(split)}  ({len(split) / max(len(names), 1):.1%})')
    print(f'  立法解释（不拆）  {len(interp)}')
    print(f'  归并后法规主体    {len(groups)}   → 消除孤立节点 {len(names) - len(groups)}')

    print(f'\n=== 聚合增益 Top {a.top}（按引用实体数） ===')
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1].members))[: a.top]
    print(f'{"法规主体":44s} {"节点数":>6s} {"归并前":>7s} {"归并后":>7s} {"增益":>7s}')
    for key, grp in ranked:
        member_names = [m.raw for m in grp.members]
        after = len(citing_entities(r, member_names))
        bare = [n for n in member_names if not ARTICLE_HINT.search(n)]
        before = len(citing_entities(r, bare)) if bare else 0
        delta = after - before
        pct = f'+{delta / before:.1%}' if before else 'n/a'
        print(f'{key[:42]:44s} {len(member_names):6d} {before:7d} {after:7d} {delta:+7d} {pct}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
