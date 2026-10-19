"""5.3 价值实证探针：对每道金标题题，把 KG 车道的原始检索证据落档。

不依赖 evalrunner 的 CLI（那条路径要同时连基线与图库），此脚本只做 KG 侧，
输出与对照表同构的 JSONL，便于与基线 run3 逐题并排。

用法（zihan）:
    set -a; . /home/ubuntu/lexkg-data/pilot.env; set +a
    python scripts/eval_kg_probe.py \
        --questions data/eval_questions_draft.json \
        --episodes /home/ubuntu/lexkg-data/pilot_episodes.jsonl \
        --out /home/ubuntu/lexkg-data/eval_kg.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fd_legal_kg import config  # noqa: E402
from fd_legal_kg.pipeline import build_graphiti  # noqa: E402
from fd_legal_kg.query import GraphitiQueryBackend, LegalKGQuery  # noqa: E402
from fd_legal_kg.supplier import corpus_index_for, load_episodes_jsonl  # noqa: E402


async def run(questions_path: str, episodes_path: str, out: str, limit: int) -> int:
    settings = config.from_env()
    settings.apply_runtime_env()
    questions = json.loads(Path(questions_path).read_text(encoding='utf-8'))['questions']
    if limit:
        questions = questions[:limit]

    graphiti = build_graphiti(settings)
    corpus = corpus_index_for(load_episodes_jsonl(episodes_path))
    kg = LegalKGQuery(GraphitiQueryBackend(graphiti), corpus_index=corpus)

    rows = []
    for q in questions:
        qid, kind, text = q['id'], q['kind'], q['query']
        evidence: list[dict] = []
        try:
            facts = await kg.search_facts(text, limit=8)
            evidence.extend(f.to_dict() for f in facts)
            # 多跳题：从首条事实的主体做二跳游走，补图链
            if kind == 'multihop' and facts:
                for start in {f.subject for f in facts[:2]}:
                    for v in await kg.walk(start, max_hops=2):
                        d = v.to_dict()
                        if d['fact'] not in {e.get('fact') for e in evidence}:
                            evidence.append(d)
        except Exception as exc:  # noqa: BLE001 - 单题故障不中断
            evidence.append({'error': f'{type(exc).__name__}: {exc}'})
        rows.append({'id': qid, 'kind': kind, 'query': text,
                     'expectation': q.get('expectation', ''),
                     'kg_evidence': evidence})
        print(f'{qid}: {len(evidence)} evidence facts', flush=True)

    Path(out).write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n',
        encoding='utf-8',
    )
    print(f'wrote {out} ({len(rows)} questions)')
    await graphiti.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--questions', required=True)
    ap.add_argument('--episodes', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    return asyncio.run(run(a.questions, a.episodes, a.out, a.limit))


if __name__ == '__main__':
    raise SystemExit(main())
