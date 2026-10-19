"""试点跑批器：512 episode -> FalkorDB 图（断点续跑，逐条容错，账本自动记录）。

用法（zihan）:
    set -a; . /home/ubuntu/lexkg-data/pilot.env; set +a
    python scripts/run_pilot.py \
        --episodes /home/ubuntu/lexkg-data/pilot_episodes.jsonl \
        --state    /home/ubuntu/lexkg-data/run_state.json \
        [--limit N] [--only-kind guide_case] [--dry-list]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fd_legal_kg import config, pipeline, supplier  # noqa: E402
from fd_legal_kg.cost import CostLedger  # noqa: E402


def load_state(path: str) -> dict:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding='utf-8'))
    return {'done': [], 'failed': []}


def save_state(path: str, state: dict) -> None:
    Path(path).write_text(
        json.dumps(state, ensure_ascii=False, indent=1), encoding='utf-8'
    )


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', required=True)
    ap.add_argument('--state', required=True)
    ap.add_argument('--limit', type=int, default=0, help='0=全部')
    ap.add_argument('--only-kind', default='')
    ap.add_argument('--dry-list', action='store_true', help='只列出待跑清单不入图')
    args = ap.parse_args()

    settings = config.from_env()
    settings.apply_runtime_env()
    episodes = supplier.load_episodes_jsonl(args.episodes)
    if args.only_kind:
        episodes = [e for e in episodes if e.kind == args.only_kind]
    if args.limit:
        episodes = episodes[: args.limit]

    state = load_state(args.state)
    done = set(state['done'])
    pending = [e for e in episodes if e.episode_id not in done]
    print(f'episodes total={len(episodes)} done={len(done)} pending={len(pending)}')

    if args.dry_list:
        for e in pending[:20]:
            print(' ', e.episode_id, e.kind, len(e.content))
        return 0

    ledger = CostLedger(settings.cost_ledger_path)
    llm, embedder, cross = pipeline.build_cost_wired_clients(settings, ledger)
    graphiti = pipeline.build_graphiti(settings, llm_client=llm, embedder=embedder, cross_encoder=cross)

    import time

    t0 = time.time()
    for i, ep in enumerate(pending, 1):
        try:
            await pipeline.ingest_episode(
                graphiti, ep, ledger, structured_mode=settings.structured_mode
            )
            state['done'].append(ep.episode_id)
            save_state(args.state, state)
            el = time.time() - t0
            print(
                f'[{i}/{len(pending)}] OK {ep.episode_id} ({ep.kind}, {len(ep.content)} ch)'
                f' {el / i:.1f}s/ep avg',
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - 逐条容错，失败记录后继续
            state['failed'].append({'id': ep.episode_id, 'error': str(exc)[:300]})
            save_state(args.state, state)
            print(f'[{i}/{len(pending)}] FAIL {ep.episode_id}: {exc!r}', flush=True)

    await graphiti.close()
    print(
        f"batch finished: done={len(state['done'])} failed={len(state['failed'])} "
        f"wall={time.time() - t0:.0f}s"
    )
    return 0 if not state['failed'] else 1


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
