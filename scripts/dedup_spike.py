"""spike 1.3：高频实体去重量化（对已完成批次的图直接统计，不触发新 LLM 调用）。

要回答两个反向问题：
  ① 实体爆炸：同名实体是否被拆成多个节点（name 重复）？
  ② 过度合并：不同实体是否被错误并成一个节点（同 name 但属不同类型/不同辖区）？

判据（可复核）：
  - duplicate_rate = 有重名的 name 数 / 总 name 数  → 爆炸指标
  - cross_type_collisions = 同一 name 节点集合里 labels 类型多于 1 个的组数 → 过并信号
  - 高频实体（关联 episode 数 ≥ 10）逐条列出，人工可读性留档
"""
from __future__ import annotations

import argparse
import json
import collections
from typing import Any

try:  # 轻依赖：直连 FalkorDB 的 redis 协议，不引入 graphiti
    import redis  # type: ignore
except ImportError:  # pragma: no cover
    redis = None


# FalkorDB 把 labels(n) 与 collect(...) 都序列化成字符串（'[Entity, Case]' /
# '[[ep1],[ep2]]'），不是原生列表——这里统一按字符串解析，避免把标签当字符迭代。
ENTITY_QUERY = """
MATCH (n:Entity)
OPTIONAL MATCH (n)-[r:RELATES_TO]-()
WITH n, count(DISTINCT r.uuid) AS edges,
     collect(DISTINCT r.episodes) AS buckets
RETURN n.name AS name,
       labels(n) AS labels,
       edges AS edges,
       buckets AS buckets
"""


def _connect(host: str, port: int):
    if redis is None:
        raise SystemExit('pip install redis  (或直接用 redis-cli 导出 JSON)')
    return redis.Redis(host=host, port=port, decode_responses=True)


def _graph_query(r, graph: str, query: str) -> list[list[Any]]:
    """GRAPH.QUERY 的 RESP 解析：返回 rows 部分（跳过表头与统计尾巴）。"""
    raw = r.execute_command('GRAPH.QUERY', graph, query)
    # 形状: [[header], [rows...], [stats...]]
    if not raw or len(raw) < 2:
        return []
    return raw[1]


def _as_list(value: Any) -> list[Any]:
    """把 FalkorDB 的 '[a, b]' 字符串或原生 list 统一成 list。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if text.startswith('[') and text.endswith(']'):
        text = text[1:-1]
    if not text:
        return []
    return [part.strip().strip("'\"") for part in text.split(',') if part.strip()]


def _episode_count(buckets: Any) -> int:
    """collect(DISTINCT r.episodes) -> 每个元素是该边的 episode uuid 数组（字符串形态）。

    真实口径是「该实体关联的不同 episode 数」，故把每个内层数组再拆开取并集。
    """
    total: set[str] = set()
    for bucket in _as_list(buckets):
        for ep in _as_list(bucket):
            if ep and ep.lower() not in ('null', 'none'):
                total.add(ep)
    return len(total)


def collect(r, graph: str) -> list[dict[str, Any]]:
    rows = _graph_query(r, graph, ENTITY_QUERY)
    out = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            continue
        name, labels, edges, buckets = row[0], row[1], row[2], row[3]
        types = [t for t in _as_list(labels) if t and t != 'Entity']
        out.append({
            'name': name,
            'types': types,
            'edges': int(edges or 0),
            'episode_count': _episode_count(buckets),
        })
    return out


def analyze(entities: list[dict[str, Any]], high_freq_threshold: int = 10) -> dict[str, Any]:
    by_name: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for e in entities:
        by_name[e['name']].append(e)

    duplicated = {name: len(v) for name, v in by_name.items() if len(v) > 1}
    cross_type = {
        name: sorted({t for item in v for t in item['types']})
        for name, v in by_name.items()
        if len({t for item in v for t in item['types']}) > 1
    }
    # 过并的第二个信号：同名节点带着不止一种类型（一个名字被当成法院又当成案例）。
    multi_typed = {
        e['name']: e['types'] for e in entities if len(e['types']) > 1
    }
    high_freq = sorted(
        (e for e in entities if e['episode_count'] >= high_freq_threshold),
        key=lambda e: (-e['episode_count'], -e['edges']),
    )
    type_counts: collections.Counter[str] = collections.Counter()
    for e in entities:
        for t in e['types']:
            type_counts[t] += 1

    total_names = len(by_name)
    total_nodes = len(entities)
    return {
        'total_nodes': total_nodes,
        'total_distinct_names': total_names,
        'duplicate_names': len(duplicated),
        'duplicate_rate': (len(duplicated) / total_names) if total_names else 0.0,
        'duplicate_examples': dict(sorted(duplicated.items(), key=lambda kv: -kv[1])[:20]),
        'cross_type_collisions': len(cross_type),
        'cross_type_examples': dict(list(cross_type.items())[:20]),
        'multi_typed_nodes': len(multi_typed),
        'multi_typed_examples': dict(list(multi_typed.items())[:20]),
        'type_counts': dict(type_counts.most_common()),
        'high_freq_threshold': high_freq_threshold,
        'high_freq_entities': high_freq[:40],
        'entity_explosion_ratio': total_nodes / total_names if total_names else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=6889)
    ap.add_argument('--graph', default='pilot')
    ap.add_argument('--threshold', type=int, default=10)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    r = _connect(args.host, args.port)
    entities = collect(r, args.graph)
    report = analyze(entities, high_freq_threshold=args.threshold)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
