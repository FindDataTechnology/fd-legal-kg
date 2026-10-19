"""query：纯逻辑部分 + fake 图后端（不需要 FalkorDB）。"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from fd_legal_kg.query import (
    EdgeRecord,
    LegalKGQuery,
    NodeRecord,
    fact_valid_at,
)

T0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2021, 6, 1, tzinfo=timezone.utc)
T2 = datetime(2023, 1, 1, tzinfo=timezone.utc)
T3 = datetime(2025, 1, 1, tzinfo=timezone.utc)


class FakeBackend:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    async def get_nodes_by_name(self, name, group_ids):
        return [n for n in self.nodes if n.name == name and n.group_id in group_ids]

    async def get_nodes_by_uuids(self, uuids):
        return [n for n in self.nodes if n.uuid in uuids]

    async def get_edges_touching_nodes(self, uuids):
        return [e for e in self.edges
                if e.source_node_uuid in uuids or e.target_node_uuid in uuids]

    async def search_edges(self, query, group_ids, limit):
        return [e for e in self.edges if query.lower() in e.fact.lower()][:limit]


def _graph():
    nodes = [
        NodeRecord(uuid='A', name='刑法', group_id='pilot'),
        NodeRecord(uuid='B', name='故意伤害罪', group_id='pilot'),
        NodeRecord(uuid='C', name='指导案例24号', group_id='pilot'),
        NodeRecord(uuid='D', name='远端节点', group_id='pilot'),
    ]
    edges = [
        EdgeRecord(uuid='e1', fact='故意伤害罪规定于刑法', name='规定于',
                   source_node_uuid='B', target_node_uuid='A',
                   episodes=['statute_article:law-1:第二条'], valid_at=T0, invalid_at=None),
        EdgeRecord(uuid='e2', fact='指导案例24号适用故意伤害罪', name='适用',
                   source_node_uuid='C', target_node_uuid='B',
                   episodes=['guide_case:gc-24'], valid_at=T1, invalid_at=T2),
        EdgeRecord(uuid='e3', fact='远端节点连接指导案例24号', name='关联',
                   source_node_uuid='D', target_node_uuid='C',
                   episodes=['guide_case:gc-x'], valid_at=T0, invalid_at=None),
    ]
    return FakeBackend(nodes, edges)


def _provenance_index():
    return {
        'statute_article:law-1:第二条': {
            'source_db': 'law_db', 'stable_id': 'law-1#第二条', 'doc_title': '刑法',
            'span': {'char_start': 0, 'char_end': 20},
        },
        'guide_case:gc-24': {
            'source_db': 'law_db', 'stable_id': 'gc-24', 'doc_title': '指导案例24号',
        },
    }


def test_fact_valid_at_boundaries():
    e = EdgeRecord(uuid='x', fact='f', name='r', source_node_uuid='s', target_node_uuid='t',
                   valid_at=T1, invalid_at=T2)
    assert fact_valid_at(e, T1) is True       # valid_at <= t
    assert fact_valid_at(e, T2) is False      # t >= invalid_at
    assert fact_valid_at(e, T0) is False      # 早于 valid_at
    open_ended = EdgeRecord(uuid='y', fact='f', name='r', source_node_uuid='s',
                            target_node_uuid='t', valid_at=None, invalid_at=None)
    assert fact_valid_at(open_ended, T3) is True


async def test_search_entities_by_name():
    q = LegalKGQuery(_graph(), provenance_index=_provenance_index())
    hits = await q.search_entities('刑法')
    assert [n.uuid for n in hits] == ['A']
    assert (await q.search_entities('不存在')) == []


async def test_search_facts_includes_provenance_and_validity():
    q = LegalKGQuery(_graph(), provenance_index=_provenance_index())
    facts = await q.search_facts('故意伤害罪')
    assert len(facts) == 2
    by_uuid = {f.fact: f for f in facts}
    f2 = by_uuid['指导案例24号适用故意伤害罪']
    assert f2.subject == '指导案例24号' and f2.object == '故意伤害罪'
    assert f2.validity_window() == {'valid_at': T1, 'invalid_at': T2, 'expired_at': None}
    assert f2.provenance == [{'source_db': 'law_db', 'stable_id': 'gc-24',
                              'doc_title': '指导案例24号'}]
    d = f2.to_dict()
    assert d['valid_at'].startswith('2021-06-01') and d['invalid_at'].startswith('2023-01-01')


async def test_search_facts_as_of_filter():
    q = LegalKGQuery(_graph(), provenance_index=_provenance_index())
    # 2021 年时 e2（2021-06 生效）尚未生效
    facts_2021 = await q.search_facts('故意伤害罪', as_of=datetime(2021, 1, 1, tzinfo=timezone.utc))
    assert [f.fact for f in facts_2021] == ['故意伤害罪规定于刑法']
    # 2022 年：两条都为真
    facts_2022 = await q.search_facts('故意伤害罪', as_of=datetime(2022, 1, 1, tzinfo=timezone.utc))
    assert len(facts_2022) == 2
    # 2024 年：e2 已失效（superseded → invalid_at，不是删除）
    facts_2024 = await q.search_facts('故意伤害罪', as_of=datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert [f.fact for f in facts_2024] == ['故意伤害罪规定于刑法']


async def test_walk_respects_hop_limit():
    q = LegalKGQuery(_graph(), provenance_index=_provenance_index())
    # A --e1-- B --e2-- C --e3-- D：从 A 出发 2 跳只能到 e1、e2
    facts = await q.walk('刑法', max_hops=2)
    assert {f.fact for f in facts} == {'故意伤害罪规定于刑法', '指导案例24号适用故意伤害罪'}
    facts3 = await q.walk('刑法', max_hops=3)
    assert len(facts3) == 3  # 第三跳补上 e3


async def test_walk_as_of_and_unknown_start():
    q = LegalKGQuery(_graph(), provenance_index=_provenance_index())
    # 从 C 出发 2 跳可达 e2、e3（第 1 跳）与 e1（第 2 跳），2022 年三条都为真
    facts = await q.walk('指导案例24号', max_hops=2, as_of=datetime(2022, 1, 1, tzinfo=timezone.utc))
    assert len(facts) == 3
    # as_of 落在 2024：e2 已失效（superseded → invalid_at，不是删除），剩 e1、e3
    facts_2024 = await q.walk('指导案例24号', max_hops=2, as_of=datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert {f.fact for f in facts_2024} == {'故意伤害罪规定于刑法', '远端节点连接指导案例24号'}
    assert await q.walk('查无此人') == []
    with pytest.raises(ValueError):
        await q.walk('刑法', max_hops=0)


async def test_walk_includes_provenance_for_all_edges():
    q = LegalKGQuery(_graph(), provenance_index=_provenance_index())
    facts = await q.walk('刑法', max_hops=2)
    statute_fact = next(f for f in facts if f.fact == '故意伤害罪规定于刑法')
    assert statute_fact.provenance[0]['span'] == {'char_start': 0, 'char_end': 20}
    assert statute_fact.episode_ids == ['statute_article:law-1:第二条']


async def test_graphiti_backend_walk_maps_temporal_fields():
    """回归：walk 走 GraphitiQueryBackend 时时效字段必须映射（曾静默丢弃致 as_of 放行一切）。"""
    from fd_legal_kg.query import EdgeRecord, GraphitiQueryBackend

    class _Driver:
        async def execute_query(self, cypher, **kw):
            return (
                [dict(
                    uuid='e-1', fact='旧条文规定X', name='APPLIES_LAW',
                    source_node_uuid='a', target_node_uuid='b', episodes=['ep1'],
                    valid_at='2012-01-11T00:00:00+00:00',
                    invalid_at='2021-01-01T00:00:00+00:00',
                    expired_at=None,
                    reference_time='2012-01-11T00:00:00+00:00',
                )],
                None,
                None,
            )

    class _G:
        driver = _Driver()

    backend = GraphitiQueryBackend(_G())
    edges = await backend.get_edges_touching_nodes(['a'])
    assert isinstance(edges[0], EdgeRecord)
    assert edges[0].valid_at is not None and edges[0].valid_at.year == 2012
    assert edges[0].invalid_at is not None and edges[0].invalid_at.year == 2021
    assert edges[0].reference_time is not None and edges[0].reference_time.year == 2012


async def test_graphiti_backend_switches_to_the_group_graph():
    """FalkorDB 下 graph_name 就是 group_id：直连 graphiti.driver 会一直落在
    default_db（空图），查询面全部返回空。后端必须显式 clone(database=group_id)。"""
    from fd_legal_kg.query import GraphitiQueryBackend

    clones: list[str] = []

    class FakeDriver:
        _database = 'default_db'

        def clone(self, database: str):
            clones.append(database)
            return FakeDriver()

    graphiti = SimpleNamespace(driver=FakeDriver())
    backend = GraphitiQueryBackend(graphiti)
    backend._driver(['pilot'])
    assert clones == ['pilot']
    # 已在目标图时不再 clone
    clone_once = FakeDriver()
    clone_once._database = 'pilot'
    clone_once.clone = lambda database: (_ for _ in ()).throw(AssertionError('no clone'))
    backend2 = GraphitiQueryBackend(SimpleNamespace(driver=clone_once))
    assert backend2._driver(['pilot']) is clone_once


async def test_provenance_resolved_lazily_through_episode_uuid():
    """事实边只带 episode uuid；查询面必须能把它还原成语料 id 并回填溯源。

    否则 spec 的「每条 fact 可回指 episode 与原文」只有 episode_ids 一串 uuid，
    人工没法据此找到原文。
    """
    from fd_legal_kg.query import GraphitiQueryBackend

    corpus = {
        'gc:4214': {'episode_id': 'gc:4214', 'source_db': 'law_db',
                    'stable_id': '4214', 'as_of': '2012-01-11', 'span': None},
    }

    class Backend:
        async def get_nodes_by_name(self, name, group_ids):
            return [NodeRecord(uuid='n1', name=name)]

        async def get_nodes_by_uuids(self, uuids):
            return [NodeRecord(uuid=u, name=f'name-{u}') for u in uuids]

        async def get_edges_touching_nodes(self, uuids):
            return [EdgeRecord(uuid='e1', fact='某事实', name='RELATES_TO',
                               source_node_uuid='n1', target_node_uuid='n2',
                               episodes=['ep-uuid-1'])]

        async def search_edges(self, query, group_ids, limit):
            return await self.get_edges_touching_nodes([])

        async def get_episode_names(self, uuids):
            return {'ep-uuid-1': 'gc:4214'} if 'ep-uuid-1' in uuids else {}

    q = LegalKGQuery(Backend(), corpus_index=corpus)
    views = await q.walk('某实体', max_hops=1)
    assert views[0].episode_ids == ['ep-uuid-1']
    assert views[0].provenance == [corpus['gc:4214']]
    assert views[0].to_dict()['provenance'][0]['stable_id'] == '4214'
