"""试点查询面：实体/关键词检索、多跳图游走、as-of 时间过滤。

纯逻辑（时序过滤、路径组装、provenance 注入）与图后端解耦：
  QueryBackend —— 协议（live 实现是 GraphitiQueryBackend，测试注入 fake）；
  LegalKGQuery —— 业务面，每条返回的事实都带 provenance + 有效性窗口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol


@dataclass
class NodeRecord:
    uuid: str
    name: str
    group_id: str | None = None
    labels: list[str] = field(default_factory=list)
    summary: str | None = None


@dataclass
class EdgeRecord:
    uuid: str
    fact: str
    name: str
    source_node_uuid: str
    target_node_uuid: str
    episodes: list[str] = field(default_factory=list)
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    expired_at: datetime | None = None
    reference_time: datetime | None = None


class QueryBackend(Protocol):
    async def get_nodes_by_name(self, name: str, group_ids: list[str]) -> list[NodeRecord]: ...

    async def get_nodes_by_uuids(self, uuids: list[str]) -> list[NodeRecord]: ...

    async def get_edges_touching_nodes(self, uuids: list[str]) -> list[EdgeRecord]: ...

    async def search_edges(self, query: str, group_ids: list[str], limit: int) -> list[EdgeRecord]: ...

    async def get_episode_names(self, uuids: list[str]) -> dict[str, str]: ...


def _parse_dt(value: Any) -> datetime | None:
    """FalkorDB 时间字段是 ISO 字符串（driver 写入前已序列化），转回 datetime。"""
    if value is None or isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return None


def fact_valid_at(edge: EdgeRecord, at: datetime) -> bool:
    """双时间判断：事实在时刻 at 为真（valid_at <= at < invalid_at；None 表示开放端）。"""
    if edge.valid_at is not None and at < edge.valid_at:
        return False
    if edge.invalid_at is not None and at >= edge.invalid_at:
        return False
    return True


@dataclass
class FactView:
    fact: str
    relation: str
    subject: str
    object: str
    valid_at: datetime | None
    invalid_at: datetime | None
    expired_at: datetime | None
    episode_ids: list[str]
    provenance: list[dict[str, Any]] = field(default_factory=list)

    def validity_window(self) -> dict[str, datetime | None]:
        return {'valid_at': self.valid_at, 'invalid_at': self.invalid_at,
                'expired_at': self.expired_at}

    def to_dict(self) -> dict[str, Any]:
        return {
            'fact': self.fact,
            'relation': self.relation,
            'subject': self.subject,
            'object': self.object,
            'valid_at': self.valid_at.isoformat() if self.valid_at else None,
            'invalid_at': self.invalid_at.isoformat() if self.invalid_at else None,
            'expired_at': self.expired_at.isoformat() if self.expired_at else None,
            'episode_ids': self.episode_ids,
            'provenance': self.provenance,
        }


class LegalKGQuery:
    def __init__(self, backend: QueryBackend, group_id: str = 'pilot',
                 provenance_index: dict[str, dict[str, Any]] | None = None,
                 corpus_index: dict[str, dict[str, Any]] | None = None):
        self.backend = backend
        self.group_ids = [group_id]
        # provenance 索引以 episode **uuid** 为键（edges.episodes 存的就是 uuid），
        # 由 _ensure_provenance 从 corpus_index（episode_id -> 溯源三元组）懒解析。
        self.provenance_index = provenance_index or {}
        self.corpus_index = corpus_index or {}

    def _provenance_for(self, edge: EdgeRecord) -> list[dict[str, Any]]:
        return [self.provenance_index[e] for e in edge.episodes if e in self.provenance_index]

    async def _ensure_provenance(self, edges: list[EdgeRecord]) -> None:
        """按需把「图内 episode uuid」解析成语料 episode id 并回填 provenance_index。

        懒解析而非构造期全量拉取：一次查询只碰它真正用到的那几条 episode。
        解析不到的 uuid 静默跳过（半写入的残留 episode），不伪造溯源。
        """
        missing = {u for e in edges for u in e.episodes if u not in self.provenance_index}
        if not missing:
            return
        resolver = getattr(self.backend, 'get_episode_names', None)
        if resolver is None:
            return
        names = await resolver(sorted(missing))
        for uuid, name in names.items():
            entry = self.corpus_index.get(name)
            if entry is not None:
                self.provenance_index[uuid] = entry

    async def _edge_records_to_views(self, edges: list[EdgeRecord],
                                     as_of: datetime | None = None) -> list[FactView]:
        if as_of is not None:
            edges = [e for e in edges if fact_valid_at(e, as_of)]
        if not edges:
            return []
        await self._ensure_provenance(edges)
        nodes = await self.backend.get_nodes_by_uuids(
            [u for e in edges for u in (e.source_node_uuid, e.target_node_uuid)]
        )
        names = {n.uuid: n.name for n in nodes}
        return [
            FactView(
                fact=e.fact,
                relation=e.name,
                subject=names.get(e.source_node_uuid, e.source_node_uuid),
                object=names.get(e.target_node_uuid, e.target_node_uuid),
                valid_at=e.valid_at,
                invalid_at=e.invalid_at,
                expired_at=e.expired_at,
                episode_ids=list(e.episodes),
                provenance=self._provenance_for(e),
            )
            for e in edges
        ]

    async def search_entities(self, name: str) -> list[NodeRecord]:
        """按名称精确/前缀检索实体。"""
        return await self.backend.get_nodes_by_name(name, self.group_ids)

    async def search_facts(self, query: str, limit: int = 10,
                           as_of: datetime | None = None) -> list[FactView]:
        """关键词混合检索事实（每条带 provenance + 有效性窗口）。"""
        edges = await self.backend.search_edges(query, self.group_ids, limit)
        return await self._edge_records_to_views(edges, as_of)

    async def walk(self, start_name: str, max_hops: int = 2,
                   as_of: datetime | None = None) -> list[FactView]:
        """从起始实体做多跳游走（<= max_hops），收集沿途事实边，去重保序。"""
        if max_hops < 1:
            raise ValueError('max_hops must be >= 1')
        start_nodes = await self.backend.get_nodes_by_name(start_name, self.group_ids)
        if not start_nodes:
            return []
        visited: set[str] = set()
        frontier = {n.uuid for n in start_nodes}
        seen_edges: dict[str, EdgeRecord] = {}
        for _ in range(max_hops):
            if not frontier:
                break
            edges = await self.backend.get_edges_touching_nodes(sorted(frontier))
            visited |= frontier
            nxt: set[str] = set()
            for edge in edges:
                seen_edges.setdefault(edge.uuid, edge)
                nxt.add(edge.source_node_uuid)
                nxt.add(edge.target_node_uuid)
            frontier = nxt - visited
        return await self._edge_records_to_views(list(seen_edges.values()), as_of)


class GraphitiQueryBackend:
    """QueryBackend 的 graphiti live 实现（需要真实 FalkorDB + 网关）。

    **FalkorDB 图名 = group_id**：graphiti 用 ``@traceable`` 装饰器在方法调用前
    ``driver.clone(database=group_id)`` 切图，直连 ``graphiti.driver`` 会一直落在
    ``default_db``（空图）。故这里的每条图访问都必须显式按 group_id 取 driver。
    """

    def __init__(self, graphiti: Any):
        self.graphiti = graphiti

    def _driver_for(self, group_id: str) -> Any:
        """按 group_id 切到对应图的 driver（FalkorDB 的 graph_name 就是 group_id）。"""
        driver = self.graphiti.driver
        current = getattr(driver, '_database', None)
        if current == group_id or not hasattr(driver, 'clone'):
            return driver
        return driver.clone(database=group_id)

    def _driver(self, group_ids: list[str] | None = None) -> Any:
        return self._driver_for((group_ids or ['pilot'])[0])

    async def get_nodes_by_name(self, name: str, group_ids: list[str]) -> list[NodeRecord]:
        from graphiti_core.nodes import EntityNode

        nodes = await EntityNode.get_by_names(self._driver(group_ids), [name], group_ids)
        return [
            NodeRecord(uuid=n.uuid, name=n.name, group_id=n.group_id,
                       labels=list(n.labels or []), summary=n.summary)
            for n in nodes
        ]

    async def get_nodes_by_uuids(self, uuids: list[str]) -> list[NodeRecord]:
        from graphiti_core.nodes import EntityNode

        nodes = await EntityNode.get_by_uuids(self._driver(), uuids)
        return [
            NodeRecord(uuid=n.uuid, name=n.name, group_id=n.group_id,
                       labels=list(n.labels or []), summary=n.summary)
            for n in nodes
        ]

    async def get_episode_names(self, uuids: list[str]) -> dict[str, str]:
        """episode uuid -> episode name（对方是语料里的 gc:4214 / law:18726:第一条）。

        边的 episodes 字段存 uuid，而溯源要的是语料 id；这一步是「事实可回指」
        链路上不可省的一环。
        """
        if not uuids:
            return {}
        records, _, _ = await self._driver().execute_query(
            'MATCH (e:Episodic) WHERE e.uuid IN $uuids RETURN e.uuid AS uuid, e.name AS name',
            uuids=list(uuids),
        )
        return {rec['uuid']: rec['name'] for rec in records if rec.get('name')}

    async def get_edges_touching_nodes(self, uuids: list[str]) -> list[EdgeRecord]:
        records, _, _ = await self._driver().execute_query(
            """
            MATCH (n:Entity)-[r:RELATES_TO]-(m:Entity)
            WHERE n.uuid IN $uuids
            RETURN r.uuid AS uuid, r.fact AS fact, r.name AS name,
                   r.source_node_uuid AS source_node_uuid,
                   r.target_node_uuid AS target_node_uuid,
                   r.episodes AS episodes, r.valid_at AS valid_at,
                   r.invalid_at AS invalid_at, r.expired_at AS expired_at,
                   r.reference_time AS reference_time
            """,
            uuids=uuids,
        )
        out = []
        for rec in records:
            out.append(EdgeRecord(
                uuid=rec['uuid'],
                fact=rec['fact'],
                name=rec['name'],
                source_node_uuid=rec['source_node_uuid'],
                target_node_uuid=rec['target_node_uuid'],
                episodes=list(rec['episodes'] or []),
                valid_at=_parse_dt(rec.get('valid_at')),
                invalid_at=_parse_dt(rec.get('invalid_at')),
                expired_at=_parse_dt(rec.get('expired_at')),
                reference_time=_parse_dt(rec.get('reference_time')),
            ))
        return out

    async def search_edges(self, query: str, group_ids: list[str], limit: int) -> list[EdgeRecord]:
        # graphiti.search() 用 RRF 混合检索（不走 cross-encoder），返回 EntityEdge
        edges = await self.graphiti.search(query, group_ids=group_ids, num_results=limit)
        return [
            EdgeRecord(
                uuid=e.uuid,
                fact=e.fact,
                name=e.name,
                source_node_uuid=e.source_node_uuid,
                target_node_uuid=e.target_node_uuid,
                episodes=list(e.episodes or []),
                valid_at=e.valid_at,
                invalid_at=e.invalid_at,
                expired_at=e.expired_at,
                reference_time=e.reference_time,
            )
            for e in edges
        ]
