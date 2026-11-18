"""Graphiti 组装与 episode 摄取。

FalkorDriver 的构造函数会立刻连接 FalkorDB（内部做 Is_Cluster 探测），
因此 build_graphiti 只能在有真实图库的环境调用；离线测试不要碰它。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI

from . import config, gateway, prompts
from .cost import (
    CostLedger,
    CostRecordingLLMClient,
    LedgerEmbedder,
    RecordingAsyncOpenAI,
)
from .ontology import ENTITY_TYPES
from .supplier import Episode


def build_graphiti(
    settings: config.Settings,
    llm_client: Any = None,
    embedder: Any = None,
    cross_encoder: Any = None,
) -> Any:
    """组装 Graphiti：FalkorDB driver + 网关 LLM/embedder + 网关 reranker + 并发上限。"""
    from graphiti_core import Graphiti
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    host, port = settings.falkor_endpoint()
    try:
        driver = FalkorDriver(
            host=host,
            port=port,
            username=settings.falkor_user,
            password=settings.falkor_password,
        )
    except Exception as exc:  # FalkorDriver 构造即连库
        raise config.ConfigError(
            f'cannot reach FalkorDB at {settings.falkor_uri} (LEXKG_FALKORDB_URI): {exc}'
        ) from exc

    if llm_client is None:
        llm_client = gateway.build_llm_client(settings)
    if embedder is None:
        embedder = gateway.build_embedder(settings)
    if cross_encoder is None:
        # 不传的话 Graphiti 会急切构造 OpenAI 官方 reranker（要 OPENAI_API_KEY）
        cross_encoder = gateway.build_cross_encoder(settings)

    settings.apply_runtime_env()
    return Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
        max_coroutines=settings.semaphore_limit,
    )


def build_cost_wired_clients(
    settings: config.Settings, ledger: CostLedger
) -> tuple[CostRecordingLLMClient, LedgerEmbedder, Any]:
    """构造三层包装的生产客户端：记账 AsyncOpenAI -> 网关 LLM -> 记账 LLM。"""
    raw_openai = AsyncOpenAI(
        api_key=settings.llm_api_key, base_url=settings.llm_base_url
    )
    recording_openai = RecordingAsyncOpenAI(raw_openai, ledger)
    llm = CostRecordingLLMClient(gateway.build_llm_client(settings, chat_client=recording_openai), ledger)
    embedder = LedgerEmbedder(gateway.build_embedder(settings), ledger, settings.embed_model)
    cross = gateway.build_cross_encoder(settings)
    return llm, embedder, cross


def episode_to_add_episode_kwargs(ep: Episode, structured_mode: str = 'native') -> dict[str, Any]:
    """Episode -> graphiti add_episode(...) 的关键字参数（纯函数，离线可测）。"""
    from graphiti_core.nodes import EpisodeType

    as_of = ep.provenance.get('as_of')
    reference_time = as_of if isinstance(as_of, datetime) else datetime.now(timezone.utc)
    return {
        'name': ep.episode_id,
        'episode_body': ep.content,
        'source_description': f'law_db:{ep.kind}',
        'reference_time': reference_time,
        'source': EpisodeType.text,
        'group_id': ep.group_id,
        # 不传 uuid：传入即走 EpisodicNode.get_by_uuid 路径，FalkorDB 下实测连接被服务端
        # 关闭（图主机 05:4x 两轮复现）；断点续跑由 runner 的 state 文件承担，不依赖图内 uuid。
        'entity_types': dict(ENTITY_TYPES),
        'custom_extraction_instructions': prompts.build_custom_extraction_instructions(
            structured_mode
        ),
    }


async def ingest_episode(graphiti: Any, episode: Episode, ledger: CostLedger | None = None,
                         structured_mode: str = 'native') -> Any:
    if ledger is not None:
        ledger.begin_episode(episode.episode_id)
    return await graphiti.add_episode(**episode_to_add_episode_kwargs(episode, structured_mode))


async def ingest_episodes(
    graphiti: Any, episodes: list[Episode], ledger: CostLedger | None = None,
    structured_mode: str = 'native'
) -> list[Any]:
    # graphiti 明确要求 episode 逐条顺序 await，不能并发
    results = []
    for ep in episodes:
        results.append(await ingest_episode(graphiti, ep, ledger, structured_mode))
    return results
