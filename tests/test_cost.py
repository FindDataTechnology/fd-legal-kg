"""cost：台账记录、operation 分类、聚合、JSONL 落盘（fake 响应，零网络）。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from graphiti_core.prompts.models import Message

from fd_legal_kg import config, gateway
from fd_legal_kg.cost import (
    CostLedger,
    CostRecordingLLMClient,
    LedgerEmbedder,
    RecordingAsyncOpenAI,
    classify_operation,
)


class FakeCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs.get('model', 'm0'),
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
        )


def _fake_chat() -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))


def _llm(settings: config.Settings, chat) -> CostRecordingLLMClient:
    # 与 pipeline.build_cost_wired_clients 相同的层次：记账包装 -> 网关 LLM -> 记账 LLM
    ledger = CostLedger('/tmp/unused-ledger.jsonl')
    recording_openai = RecordingAsyncOpenAI(chat, ledger)
    return CostRecordingLLMClient(
        gateway.build_llm_client(settings, chat_client=recording_openai), ledger
    )


async def test_llm_call_records_usage_with_operation(settings):
    chat = _fake_chat()
    client = _llm(settings, chat)
    client._ledger.begin_episode('guide_case:gc-1')

    result = await client.generate_response(
        [Message(role='user', content='extract all entities from the passage')]
    )
    assert result == {'ok': True}
    assert len(client._ledger.records) == 1
    rec = client._ledger.records[0]
    assert rec.episode_id == 'guide_case:gc-1'
    assert rec.operation == 'extraction'
    assert rec.prompt_tokens == 11
    assert rec.completion_tokens == 7
    assert rec.model == settings.llm_model


async def test_prompt_name_drives_operation_classification(settings):
    chat = _fake_chat()
    client = _llm(settings, chat)
    client._ledger.begin_episode('e1')

    await client.generate_response(
        [Message(role='user', content='x')], prompt_name='dedupe_nodes.node'
    )
    await client.generate_response(
        [Message(role='user', content='x')], prompt_name='extract_edges.extract_timestamps'
    )
    await client.generate_response(
        [Message(role='user', content='list any invalidated edges')]
    )
    await client.generate_response(
        [Message(role='user', content='summarize this community')
         ], prompt_name='summarize_nodes.summary_description'
    )
    ops = [r.operation for r in client._ledger.records]
    assert ops == ['dedup', 'extraction', 'invalidation', 'other']


def test_classify_operation_markers():
    assert classify_operation(None, [Message(role='system', content='Find duplicate entities')]) == 'dedup'
    assert classify_operation('extract_nodes.extract_message') == 'extraction'
    assert classify_operation('plain_name') == 'other'
    assert classify_operation(None, [Message(role='user', content='hello')]) == 'other'


def test_ledger_writes_jsonl_and_aggregates(tmp_path):
    path = tmp_path / 'ledger' / 'cost.jsonl'
    ledger = CostLedger(path)
    ledger.begin_episode('ep-a')
    ledger.record(operation='extraction', model='m', prompt_tokens=100, completion_tokens=50)
    ledger.record(operation='extraction', model='m', prompt_tokens=10, completion_tokens=5)
    ledger.record(operation='dedup', model='m', prompt_tokens=1, completion_tokens=1)
    ledger.begin_episode('ep-b')
    ledger.record(operation='embedding', model='emb', prompt_tokens=0, completion_tokens=0,
                  meta={'texts': 3})

    lines = path.read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) == 4
    assert '"episode_id": "ep-a"' in lines[0]

    agg = ledger.aggregate()
    assert agg['ep-a']['extraction'] == {
        'calls': 2, 'prompt_tokens': 110, 'completion_tokens': 55, 'total_tokens': 165,
    }
    assert agg['ep-a']['dedup']['calls'] == 1
    assert agg['ep-b']['embedding']['calls'] == 1
    assert set(agg) == {'ep-a', 'ep-b'}


class FakeEmbedderInner:
    async def create(self, input_data):
        return [0.1, 0.2]

    async def create_batch(self, input_data_list):
        return [[0.1] for _ in input_data_list]


async def test_ledger_embedder_records_texts(tmp_path):
    ledger = CostLedger(tmp_path / 'l.jsonl')
    ledger.begin_episode('ep-e')
    embedder = LedgerEmbedder(FakeEmbedderInner(), ledger, 'Qwen/Qwen3-Embedding-8B')

    vec = await embedder.create('text')
    assert vec == [0.1, 0.2]
    await embedder.create_batch(['a', 'b', 'c'])

    records = [r for r in ledger.records if r.operation == 'embedding']
    assert [r.meta['texts'] for r in records] == [1, 3]
    assert all(r.model == 'Qwen/Qwen3-Embedding-8B' for r in records)


def test_recording_openai_delegates_other_attributes(settings):
    from fd_legal_kg.cost import RecordingAsyncOpenAI

    inner = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()), api_key='k')
    wrapped = RecordingAsyncOpenAI(inner, CostLedger('/tmp/x.jsonl'))
    assert wrapped.api_key == 'k'
    assert wrapped.chat.completions is not inner.chat.completions  # chat 被包装
    # 未知属性经内层解析，内层也没有则 AttributeError（不递归）
    with pytest.raises(AttributeError):
        _ = wrapped.does_not_exist_anywhere
