"""gateway：provider 钉扎 extra_body、结构化模式映射、缺配置显式失败。"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from graphiti_core.embedder.openai import OpenAIEmbedderConfig
from graphiti_core.prompts.models import Message

from fd_legal_kg import config, gateway
from fd_legal_kg.gateway import GatewayEmbedder, GatewayLLMClient, HintModeClient


def _chat_returning(content: str):
    """返回 content 作为消息体的 fake AsyncOpenAI 形状客户端。"""

    async def _create(**kwargs):
        return SimpleNamespace(
            model=kwargs.get('model', 'm0'),
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        texts = kwargs['input']
        n = len(texts) if isinstance(texts, list) else 1
        dim = int(kwargs.get('dimensions') or 4)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.5] * dim) for _ in range(n)])


def _embedder(settings, provider_only):
    fake_client = SimpleNamespace(embeddings=FakeEmbeddings())
    embedder = GatewayEmbedder(
        config=OpenAIEmbedderConfig(
            embedding_model=settings.embed_model,
            api_key=settings.embed_api_key,
            base_url=settings.embed_base_url,
            embedding_dim=settings.embed_dim,
        ),
        client=fake_client,
        provider_only=provider_only,
    )
    return embedder, fake_client.embeddings


async def test_provider_pin_extra_body_present_when_configured(settings):
    embedder, embeddings = _embedder(settings, provider_only='SiliconFlow')
    vec = await embedder.create('最高人民法院')
    assert len(vec) == settings.embed_dim
    assert embeddings.calls[0]['dimensions'] == settings.embed_dim  # 恒下发（服务端 MRL）
    assert embeddings.calls[0]['extra_body'] == {'provider': {'only': ['SiliconFlow']}}
    assert embeddings.calls[0]['model'] == 'Qwen/Qwen3-Embedding-8B'

    await embedder.create_batch(['a', 'b'])
    assert embeddings.calls[1]['extra_body'] == {'provider': {'only': ['SiliconFlow']}}


async def test_provider_pin_extra_body_absent_when_empty():
    s = config.Settings(
        llm_base_url='https://x', llm_api_key='k', llm_model='m', structured_mode='native',
        embed_base_url='https://y', embed_api_key='k2', embed_model='e',
        embed_provider_only=None, embed_dim=8,
        falkor_uri='bolt://127.0.0.1:6889', falkor_user=None, falkor_password=None,
        law_database_url=None, telemetry_optout=True, semaphore_limit=5,
        llm_max_tokens=32000,
        cost_ledger_path='data/cost_ledger.jsonl',
    )
    embedder, embeddings = _embedder(s, provider_only=None)
    await embedder.create('文本')
    assert 'extra_body' not in embeddings.calls[0]


@pytest.mark.parametrize('mode,expected', [
    ('native', 'json_schema'),
    ('json_object', 'json_object'),
    ('hint', 'json_object'),
])
def test_structured_mode_mapping(settings, monkeypatch, mode, expected):
    monkeypatch.setenv('LEXKG_LLM_STRUCTURED_MODE', mode)
    s = config.from_env()
    client = gateway.build_llm_client(s, chat_client=SimpleNamespace())
    assert client.structured_output_mode == expected
    assert isinstance(client, GatewayLLMClient)
    if mode == 'hint':
        assert isinstance(client, HintModeClient)
        # hint：不启用 API 级 JSON 约束，schema 只在 prompt 里提示
        assert client._build_response_format(None) == {'type': 'text'}
    else:
        assert not isinstance(client, HintModeClient)


async def test_deepseek_properties_envelope_unwrapped(settings):
    """live 验证：json_object 模式下 deepseek 会多包一层 {"properties": {...}} 信封。"""
    client = gateway.build_llm_client(
        settings,
        chat_client=_chat_returning('{"properties": {"entities": [], "edges": []}}'),
    )
    result = await client.generate_response([Message(role='user', content='抽取')])
    assert result == {'entities': [], 'edges': []}


async def test_non_envelope_payloads_pass_through_untouched(settings):
    plain = gateway.build_llm_client(settings, chat_client=_chat_returning('{"ok": true}'))
    assert await plain.generate_response([Message(role='user', content='x')]) == {'ok': True}

    schema_shell = gateway.build_llm_client(
        settings,
        chat_client=_chat_returning('{"properties": {"a": 1}, "title": "t"}'),
    )
    # title 属 schema 关键字 → 视作外壳剥掉（实测 deepseek 会回吐带 title 的 schema 头）
    assert await schema_shell.generate_response(
        [Message(role='user', content='x')]
    ) == {'a': 1}


def test_json_object_mode_note_injected_into_prompts():
    from fd_legal_kg import prompts

    with_note = prompts.build_custom_extraction_instructions('json_object')
    assert 'JSON' in with_note
    without = prompts.build_custom_extraction_instructions('native')
    assert 'JSON 输出任务' not in without
    for text in (with_note, without):
        assert '实体类型定义' in text and 'few-shot' in text


def test_gateway_requires_explicit_config():
    s = config.Settings(
        llm_base_url='', llm_api_key='', llm_model='m', structured_mode='native',
        embed_base_url='', embed_api_key='', embed_model='e',
        embed_provider_only='SiliconFlow', embed_dim=8,
        falkor_uri='bolt://127.0.0.1:6889', falkor_user=None, falkor_password=None,
        law_database_url=None, telemetry_optout=True, semaphore_limit=5,
        llm_max_tokens=32000,
        cost_ledger_path='data/cost_ledger.jsonl',
    )
    with pytest.raises(config.ConfigError, match='LEXKG_LLM_BASE_URL'):
        gateway.build_llm_client(s)
    with pytest.raises(config.ConfigError, match='LEXKG_EMBED_BASE_URL'):
        gateway.build_embedder(s)
    with pytest.raises(config.ConfigError, match='LEXKG_LLM_BASE_URL'):
        gateway.build_cross_encoder(s)


def test_build_llm_client_routes_to_gateway(settings):
    client = gateway.build_llm_client(settings, chat_client=SimpleNamespace())
    assert client.client is not None
    assert client.model == settings.llm_model
    assert client.config.base_url == settings.llm_base_url
    assert client.config.api_key == settings.llm_api_key


def test_cross_encoder_routes_to_gateway_not_openai(settings):
    ce = gateway.build_cross_encoder(settings, chat_client=SimpleNamespace())
    assert ce.client is not None
    assert ce.config.base_url == settings.llm_base_url


async def test_dimensions_always_sent_and_no_client_truncation(settings):
    """服务端 MRL 截断语义：必须下发 dimensions，且不得在客户端裁前 N 维。

    客户端硬砍前 N 维与服务端 MRL 不是同一空间（词元切换报告 §2.2/§四-事项2）。
    """
    embedder, embeddings = _embedder(settings, provider_only=None)
    vec = await embedder.create('最高人民法院')
    call = embeddings.calls[0]
    assert call['dimensions'] == settings.embed_dim
    assert 'extra_body' not in call
    assert len(vec) == settings.embed_dim


async def test_dimension_mismatch_fails_loudly(settings):
    """上游忽略 dimensions（返回 4096）时必须显式失败，不能静默入库到错误空间。"""
    import pytest as _pytest

    class WrongDim(FakeEmbeddings):
        async def create(self, **kwargs):
            self.calls.append(kwargs)
            texts = kwargs['input']
            n = len(texts) if isinstance(texts, list) else 1
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.5] * 4096) for _ in range(n)])

    fake = SimpleNamespace(embeddings=WrongDim())
    embedder = GatewayEmbedder(
        config=OpenAIEmbedderConfig(
            embedding_model=settings.embed_model, api_key=settings.embed_api_key,
            base_url=settings.embed_base_url, embedding_dim=settings.embed_dim,
        ),
        client=fake,
        provider_only=None,
    )
    with _pytest.raises(ValueError, match='dimension mismatch'):
        await embedder.create('文本')


def test_schema_envelope_unwrapped_in_both_shapes():
    """实测两种信封形状都要剥；含业务字段的真实结果不能剥。"""
    from fd_legal_kg.gateway import GatewayLLMClient as C

    # A: 只有 properties
    assert C._unwrap_schema_envelope({'properties': {'a': 1}}) == {'a': 1}
    # B: 整个 schema 外壳
    assert C._unwrap_schema_envelope(
        {'properties': {'duplicate_facts': []}, 'type': 'object'}
    ) == {'duplicate_facts': []}
    # C: 真实结果（properties 是业务字段之一）不剥
    real = {'properties': {'x': 1}, 'duplicate_facts': []}
    assert C._unwrap_schema_envelope(real) == real
    # D: 无 properties 不动
    assert C._unwrap_schema_envelope({'ok': True}) == {'ok': True}


def test_forgiving_json_parse_handles_raw_newlines_and_truncation():
    """实测 deepseek 高频产出：字符串内含未转义换行 / JSON 尾部被截断。"""
    from fd_legal_kg.gateway import GatewayLLMClient as C

    # 1) 字符串内含原始换行（严格 json.loads 会报 Invalid control character）
    raw_nl = '{"fact": "第一行\n第二行", "n": 1}'
    assert C._loads_forgiving(raw_nl) == {'fact': '第一行\n第二行', 'n': 1}

    # 2) 尾部截断在字符串中间（实测 Unterminated string）
    truncated = '{"duplicate_facts": ["事实一", "事实二"], "contradicted_facts": ["矛盾'
    got = C._loads_forgiving(truncated)
    assert got['duplicate_facts'] == ['事实一', '事实二']
    assert isinstance(got.get('contradicted_facts'), list)

    # 3) 尾部截断在数组中间
    trunc_arr = '{"entities": [{"name": "某法院"}, {"name": "某'
    got2 = C._loads_forgiving(trunc_arr)
    assert got2['entities'][0]['name'] == '某法院'


def test_schema_artifacts_stripped_from_values():
    """模型把 schema 片段当值回吐时降级为 None，真实值不受影响。"""
    from fd_legal_kg.gateway import GatewayLLMClient as C

    dirty = {
        'edges': [{'invalid_at': {'anyOf': [{'type': 'string'}], 'title': 'Invalid At'},
                   'fact': '某事实'}],
        'ok': 'keep',
    }
    out = C._strip_schema_artifacts(dirty)
    assert out['edges'][0]['invalid_at'] is None
    assert out['edges'][0]['fact'] == '某事实'
    assert out['ok'] == 'keep'
    # 正常 dict 不被误伤
    real = {'name': '某法院', 'level': {'type': '中级'}}
    assert C._strip_schema_artifacts(real) == real


def test_null_list_fields_normalized_to_empty_list():
    """实测 508 条跑批里 20 条死在模型把 list 字段回吐成 null。

    null 对这些字段的语义就是「没有」，还原成 [] 与模型本意等价。
    非列表字段的 null 必须原样保留，不能顺手清空。
    """
    from fd_legal_kg.gateway import GatewayLLMClient as C

    got = C._normalize_nulls(
        {'duplicate_facts': None, 'contradicted_facts': None, 'note': None}
    )
    assert got == {'duplicate_facts': [], 'contradicted_facts': [], 'note': None}

    assert C._normalize_nulls({'extracted_entities': None}) == {'extracted_entities': []}
    assert C._normalize_nulls({'summaries': None}) == {'summaries': []}
    # 嵌套（消解结果里的子字段）
    nested = {'entity_resolutions': None, 'items': [{'edges': None}]}
    assert C._normalize_nulls(nested) == {'entity_resolutions': [], 'items': [{'edges': []}]}
    # 已有内容不动
    real = {'duplicate_facts': [1, 2]}
    assert C._normalize_nulls(real) == real


def test_chinese_key_names_mapped_to_schema_names():
    """实测 1 条死在模型把键名直译成中文（{'提取实体': [...]}）。"""
    from fd_legal_kg.gateway import GatewayLLMClient as C

    got = C._normalize_nulls({'提取实体': [{'name': '某法院'}], '矛盾事实': None})
    assert got == {'extracted_entities': [{'name': '某法院'}], 'contradicted_facts': []}


async def test_gateway_5xx_retried_then_succeeds(settings, monkeypatch):
    """网关 502 必须被重试：graphiti 的 tenacity 只认 httpx.HTTPStatusError，
    而 OpenAI SDK 抛 openai.InternalServerError，两者无继承关系，故上面那层重试无效。"""
    import openai
    from graphiti_core.llm_client.config import LLMConfig

    calls = {'n': 0}

    async def _create(**kwargs):
        calls['n'] += 1
        if calls['n'] < 3:
            raise openai.InternalServerError(
                'Error code: 502',
                response=httpx.Response(502, request=httpx.Request('POST', 'https://x')),
                body=None,
            )
        return SimpleNamespace(
            model='m0',
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
        )

    client = GatewayLLMClient(
        config=LLMConfig(api_key='k', base_url='https://x', model='m0'),
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create))),
        structured_output_mode='json_object',
        max_tokens=128,
    )
    client._GATEWAY_BACKOFF = (0, 0, 0, 0)
    client._GATEWAY_RETRIES = 5

    out = await client._generate_response(
        [Message(role='user', content='请输出 JSON')], response_model=None, max_tokens=128
    )
    assert out == {'ok': True}
    assert calls['n'] == 3


def test_small_hardcoded_token_budget_is_raised_to_configured():
    """graphiti 在关系抽取里硬编码 extract_edges_max_tokens=16384，不看客户端配置。

    推理型模型先产 reasoning_content 再产正文，16384 会被思考吃光、正文 JSON
    截断在半句上，整条 episode 白跑（实测 gc:13230 等）。这里只上调不下调。
    """
    from graphiti_core.llm_client.config import LLMConfig

    client = GatewayLLMClient(
        config=LLMConfig(api_key='k', base_url='https://x', model='m'),
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace())),
        structured_output_mode='json_object',
        max_tokens=32000,
    )
    assert client._effective_max_tokens(16384) == 32000
    # 调用方要更多预算时照给（不下调）
    assert client._effective_max_tokens(64000) == 64000
    assert client._effective_max_tokens(None) == 32000
