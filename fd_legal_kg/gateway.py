"""OpenAI 兼容网关的客户端构造：LLM、Embedder、CrossEncoder 全部走 LEXKG_* 网关。

嵌入的两个硬约束（2026-10-18 词元嵌入通道切换，见 paas change ciyuan-embedding-lane）：

1. **provider 钉扎由网关注入**：词元账号级配置 extra.embedding_provider_only=Nebius，
   网关会覆盖调用方自带的 provider 字段（实测传不存在的 provider 仍 200）。
   故 LEXKG_EMBED_PROVIDER_ONLY 在词元通道下是无效配置，切换时应置空；
   直连硅基流动（单一上游）时也无需钉扎——留空即可。
2. **dimensions 必须由调用方显式下发**：不带 dimensions 时上游返回 4096 维（8B 原生），
   网关不做截断。服务端 dimensions=1024 是 MRL 截断（与原生前 1024 维余弦 0.999892），
   而客户端 `embedding[:1024]` 是硬砍前 N 维——**两者不是同一空间**，混用会让库内
   向量与查询向量不可比。这里一律下发 dimensions，并禁止客户端截断。
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from . import config

logger = logging.getLogger(__name__)

# LEXKG_LLM_STRUCTURED_MODE -> graphiti OpenAIGenericClient.structured_output_mode。
# graphiti 0.31.1 只接受 'json_schema' | 'json_object'（没有 'native'/'hint' 这两个值）。
STRUCTURED_MODE_MAP: dict[str, str] = {
    'native': 'json_schema',
    'json_object': 'json_object',
    'hint': 'json_object',
}


class GatewayLLMClient(OpenAIGenericClient):
    """网关 LLM 客户端基类：解包 json_object 模式下模型回吐的 schema 外壳。

    live 验证（deepseek-v4.1-flash, json_object 模式）：模型会把 schema 片段误当
    输出结构包一层，且形状不稳定，实测出现过两种：
      A. {"properties": {...}}                      （只有 properties）
      B. {"properties": {...}, "type": "object"}    （整个 schema 外壳）
    graphiti 只剥 ``` 围栏、不解这两种信封，下游会拿到外壳而非结果本体
    （表现为 EdgeDuplicate 缺 duplicate_facts 等校验错）。这里统一剥掉。
    """

    # 网关侧瞬时故障（502/503/504）的重试次数。graphiti 的 tenacity 只认
    # httpx.HTTPStatusError，而 OpenAI SDK 抛的是 openai.InternalServerError——
    # 两者无继承关系，故上面那层重试对 502 完全无效（实测 508 条里 35 条死在 502）。
    _GATEWAY_RETRIES = 5
    _GATEWAY_BACKOFF = (5, 15, 30, 60)  # 秒，逐次递增；共 5 次尝试

    # 模型把 schema 里的 "list" 输出成 null 时，按字段名还原成空列表。
    # 实测 deepseek-v4.1-flash 在 json_object 模式下高频如此（508 条里 20 条），
    # 而这四个字段的 null 语义就是「没有」——空列表与模型本意等价，不是降级猜测。
    _LIST_FIELDS = ('duplicate_facts', 'contradicted_facts', 'entity_resolutions',
                    'summaries', 'extracted_entities', 'edges')

    # 模型偶尔用中文键名回吐（实测 {'提取实体': [...]}）。这些是 schema 字段的
    # 直译，映射回英文后语义不变。
    _KEY_ALIASES = {
        '提取实体': 'extracted_entities',
        '重复事实': 'duplicate_facts',
        '矛盾事实': 'contradicted_facts',
        '实体消解': 'entity_resolutions',
        '摘要': 'summaries',
        '事实': 'edges',
        '边': 'edges',
    }

    @staticmethod
    def _normalize_nulls(data: Any) -> Any:
        """把列表字段的 null 还原成 []（递归），并映射中文键名。"""
        if isinstance(data, list):
            return [GatewayLLMClient._normalize_nulls(v) for v in data]
        if not isinstance(data, dict):
            return data
        out: dict[str, Any] = {}
        for key, value in data.items():
            canonical = GatewayLLMClient._KEY_ALIASES.get(key, key)
            if value is None and canonical in GatewayLLMClient._LIST_FIELDS:
                out[canonical] = []
            else:
                out[canonical] = GatewayLLMClient._normalize_nulls(value)
        return out

    @staticmethod
    def _unwrap_schema_envelope(data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        props = data.get('properties')
        if not isinstance(props, dict):
            return data
        # 信封判定：除 properties 外只允许 schema 关键字。出现任何业务字段说明
        # 这是真实结果，不能剥（避免把正常输出里的 properties 键误伤）。
        schema_keys = {'type', 'required', 'additionalProperties', '$defs',
                       'description', 'title', 'definitions', '$schema'}
        if set(data.keys()) - {'properties'} - schema_keys:
            return data
        return props

    @staticmethod
    def _loads_forgiving(text: str) -> Any:
        """容错解析：先严格，再 strict=False（容忍字符串内原始控制字符/换行），
        最后尝试截断修复（推理模型把 JSON 写到一半被 max_tokens 截断）。

        实测 deepseek-v4.1-flash 在中文长文本上高频产出「字符串内含未转义换行」
        与「尾部截断」，严格 json.loads 直接抛 Unterminated string / Invalid control
        character，会让整条 episode 白跑（每条约 13 万 token）。
        """
        text = text.strip()
        for loader in (lambda t: json.loads(t), lambda t: json.loads(t, strict=False)):
            try:
                return loader(text)
            except json.JSONDecodeError:
                pass
        # 截断修复：从尾部逐步回退，补齐未闭合的字符串/括号
        trimmed = text
        for _ in range(2000):
            candidate = trimmed
            # 若停在字符串中间，先闭合引号
            in_string = False
            escaped = False
            for ch in candidate:
                if escaped:
                    escaped = False
                    continue
                if ch == '\\':
                    escaped = True
                elif ch == '"':
                    in_string = not in_string
            if in_string:
                candidate += '"'
            # 补齐未闭合的括号
            opens = candidate.count('{') - candidate.count('}')
            squares = candidate.count('[') - candidate.count(']')
            candidate += ']' * max(squares, 0) + '}' * max(opens, 0)
            try:
                return json.loads(candidate, strict=False)
            except json.JSONDecodeError:
                # 退掉最后一个可能不完整的 token 再试
                cut = max(trimmed.rfind(','), trimmed.rfind(':'))
                if cut <= 0:
                    break
                trimmed = trimmed[:cut]
        raise ValueError(
            f'unparseable LLM JSON (len={len(text)}): {text[:200]!r} ... {text[-120:]!r}'
        )

    @staticmethod
    def _strip_schema_artifacts(data: Any) -> Any:
        """清理模型把 schema 描述当成值回吐的产物（递归）。

        实测：deepseek 会把 `{'anyOf': [...], 'title': 'Invalid At'}` 这类 schema
        片段塞进字段值里，导致 pydantic 报 "Input should be a valid string"。
        规则：dict 若只由 schema 关键字构成，视为噪声，降级为 None。
        """
        # 必须含纯结构关键字才算 schema 片段：单靠 type/format 等会被业务值
        #（如 {'type': '中级'}）误伤，故以 anyOf/allOf/oneOf/$defs 为触发条件。
        structural = {'anyOf', 'allOf', 'oneOf', '$defs', 'definitions', 'items',
                      'additionalProperties', 'pattern'}
        schema_words = structural | {'type', 'title', 'description', 'format', 'enum',
                                      'properties', 'required', '$schema', 'default',
                                      'examples', 'minimum', 'maximum', 'minLength',
                                      'maxLength', 'const'}
        if isinstance(data, dict):
            keys = set(data.keys())
            if data and keys & structural and keys <= schema_words:
                return None
            return {
                k: GatewayLLMClient._strip_schema_artifacts(v) for k, v in data.items()
            }
        if isinstance(data, list):
            return [GatewayLLMClient._strip_schema_artifacts(v) for v in data]
        return data

    async def _generate_response(
        self,
        messages: list[Any],
        response_model: Any = None,
        max_tokens: int = 16384,
        model_size: Any = None,
        *,
        model: str | None = None,
    ) -> Any:
        """自建请求+容错解析（父类内部是严格 json.loads，没有解析钩子）。

        与父类的差异仅在解析：请求构造、response_format、错误分类保持一致。
        """
        import openai

        from graphiti_core.llm_client.errors import EmptyResponseError, RateLimitError

        openai_messages: list[dict[str, Any]] = []
        for m in messages:
            content = self._clean_input(m.content)
            if m.role in ('user', 'system'):
                openai_messages.append({'role': m.role, 'content': content})

        response = await self._create_with_gateway_retry(
            model=model or self.model or 'gpt-4.1-mini',
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self._effective_max_tokens(max_tokens),
            response_format=self._build_response_format(response_model),
        )
        try:
            result = response.choices[0].message.content or ''
            if not result.strip():
                raise EmptyResponseError('LLM returned an empty response')
            parsed = self._loads_forgiving(self._strip_code_fences(result))
            parsed = self._unwrap_schema_envelope(parsed)
            parsed = self._strip_schema_artifacts(parsed)
            return self._normalize_nulls(parsed)
        except openai.RateLimitError as exc:
            raise RateLimitError from exc
        except (EmptyResponseError, RateLimitError, ValueError):
            raise
        except Exception as exc:
            logger.error('Error in generating LLM response: %s', exc)
            raise

    def _effective_max_tokens(self, requested: int | None) -> int:
        """把 graphiti 传下来的偏小预算抬到配置值。

        实测 graphiti 0.31.1 在关系抽取里硬编码 ``extract_edges_max_tokens = 16384``
        （见 utils/maintenance/edge_operations.py），不看客户端的 max_tokens 配置。
        对推理型模型（deepseek-v4.1-flash 先产 reasoning_content 再产正文），
        16384 会被思考吃光、正文 JSON 截断在半句上，整条 episode 白跑。
        故只上调不下调：调用方要更多预算时照给。
        """
        configured = int(getattr(self, 'max_tokens', 0) or 0)
        want = int(requested or 0)
        return max(want, configured) if configured else want

    async def _create_with_gateway_retry(self, **kwargs: Any) -> Any:
        """对网关 5xx 做进程内退避重试。

        OpenAI SDK 自带 max_retries=2 且只重试 408/409/429/5xx 的特定形态；
        实测网关 502 会连续稳定失败（上游账号池切换期），需要更长的耐心。
        """
        import asyncio

        import openai

        last: Exception | None = None
        for attempt in range(self._GATEWAY_RETRIES):
            try:
                return await self.client.chat.completions.create(**kwargs)
            except (openai.InternalServerError, openai.APIStatusError,
                    httpx.HTTPStatusError) as exc:
                status = getattr(getattr(exc, 'response', None), 'status_code', None)
                transient = isinstance(exc, openai.InternalServerError) or (
                    status is not None and 500 <= int(status) < 600
                ) or isinstance(exc, httpx.HTTPStatusError) and (
                    500 <= exc.response.status_code < 600
                )
                if not transient or attempt == self._GATEWAY_RETRIES - 1:
                    raise
                last = exc
                delay = self._GATEWAY_BACKOFF[min(attempt, len(self._GATEWAY_BACKOFF) - 1)]
                logger.warning(
                    'gateway %s (attempt %d/%d), retrying in %ds',
                    status or type(exc).__name__, attempt + 1, self._GATEWAY_RETRIES, delay,
                )
                await asyncio.sleep(delay)
        raise last  # pragma: no cover - 循环内必已返回或抛出


class HintModeClient(GatewayLLMClient):
    """hint 模式：结构只靠 prompt 里的 schema 提示，不用 API 级 JSON 约束。

    借道 json_object 模式（其 generate_response 会把 schema 注入 prompt），
    同时把 response_format 降级为 {'type': 'text'}，即不启用任何 JSON 强制。
    """

    def _build_response_format(self, response_model: Any) -> dict[str, Any]:
        return {'type': 'text'}


class GatewayEmbedder(OpenAIEmbedder):
    """嵌入客户端：显式下发 dimensions（服务端 MRL 截断），不做客户端截断。

    - dimensions 恒下发，除非 embedding_dim 未配置（此时由上游决定，且下面会校验）。
    - provider_only 仅在直连多上游、需要钉死时使用；词元通道置空（网关注入 Nebius）。
    - 返回维度与配置不符时显式报错：库内向量空间一致性比"能跑"重要。
    """

    def __init__(self, config: OpenAIEmbedderConfig | None = None, client: Any = None,
                 provider_only: str | None = None):
        super().__init__(config=config, client=client)
        self.provider_only = provider_only or None

    @property
    def _extra_body(self) -> dict[str, Any] | None:
        if not self.provider_only:
            return None
        return {'provider': {'only': [self.provider_only]}}

    def _kwargs(self, input_data: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {'input': input_data, 'model': self.config.embedding_model}
        if self.config.embedding_dim:
            kwargs['dimensions'] = int(self.config.embedding_dim)
        if self._extra_body is not None:
            kwargs['extra_body'] = self._extra_body
        return kwargs

    def _check_dim(self, vec: list[float]) -> list[float]:
        expected = int(self.config.embedding_dim or 0)
        if expected and len(vec) != expected:
            raise ValueError(
                f'embedding dimension mismatch: got {len(vec)}, LEXKG_EMBED_DIM={expected} '
                f'(upstream ignores `dimensions`? check the gateway/model)'
            )
        return vec

    async def create(self, input_data: Any) -> list[float]:
        result = await self.client.embeddings.create(**self._kwargs(input_data))
        return self._check_dim(result.data[0].embedding)

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        result = await self.client.embeddings.create(**self._kwargs(input_data_list))
        return [self._check_dim(e.embedding) for e in result.data]


def build_llm_client(settings: config.Settings, chat_client: Any = None) -> OpenAIGenericClient:
    """构造走网关的 LLM 客户端。缺 base_url/api_key 显式报错，绝不回落 OpenAI 官方。"""
    if not settings.llm_base_url or not settings.llm_api_key:
        raise config.ConfigError(
            'LEXKG_LLM_BASE_URL and LEXKG_LLM_API_KEY are required to build the LLM client'
        )
    llm_config = LLMConfig(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )
    client_cls = HintModeClient if settings.structured_mode == 'hint' else GatewayLLMClient
    return client_cls(
        config=llm_config,
        client=chat_client,
        structured_output_mode=STRUCTURED_MODE_MAP[settings.structured_mode],
        # 推理型模型（deepseek-v4.1-flash）先产 reasoning_content 再产正文；
        # 16384 实测会被思考吃光导致 JSON 截断（Unterminated string）。放大预算。
        max_tokens=settings.llm_max_tokens,
    )


def build_embedder(settings: config.Settings, http_client: Any = None) -> GatewayEmbedder:
    """构造嵌入客户端。provider 钉扎由网关注入（词元通道），故 provider_only 通常为空。"""
    if not settings.embed_base_url or not settings.embed_api_key:
        raise config.ConfigError(
            'LEXKG_EMBED_BASE_URL and LEXKG_EMBED_API_KEY are required to build the embedder'
        )
    embedder_config = OpenAIEmbedderConfig(
        embedding_model=settings.embed_model,
        api_key=settings.embed_api_key,
        base_url=settings.embed_base_url,
        embedding_dim=settings.embed_dim,
    )
    return GatewayEmbedder(
        config=embedder_config,
        client=http_client,
        provider_only=settings.embed_provider_only,
    )


def build_cross_encoder(settings: config.Settings, chat_client: Any = None) -> OpenAIRerankerClient:
    """构造走同一网关的 reranker。

    Graphiti 构造函数在 cross_encoder 缺省时会急切地 new 一个 OpenAI 官方
    OpenAIRerankerClient（读 OPENAI_API_KEY），必须显式传入网关版。
    """
    if not settings.llm_base_url or not settings.llm_api_key:
        raise config.ConfigError(
            'LEXKG_LLM_BASE_URL and LEXKG_LLM_API_KEY are required to build the cross encoder'
        )
    return OpenAIRerankerClient(
        config=LLMConfig(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
        ),
        client=chat_client,
    )
