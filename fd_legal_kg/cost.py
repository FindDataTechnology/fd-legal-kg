"""按 episode × operation 的 token 成本台账。

两层包装：
  CostRecordingLLMClient —— 包在 graphiti LLM 客户端外层，按 prompt_name/消息特征
      判定 operation（extraction / dedup / invalidation / other），把分类塞进
      ContextVar，再委托内层客户端；
  RecordingChat —— 包在 AsyncOpenAI 形状的客户端外层，从原始响应的 usage 里
      拿 prompt/completion tokens 落账（graphiti 的 _generate_response 会丢弃
      usage，只有这一层还能看到）。

embedding 的 token 数由 embedder 侧（gateway.ProviderPinnedEmbedder 之外再包一层
LedgerEmbedder，见 pipeline）以请求数/条数记账，prompt/completion tokens 记 0。
"""

from __future__ import annotations

import json
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.prompts.models import Message

OPERATIONS = ('extraction', 'dedup', 'invalidation', 'embedding', 'other')

_EXTRACTION_MARKERS = ('extract', 'entities', 'relationships', 'classify')
_DEDUP_MARKERS = ('dedup', 'duplicate')
_INVALIDATION_MARKERS = ('invalidat',)


def classify_operation(prompt_name: str | None, messages: list[Message] | None = None) -> str:
    """启发式归类一次 LLM 调用属于哪个环节（graphiti 大部分调用不传 prompt_name）。"""
    text = (prompt_name or '')
    if not text and messages:
        text = ' '.join(getattr(m, 'content', '') or '' for m in messages[:2]).lower()
    text = text.lower()
    if any(k in text for k in _INVALIDATION_MARKERS):
        return 'invalidation'
    if any(k in text for k in _DEDUP_MARKERS):
        return 'dedup'
    if any(k in text for k in _EXTRACTION_MARKERS):
        return 'extraction'
    return 'other'


@dataclass
class CostRecord:
    episode_id: str
    operation: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    ts: float
    meta: dict[str, Any] | None = None

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                'episode_id': self.episode_id,
                'operation': self.operation,
                'prompt_tokens': self.prompt_tokens,
                'completion_tokens': self.completion_tokens,
                'model': self.model,
                'ts': self.ts,
                'meta': self.meta,
            },
            ensure_ascii=False,
        )


_current_context: ContextVar[tuple[str, str] | None] = ContextVar('lexkg_cost_context', default=None)


class CostLedger:
    """JSONL 台账：一行一次调用。begin_episode 设定后续记录的默认 episode。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._episode_id = 'unknown'
        self.records: list[CostRecord] = []

    def begin_episode(self, episode_id: str) -> None:
        self._episode_id = episode_id

    def record(
        self,
        *,
        operation: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        episode_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> CostRecord:
        rec = CostRecord(
            episode_id=episode_id or self._episode_id,
            operation=operation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
            ts=time.time(),
            meta=meta,
        )
        self.records.append(rec)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as fh:
            fh.write(rec.to_jsonl() + '\n')
        return rec

    def aggregate(self) -> dict[str, dict[str, dict[str, int]]]:
        """按 episode × operation 汇总：{episode_id: {operation: {calls, prompt_tokens, completion_tokens, total_tokens}}}"""
        out: dict[str, dict[str, dict[str, int]]] = {}
        for rec in self.records:
            per_ep = out.setdefault(rec.episode_id, {})
            agg = per_ep.setdefault(
                rec.operation,
                {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            )
            agg['calls'] += 1
            agg['prompt_tokens'] += rec.prompt_tokens
            agg['completion_tokens'] += rec.completion_tokens
            agg['total_tokens'] += rec.prompt_tokens + rec.completion_tokens
        return out


class RecordingChatCompletions:
    def __init__(self, inner: Any, ledger: CostLedger):
        self._inner = inner
        self._ledger = ledger

    async def create(self, **kwargs: Any) -> Any:
        response = await self._inner.create(**kwargs)
        usage = getattr(response, 'usage', None)
        if usage is not None:
            ctx = _current_context.get()
            episode_id, operation = ctx if ctx else (None, 'other')
            self._ledger.record(
                operation=operation,
                model=str(getattr(response, 'model', None) or kwargs.get('model') or 'unknown'),
                prompt_tokens=getattr(usage, 'prompt_tokens', 0) or 0,
                completion_tokens=getattr(usage, 'completion_tokens', 0) or 0,
                episode_id=episode_id,
            )
        return response


class RecordingChat:
    def __init__(self, inner: Any, ledger: CostLedger):
        self.completions = RecordingChatCompletions(inner.chat.completions, ledger)


def _delegate(inner_name: str):
    """生成安全的 __getattr__：内层未绑定前抛 AttributeError，防递归。"""

    def __getattr__(self, name: str) -> Any:
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        inner = self.__dict__.get(inner_name)
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    return __getattr__


class RecordingAsyncOpenAI:
    """AsyncOpenAI 形状的包装：只拦截 chat.completions.create 记 usage，其余属性透传。"""

    def __init__(self, inner: Any, ledger: CostLedger):
        self._inner = inner
        self.chat = RecordingChat(inner, ledger)

    __getattr__ = _delegate('_inner')


class CostRecordingLLMClient(LLMClient):
    """graphiti LLMClient 的记账包装：判定 operation -> 设 ContextVar -> 委托内层。"""

    def __init__(self, inner: LLMClient, ledger: CostLedger):
        super().__init__(getattr(inner, 'config', None), cache=False)
        self._inner = inner
        self._ledger = ledger

    __getattr__ = _delegate('_inner')

    async def _generate_response(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return await self._inner._generate_response(*args, **kwargs)

    async def generate_response(
        self,
        messages: list[Message],
        response_model: Any = None,
        max_tokens: int | None = None,
        model_size: Any = None,
        group_id: str | None = None,
        prompt_name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        operation = classify_operation(prompt_name, messages)
        forward: dict[str, Any] = {
            'response_model': response_model,
            'max_tokens': max_tokens,
            'group_id': group_id,
            'prompt_name': prompt_name,
            **kwargs,
        }
        if model_size is not None:
            forward['model_size'] = model_size
        token = _current_context.set((self._ledger._episode_id, operation))
        try:
            return await self._inner.generate_response(messages, **forward)
        finally:
            _current_context.reset(token)


class LedgerEmbedder(EmbedderClient):
    """包在任意 embedder 外层记 embedding 成本（按请求与文本条数，token 记 0）。

    必须继承 EmbedderClient：Graphiti 构造期用 isinstance 校验 embedder 类型。
    """

    def __init__(self, inner: Any, ledger: CostLedger, model: str):
        self._inner = inner
        self._ledger = ledger
        self._model = model

    async def create(self, input_data: Any) -> list[float]:
        result = await self._inner.create(input_data)
        self._ledger.record(
            operation='embedding',
            model=self._model,
            prompt_tokens=0,
            completion_tokens=0,
            meta={'requests': 1, 'texts': 1},
        )
        return result

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        result = await self._inner.create_batch(input_data_list)
        self._ledger.record(
            operation='embedding',
            model=self._model,
            prompt_tokens=0,
            completion_tokens=0,
            meta={'requests': 1, 'texts': len(input_data_list)},
        )
        return result

    __getattr__ = _delegate('_inner')
