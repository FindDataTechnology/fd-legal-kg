"""环境变量驱动的设置：所有外部依赖（LLM 网关、嵌入网关、FalkorDB、law_db）只从这里进。

原则：缺必填项显式抛 :class:`ConfigError`（带确切的环境变量名），绝不静默换默认值或换 provider。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

STRUCTURED_MODES = ('native', 'json_object', 'hint')

DEFAULT_LLM_MODEL = 'deepseek-v4.1-flash'  # live 验证过的模型名（无厂商前缀；glm-5.3-flash 无账号池）
DEFAULT_EMBED_MODEL = 'Qwen/Qwen3-Embedding-8B'
DEFAULT_LLM_MAX_TOKENS = 32000  # 推理型模型需更大预算，16384 实测会截断
DEFAULT_EMBED_PROVIDER_ONLY = ''  # 词元通道由网关注入 Nebius；直连硅基流动单一上游无需钉扎
DEFAULT_EMBED_DIM = 1024
DEFAULT_FALKORDB_URI = 'bolt://127.0.0.1:6889'
DEFAULT_SEMAPHORE_LIMIT = 5
DEFAULT_COST_LEDGER_PATH = 'data/cost_ledger.jsonl'


class ConfigError(RuntimeError):
    """配置缺失或非法。message 中一定包含触发的环境变量名。"""


def _required_env(name: str) -> str:
    value = os.environ.get(name, '').strip()
    if not value:
        raise ConfigError(f'missing required env var {name}')
    return value


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name, '').strip()
    return value or None


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, '').strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f'env var {name} must be an integer, got {raw!r}') from exc


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, '').strip().lower()
    if not raw:
        return default
    if raw in ('1', 'true', 'yes', 'on'):
        return True
    if raw in ('0', 'false', 'no', 'off'):
        return False
    raise ConfigError(f'env var {name} must be a boolean, got {raw!r}')


def parse_falkor_uri(uri: str) -> tuple[str, int]:
    """bolt://host:port / redis://host:port -> (host, port)。

    graphiti 0.31.1 的 FalkorDriver 只接受 host/port 构造参数（没有 URI 形参）。
    """
    parsed = urlparse(uri if '://' in uri else f'bolt://{uri}')
    if not parsed.hostname:
        raise ConfigError(f'LEXKG_FALKORDB_URI cannot be parsed into host:port: {uri!r}')
    return parsed.hostname, parsed.port or 6889


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    structured_mode: str
    embed_base_url: str
    embed_api_key: str
    embed_model: str
    embed_provider_only: str | None
    embed_dim: int
    falkor_uri: str
    falkor_user: str | None
    falkor_password: str | None
    law_database_url: str | None
    telemetry_optout: bool
    semaphore_limit: int
    llm_max_tokens: int
    cost_ledger_path: str

    def falkor_endpoint(self) -> tuple[str, int]:
        return parse_falkor_uri(self.falkor_uri)

    def apply_runtime_env(self) -> None:
        """把 LEXKG_* 设置翻译成第三方库读取的环境变量（当前只有 graphiti 遥测开关）。"""
        os.environ['GRAPHITI_TELEMETRY_ENABLED'] = 'false' if self.telemetry_optout else 'true'

    def require_law_database_url(self) -> str:
        if not self.law_database_url:
            raise ConfigError('missing required env var LEXKG_LAW_DATABASE_URL (episode supply)')
        return self.law_database_url


def from_env() -> Settings:
    """读取全部 LEXKG_* 环境变量。LLM/嵌入网关的 base_url 与 api_key 为必填。"""
    mode = os.environ.get('LEXKG_LLM_STRUCTURED_MODE', 'native').strip() or 'native'
    if mode not in STRUCTURED_MODES:
        raise ConfigError(
            f'env var LEXKG_LLM_STRUCTURED_MODE must be one of {STRUCTURED_MODES}, got {mode!r}'
        )
    return Settings(
        llm_base_url=_required_env('LEXKG_LLM_BASE_URL'),
        llm_api_key=_required_env('LEXKG_LLM_API_KEY'),
        llm_model=os.environ.get('LEXKG_LLM_MODEL', '').strip() or DEFAULT_LLM_MODEL,
        structured_mode=mode,
        embed_base_url=_required_env('LEXKG_EMBED_BASE_URL'),
        embed_api_key=_required_env('LEXKG_EMBED_API_KEY'),
        embed_model=os.environ.get('LEXKG_EMBED_MODEL', '').strip() or DEFAULT_EMBED_MODEL,
        embed_provider_only=os.environ.get('LEXKG_EMBED_PROVIDER_ONLY', '').strip()
        or DEFAULT_EMBED_PROVIDER_ONLY,
        embed_dim=_int_env('LEXKG_EMBED_DIM', DEFAULT_EMBED_DIM),
        falkor_uri=os.environ.get('LEXKG_FALKORDB_URI', '').strip() or DEFAULT_FALKORDB_URI,
        falkor_user=_optional_env('LEXKG_FALKORDB_USER'),
        falkor_password=_optional_env('LEXKG_FALKORDB_PASSWORD'),
        law_database_url=_optional_env('LEXKG_LAW_DATABASE_URL'),
        telemetry_optout=_bool_env('LEXKG_TELEMETRY_OPTOUT', True),
        semaphore_limit=_int_env('LEXKG_SEMAPHORE_LIMIT', DEFAULT_SEMAPHORE_LIMIT),
        llm_max_tokens=_int_env('LEXKG_LLM_MAX_TOKENS', DEFAULT_LLM_MAX_TOKENS),
        cost_ledger_path=os.environ.get('LEXKG_COST_LEDGER_PATH', '').strip()
        or DEFAULT_COST_LEDGER_PATH,
    )
