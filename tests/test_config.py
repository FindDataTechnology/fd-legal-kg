"""config：环境解析、默认值、显式失败、遥测开关、FalkorDB URI 解析。"""

from __future__ import annotations

import pytest

from fd_legal_kg import config
from tests.conftest import REQUIRED_ENV


def test_defaults(settings):
    assert settings.llm_model == 'deepseek-v4.1-flash'
    assert settings.embed_model == 'Qwen/Qwen3-Embedding-8B'
    assert settings.embed_provider_only == ''  # 钉扎由网关注入（词元通道）
    assert settings.embed_dim == 1024
    assert settings.falkor_uri == 'bolt://127.0.0.1:6889'
    assert settings.falkor_user is None
    assert settings.falkor_password is None
    assert settings.telemetry_optout is True
    assert settings.semaphore_limit == 5
    assert settings.cost_ledger_path == 'data/cost_ledger.jsonl'
    assert settings.structured_mode == 'native'


def test_env_overrides(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv('LEXKG_LLM_MODEL', 'other/model')
    monkeypatch.setenv('LEXKG_EMBED_PROVIDER_ONLY', 'UpstreamA')
    monkeypatch.setenv('LEXKG_EMBED_DIM', '768')
    monkeypatch.setenv('LEXKG_FALKORDB_URI', 'bolt://falkor.internal:7000')
    monkeypatch.setenv('LEXKG_FALKORDB_USER', 'u')
    monkeypatch.setenv('LEXKG_FALKORDB_PASSWORD', 'p')
    monkeypatch.setenv('LEXKG_TELEMETRY_OPTOUT', 'false')
    monkeypatch.setenv('LEXKG_SEMAPHORE_LIMIT', '12')
    monkeypatch.setenv('LEXKG_COST_LEDGER_PATH', '/tmp/ledger.jsonl')

    s = config.from_env()
    assert s.llm_model == 'other/model'
    assert s.embed_provider_only == 'UpstreamA'
    assert s.embed_dim == 768
    assert s.falkor_endpoint() == ('falkor.internal', 7000)
    assert s.falkor_user == 'u' and s.falkor_password == 'p'
    assert s.telemetry_optout is False
    assert s.semaphore_limit == 12
    assert s.cost_ledger_path == '/tmp/ledger.jsonl'


def test_empty_embed_provider_pin_disables_pinning(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv('LEXKG_EMBED_PROVIDER_ONLY', '')
    assert config.from_env().embed_provider_only == ''  # 默认即空：钉扎交给网关


@pytest.mark.parametrize('missing', ['LEXKG_LLM_BASE_URL', 'LEXKG_LLM_API_KEY',
                                     'LEXKG_EMBED_BASE_URL', 'LEXKG_EMBED_API_KEY'])
def test_missing_required_raises_with_var_name(monkeypatch, missing):
    for key, value in REQUIRED_ENV.items():
        if key != missing:
            monkeypatch.setenv(key, value)
    with pytest.raises(config.ConfigError, match=missing):
        config.from_env()


def test_law_db_url_required_only_by_supplier(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        if key != 'LEXKG_LAW_DATABASE_URL':
            monkeypatch.setenv(key, value)
    s = config.from_env()  # law_database_url 可空：不连库的命令不应被卡
    with pytest.raises(config.ConfigError, match='LEXKG_LAW_DATABASE_URL'):
        s.require_law_database_url()


def test_invalid_structured_mode(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv('LEXKG_LLM_STRUCTURED_MODE', 'yaml')
    with pytest.raises(config.ConfigError, match='LEXKG_LLM_STRUCTURED_MODE'):
        config.from_env()


@pytest.mark.parametrize('mode', ['native', 'json_object', 'hint'])
def test_valid_structured_modes(monkeypatch, mode):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv('LEXKG_LLM_STRUCTURED_MODE', mode)
    assert config.from_env().structured_mode == mode


def test_bad_int_and_bool(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv('LEXKG_SEMAPHORE_LIMIT', 'many')
    with pytest.raises(config.ConfigError, match='LEXKG_SEMAPHORE_LIMIT'):
        config.from_env()
    monkeypatch.setenv('LEXKG_SEMAPHORE_LIMIT', '5')
    monkeypatch.setenv('LEXKG_TELEMETRY_OPTOUT', 'maybe')
    with pytest.raises(config.ConfigError, match='LEXKG_TELEMETRY_OPTOUT'):
        config.from_env()


def test_apply_runtime_env_telemetry(settings, monkeypatch):
    settings.apply_runtime_env()
    import os

    assert os.environ['GRAPHITI_TELEMETRY_ENABLED'] == 'false'

    monkeypatch.setenv('LEXKG_TELEMETRY_OPTOUT', '0')
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    config.from_env().apply_runtime_env()
    assert os.environ['GRAPHITI_TELEMETRY_ENABLED'] == 'true'


def test_parse_falkor_uri():
    assert config.parse_falkor_uri('bolt://127.0.0.1:6889') == ('127.0.0.1', 6889)
    assert config.parse_falkor_uri('redis://graph.example:6379') == ('graph.example', 6379)
    assert config.parse_falkor_uri('127.0.0.1:6889') == ('127.0.0.1', 6889)  # 无 scheme 容错
    with pytest.raises(config.ConfigError):
        config.parse_falkor_uri('::::')


def test_telemetry_env_set_at_package_import_before_graphiti(monkeypatch):
    """live 验证要求：GRAPHITI_TELEMETRY_ENABLED 必须在 import graphiti 之前落 env。

    包 __init__ 在子模块（gateway/pipeline import graphiti）之前执行，
    这里用子进程验证「import fd_legal_kg 即生效」。
    """
    import os
    import subprocess
    import sys

    monkeypatch.delenv('LEXKG_TELEMETRY_OPTOUT', raising=False)
    monkeypatch.delenv('GRAPHITI_TELEMETRY_ENABLED', raising=False)

    subprocess.run(
        [sys.executable, '-c',
         'import os, fd_legal_kg; assert os.environ["GRAPHITI_TELEMETRY_ENABLED"] == "false"'],
        check=True,
    )
    # 显式关闭 opt-out 时不设置；用户手设的 GRAPHITI_TELEMETRY_ENABLED 被尊重（setdefault）
    subprocess.run(
        [sys.executable, '-c',
         'import os, fd_legal_kg; assert "GRAPHITI_TELEMETRY_ENABLED" not in os.environ'],
        check=True, env={**os.environ, 'LEXKG_TELEMETRY_OPTOUT': 'false'},
    )
    subprocess.run(
        [sys.executable, '-c',
         'import os, fd_legal_kg; assert os.environ["GRAPHITI_TELEMETRY_ENABLED"] == "true"'],
        check=True, env={**os.environ, 'GRAPHITI_TELEMETRY_ENABLED': 'true'},
    )
