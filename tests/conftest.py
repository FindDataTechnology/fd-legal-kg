"""共享夹具：全离线，不触碰网络 / FalkorDB / law_db。"""

from __future__ import annotations

import pytest

from fd_legal_kg import config

REQUIRED_ENV = {
    'LEXKG_LLM_BASE_URL': 'https://llm-gw.example/v1',
    'LEXKG_LLM_API_KEY': 'sk-test-llm',
    'LEXKG_EMBED_BASE_URL': 'https://embed-gw.example/v1',
    'LEXKG_EMBED_API_KEY': 'sk-test-embed',
    'LEXKG_LAW_DATABASE_URL': 'postgresql://ro:ro@localhost:5432/law_db',
}

ALL_ENV_KEYS = [
    'LEXKG_LLM_BASE_URL',
    'LEXKG_LLM_API_KEY',
    'LEXKG_LLM_MODEL',
    'LEXKG_LLM_STRUCTURED_MODE',
    'LEXKG_EMBED_BASE_URL',
    'LEXKG_EMBED_API_KEY',
    'LEXKG_EMBED_MODEL',
    'LEXKG_EMBED_PROVIDER_ONLY',
    'LEXKG_EMBED_DIM',
    'LEXKG_FALKORDB_URI',
    'LEXKG_FALKORDB_USER',
    'LEXKG_FALKORDB_PASSWORD',
    'LEXKG_LAW_DATABASE_URL',
    'LEXKG_TELEMETRY_OPTOUT',
    'LEXKG_SEMAPHORE_LIMIT',
    'LEXKG_COST_LEDGER_PATH',
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ALL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv('GRAPHITI_TELEMETRY_ENABLED', raising=False)


@pytest.fixture
def settings(monkeypatch) -> config.Settings:
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    return config.from_env()
