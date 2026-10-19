# fd-legal-kg 识律·法律时序知识图谱（试点 sidecar）

在 Zep [Graphiti](https://help.getzep.com/graphiti)（`graphiti-core==0.31.1`）之上构建的法律领域时序知识图谱试点：
以「episode」为单位摄取 law_db（只读）中的指导性案例全文与法规条文级切片，
经 OpenAI 兼容 LLM 网关抽取法律实体/事实，落到 FalkorDB，并带双时间效力
（`valid_at`/`invalid_at`；被推翻的事实标记失效而非删除）。

## 架构

```
                 ┌────────────────────────── finddata ──────────────────────────┐
                 │                                                               │
  law_db (PG)    │  fd-legal-kg                                                 │
 ┌────────────┐  │  ┌────────────┐   episodes   ┌──────────────────────────┐    │
 │ guide_cases│─▶│  │ supplier   │────────────▶│ pipeline (Graphiti)      │    │
 │ laws/      │  │  │ (READ-ONLY │              │  FalkorDriver (FalkorDB) │    │
 │ law_articles│ │  │  guard×2)  │              │  entity_types = 7 类本体  │    │
 └────────────┘  │  └────────────┘              │  group_id = 'pilot'      │    │
                 │                              └───────┬──────────┬───────┘    │
                 │                       LLM 抽取/去重/失效 │          │ embedding │
                 │                              ▼          ▼            │
                 │                     LLM 网关 (OpenAI 兼容)   嵌入网关(钉扎provider)│
                 │                              │                         │
                 │                        cost ledger (JSONL, per-episode │
                 │                        × operation token 台账)◀────────│
                 │                                                        │
                 │  query: 实体/关键词检索 · 多跳游走 · as-of 时间过滤        │
                 └────────────────────────────────────────────────────────┘
```

## 快速开始

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q                      # 65 个测试，全离线（不需要 FalkorDB/网关/law_db）
```

摄取（live，需要 FalkorDB + 网关 + law_db 全部就位）：

```python
import asyncio
from fd_legal_kg import config, pipeline, supplier
from fd_legal_kg.cost import CostLedger
from fd_legal_kg.query import LegalKGQuery, GraphitiQueryBackend

settings = config.from_env()          # 缺必填项会显式报错，绝不静默降级
settings.apply_runtime_env()          # GRAPHITI_TELEMETRY_ENABLED=false（默认 opt-out）

ledger = CostLedger(settings.cost_ledger_path)
llm, embedder, cross = pipeline.build_cost_wired_clients(settings, ledger)
graphiti = pipeline.build_graphiti(settings, llm, embedder, cross)

conn = asyncio.run(supplier.connect(settings))
episodes = asyncio.run(conn.fetch_episodes(limit=10))
asyncio.run(pipeline.ingest_episodes(graphiti, episodes, ledger))

kg = LegalKGQuery(GraphitiQueryBackend(graphiti))
facts = asyncio.run(kg.search_facts("故意伤害罪", as_of=None))   # 每条带 provenance + 效力窗口
```

首次连真实 law_db 前必须核对表结构（供应商 SQL 常量块中的列名是按猜测写的）：

```bash
python -m fd_legal_kg.supplier --inspect   # 打印 laws/guide_cases/law_articles 的列
```

## 环境变量

| 变量 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `LEXKG_LLM_BASE_URL` | 是 | — | LLM 网关（OpenAI 兼容）base URL |
| `LEXKG_LLM_API_KEY` | 是 | — | LLM 网关密钥 |
| `LEXKG_LLM_MODEL` | 否 | `deepseek-v4.1-flash` | 抽取模型（live 验证过的名字，无厂商前缀；glm-5.3-flash 无试点账号池） |
| `LEXKG_LLM_STRUCTURED_MODE` | 否 | `native` | `native`/`json_object`/`hint`，映射到 graphiti structured_output_mode |
| `LEXKG_EMBED_BASE_URL` | 是 | — | 嵌入网关 base URL |
| `LEXKG_EMBED_API_KEY` | 是 | — | 嵌入网关密钥 |
| `LEXKG_EMBED_MODEL` | 否 | `Qwen/Qwen3-Embedding-8B` | 嵌入模型 |
| `LEXKG_EMBED_PROVIDER_ONLY` | 否 | `SiliconFlow` | 嵌入 provider 钉扎；每个 /embeddings 请求带 `extra_body {"provider":{"only":[…]}}`（防网关轮换上游导致向量漂移）。留空回落默认值 |
| `LEXKG_EMBED_DIM` | 否 | `1024` | 嵌入维度 |
| `LEXKG_FALKORDB_URI` | 否 | `bolt://127.0.0.1:6889` | FalkorDB 地址（解析成 host/port） |
| `LEXKG_FALKORDB_USER` / `_PASSWORD` | 否 | — | FalkorDB 认证 |
| `LEXKG_LAW_DATABASE_URL` | 供应商用 | — | law_db 只读 PG DSN |
| `LEXKG_TELEMETRY_OPTOUT` | 否 | `true` | 置 `GRAPHITI_TELEMETRY_ENABLED=false` |
| `LEXKG_SEMAPHORE_LIMIT` | 否 | `5` | 映射 Graphiti `max_coroutines` |
| `LEXKG_COST_LEDGER_PATH` | 否 | `data/cost_ledger.jsonl` | token 成本台账 |

## 命令

| 命令 | 说明 |
|---|---|
| `pytest -q` | 全离线测试 |
| `python -m fd_legal_kg.supplier --inspect` | live 核对 law_db 表结构 |
| `python -m fd_legal_kg.supplier` | 打印帮助 |

## API reality notes（README≠code 核对记录）

以下是对**已安装**的 `graphiti-core==0.31.1`（site-packages 实读）核对出的实际 API，
以及与常见文档/预期不符、需要适配的点：

```
OpenAIGenericClient.__init__(config=None, cache=False, client=None,
                             max_tokens=16384,
                             structured_output_mode: Literal['json_schema','json_object'] = 'json_schema')
Graphiti.__init__(graph_driver, llm_client, embedder, cross_encoder, max_coroutines, ...)
Graphiti.add_episode(name, episode_body, source_description, reference_time, *,
                     source=EpisodeType.message, group_id=None, uuid=None,
                     entity_types: dict[str, type[BaseModel]] | None = None,
                     excluded_entity_types=None, previous_episode_uuids=None,
                     edge_types=None, edge_type_map=None,
                     custom_extraction_instructions=None, saga=None, ...)
FalkorDriver.__init__(host='localhost', port=6379, username=None, password=None,
                      falkor_db=None, database='default_db')
EntityEdge 字段: fact/episodes/valid_at/invalid_at/expired_at/reference_time
EpisodeType: message | json | text | fact_triple
```

适配偏差（均已按实际 API 处理）：

1. **structured_output_mode 没有 `native`/`hint`**。只有 `json_schema`/`json_object`。
   映射：`native→json_schema`；`json_object→json_object`；`hint→json_object` +
   `HintModeClient` 子类把 `response_format` 降级为 `{'type':'text'}`（schema 仅在
   prompt 中提示，无 API 级 JSON 约束）。
2. **FalkorDriver 不吃 URI**，只有 host/port 构造参数（`config.parse_falkor_uri` 负责
   解析 `LEXKG_FALKORDB_URI`）；且构造函数内部立刻连库做 `Is_Cluster` 探测——离线环境
   连构造都会失败，因此测试从不构造它，`build_graphiti` 把连接失败包成显式
   `ConfigError`。
3. **`openai>=3.34` 依赖的是 `httpx2`（另一个 import 名）**，而 graphiti 内部
   `import httpx`——`pip install graphiti-core` 装完 import 即炸。`httpx` 已在
   pyproject 显式声明。
4. **FalkorDriver import 期就需要 `falkordb` 包**（graphiti 的可选 extra），已显式
   声明 `falkordb>=1.8.0`。
5. **Graphiti 构造函数在 cross_encoder 缺省时会急切 new `OpenAIRerankerClient()`**
   （读 `OPENAI_API_KEY` 的 OpenAI 官方客户端）。`pipeline.build_graphiti` 永远显式
   传入走网关的 reranker（`gateway.build_cross_encoder`，chat-completions 兼容）。
6. **0.31.1 没有 `Graphiti.initialize()`**（旧版教程里的建索引入口已不存在），不要调用。
7. **OpenAIEmbedder 不支持 extra_body**，无法钉扎 provider——`gateway.ProviderPinnedEmbedder`
   子类补上；单测断言钉扎时 extra_body 存在、未配置时不传。
8. **graphiti 的 `_generate_response` 会丢弃响应里的 usage**，token 计量只能在
   AsyncOpenAI 形状的客户端层截获（`cost.RecordingAsyncOpenAI`）。
9. 自定义抽取 prompt 走官方注入点 `add_episode(custom_extraction_instructions=...)`，
   不覆写内部 prompt_library（那是无稳定承诺的内部 API）。

## Live 验证记录（graphiti-core 0.31.1 + FalkorDriver + deepseek-v4.1-flash + SiliconFlow 嵌入，端到端跑通）

以下事实来自主链路的真实集成测试，均已落进代码：

1. `add_episode(entity_types=...)` 是 **dict**（`{"Court": CourtModel, ...}`），不是 list
   ——`ontology.ENTITY_TYPES` 即此形状。
2. **自定义实体模型的字段必须全部 Optional 且带默认值**（或只留 docstring 的空模型）：
   属性抽取会把 `{'properties': {'level': None}}` 形状直接喂给 pydantic，必填字段会让
   add_episode 崩掉。本包 7 类本体全部 `str | None = None`，并有
   `test_schema_has_no_required_fields` 守着。
3. `source` 必须是 `from graphiti_core.nodes import EpisodeType` 的
   **`EpisodeType.text` 枚举**（不是字符串）；`reference_time` 必须是 **datetime 对象**。
4. **cross_encoder 必须显式传入**（见上第 5 条）——缺省会急切构造要
   `OPENAI_API_KEY` 的官方 reranker。
5. **deepseek 在 json_object 模式会把输出多包一层 `{"properties": {...}}` 信封**（schema
   误当输出结构）。graphiti 只剥 ``` 围栏不解信封——`gateway.GatewayLLMClient` 重写
   `_generate_response`：唯一键恰为 `properties` 且值为 dict 时解包（单测覆盖解包/
   透传/歧义载荷三种情形）。这一个子类让整条管线 live 跑通。
6. **`GRAPHITI_TELEMETRY_ENABLED=false` 必须在 import graphiti 之前落 os.environ**——
   由 `fd_legal_kg/__init__.py` 在包加载时（早于 gateway/pipeline import graphiti）按
   `LEXKG_TELEMETRY_OPTOUT`（默认 true）设置，尊重用户已手设的值。
7. 读取面：`add_episode` 返回 `AddEpisodeResults(episode, episodic_edges, nodes, edges,
   communities, community_edges)`；**没有 `get_entities_and_edges` 方法**，用
   `.search()` / `.search_()` / `get_nodes_and_edges_by_episode()`（`query.py` 的
   GraphitiQueryBackend 走的就是这些）。
8. live 验证过的构造形状：
   `OpenAIEmbedderConfig(api_key=…, base_url='https://api.siliconflow.cn/v1',
   embedding_model='Qwen/Qwen3-Embedding-8B', embedding_dim=1024)` + embedder 子类
   （另加 provider 钉扎 extra_body）；LLM
   `OpenAIGenericClient(config=LLMConfig(api_key=…, base_url=…, model='deepseek-v4.1-flash'),
   structured_output_mode='json_object', max_tokens=16384)`。

## 版本钉扎

`graphiti-core==0.31.1`（连带解析出的关键版本：openai 3.34.0、httpx 0.28.1、
FalkorDB 1.8.0、redis 8.1.0、psycopg 3.3.6、pydantic 2.14.1）。
