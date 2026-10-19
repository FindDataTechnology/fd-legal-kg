"""fd-legal-kg：识律·法律时序知识图谱试点（pilot sidecar）。

核心模块：
  config    —— 环境变量驱动的设置（显式失败，绝不静默降级）
  ontology  —— 7 类法律实体 Pydantic 模型（Court/Judge/Party/LawFirm/Statute/Case/Charge）
  prompts   —— 中文法律语料调优的抽取指令（实体类型定义 + few-shot + JSON 输出要求）
  gateway   —— OpenAI 兼容网关的 LLM/Embedder 客户端构造（结构化模式映射、provider 钉扎）
  supplier  —— law_db 只读供应（指导性案例全文 / 法规条文级切片，零写入保证）
  pipeline  —— Graphiti 实例组装 + episode 摄取（group_id='pilot'，reference_time=as_of）
  cost      —— 按 episode×operation 的 token 成本台账（JSONL）
  query     —— 试点查询面（实体/关键词检索、多跳图游走、as-of 时间过滤）
  evalrunner —— 双口径评测跑分器（基线 fd-legal-search-mcp vs KG，逐题取证落对照表）
"""

import os as _os

# graphiti 的遥测开关必须在 import graphiti 之前落进 os.environ（本包的子模块
# gateway/pipeline 才 import graphiti，包 __init__ 先于它们执行）。
# LEXKG_TELEMETRY_OPTOUT 默认 true；显式回填 false/0/no/off 时不关闭。
# 用户已手动设置 GRAPHITI_TELEMETRY_ENABLED 时尊重之（setdefault）。
if _os.environ.get('LEXKG_TELEMETRY_OPTOUT', 'true').strip().lower() not in (
    '0', 'false', 'no', 'off',
):
    _os.environ.setdefault('GRAPHITI_TELEMETRY_ENABLED', 'false')

__version__ = "0.1.0"

