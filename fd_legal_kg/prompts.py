"""中文法律语料调优的抽取指令。

graphiti 0.31.1 没有公开的「整套替换 system prompt」稳定入口，但 ``add_episode`` 提供
``custom_extraction_instructions`` 形参，会被拼进实体/关系抽取 prompt。
我们选择这个官方注入点，而不是覆写 prompt_library（那是内部 API，升级易碎）。
"""

from __future__ import annotations

ENTITY_TYPE_DEFINITIONS = """
【实体类型定义】（抽取时严格按以下 7 类归类，不可归类的实体直接丢弃）
1. Court 法院 —— name（法院全称）、level（审级）、region（辖区）
2. Judge 法官 —— name（姓名）、role（审判长/审判员等）
3. Party 当事人 —— name（姓名或组织全称）、standing（原告诉讼地位）、person_type
4. LawFirm 律所 —— name（律所全称）、location
5. Statute 法规 —— title（法规名）、statute_type（法律/司法解释等）、effective_date
6. Case 案例 —— title（案件名）、case_number（案号）、case_type（民事/刑事/行政）
7. Charge 罪名 —— name（罪名全称）、statute_ref（对应刑法条文）
"""

FEW_SHOT_EXAMPLE = """
【few-shot 示例】
输入片段：
  「指导案例24号：荣宝英诉王阳、永诚财产保险股份有限公司江阴支公司机动车交通事故
    责任纠纷案。江苏省无锡市中级人民法院终审……依据《中华人民共和国侵权责任法》
    第二十六条……」
期望抽取（示意）：
  实体：Case(title=荣宝英诉王阳、永诚财保江阴支公司机动车交通事故责任纠纷案)；
        Court(name=江苏省无锡市中级人民法院, level=中级, region=江苏无锡)；
        Party(name=荣宝英, standing=原告)；Party(name=王阳, standing=被告)；
        Party(name=永诚财产保险股份有限公司江阴支公司, standing=被告)；
        Statute(title=中华人民共和国侵权责任法, statute_type=法律)
  事实（边）：荣宝英 ——[案件当事人(原告)]→ 该案；该案 ——[审理法院]→ 江苏省无锡市中级人民法院；
        该案 ——[适用法律]→ 侵权责任法（事实中带上条文号「第二十六条」）
"""

TEMPORAL_INSTRUCTIONS = """
【时间与效力要求】
- 每条事实尽量给出 valid_at（事实开始为真的时间）与 invalid_at（事实不再为真的时间）；
  被后续裁判/修订推翻的旧事实不是删除，而是标记 invalid_at。
- 条文类事实（如罪名对应条文）注明所属法规与条文号，便于跨案例归并。
- 语言：实体名、事实一律使用中文原文，不要翻译。
"""

# OpenAI json_object 模式硬性要求：messages 里必须出现 "JSON" 这个词，否则网关直接报错。
# 该附注仅在 LEXKG_LLM_STRUCTURED_MODE=json_object 时拼入。
JSON_OUTPUT_INSTRUCTIONS = """
【输出格式】
只输出一个合法的 JSON 对象，不要输出任何解释性文字、markdown 代码围栏或注释；
对象结构与系统提示中给出的 schema 完全一致。这是一条 JSON 输出任务。
"""


def build_custom_extraction_instructions(structured_mode: str = 'native') -> str:
    """组装拼入 graphiti 抽取 prompt 的自定义指令。

    structured_mode 为 json_object（以及提示词兜底的 hint）时附加 JSON 输出附注，
    保证消息里出现 "JSON" 字样以满足网关 json_object 模式的硬性要求。
    """
    parts = [
        '你正在为中文法律时序知识图谱做信息抽取（指导性案例全文、法规条文）。',
        ENTITY_TYPE_DEFINITIONS.strip(),
        FEW_SHOT_EXAMPLE.strip(),
        TEMPORAL_INSTRUCTIONS.strip(),
    ]
    if structured_mode in ('json_object', 'hint'):
        parts.append(JSON_OUTPUT_INSTRUCTIONS.strip())
    return '\n'.join(parts)
