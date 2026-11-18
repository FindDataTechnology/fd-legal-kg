"""法律领域本体：7 类实体，作为 graphiti ``add_episode(entity_types=...)`` 的类型约束。

graphiti 的 entity_types 形参是 dict：{"Court": CourtModel, ...}。
live 验证（graphiti-core 0.31.1）：实体模型的字段必须全部 Optional 且带默认值——
属性抽取环节会把 ``{'level': None}`` 这类缺值形状直接喂给 pydantic，
必填字段会让整条 add_episode 崩掉。因此这里所有字段都是 ``str | None = None``。
"""

from __future__ import annotations

# 注意：graphiti EntityNode 自带 name 属性；实体类型模型里不得再声明 name/title 等保留名。
# FalkorDB 实测（图主机, falkordblite 0.10.0 linux）：属性抽取产生的非空 attributes 在边/节点写入
# 时触发 "Property values can only be of primitive types"；空模型（仅 docstring）路径端到端已验证。
# 类型语义完全由 docstring 承载（graphiti 将其注入抽取 prompt）。
from pydantic import BaseModel


class Court(BaseModel):
    """法院：行使审判权的国家机关。属性口径（抽取提示用）：审级=最高人民法院/高级/中级/基层/专门法院；辖区如「上海」。"""


class Judge(BaseModel):
    """法官：案件的具体审理人员（审判长、审判员、代理审判员、书记员归入并注明角色）。"""


class Party(BaseModel):
    """当事人：原告、被告、上诉人、被上诉人、申请人、第三人等诉讼主体（自然人或法人/组织），注明诉讼地位与主体类型。"""


class LawFirm(BaseModel):
    """律所：当事人委托的律师事务所，注明所在地。"""


class Statute(BaseModel):
    """法规：法律、行政法规、司法解释、部门规章等规范文件或其条文载体，注明层级与生效日期。"""


class Case(BaseModel):
    """案例：指导性案例或其他裁判文书对应的案件本体，注明案号与案件类型（民事/刑事/行政/执行/知识产权）。"""


class Charge(BaseModel):
    """罪名：刑事案件中指控/认定的具体罪名，注明对应的刑法条文引用。"""


# graphiti 的 entity_types 形参：dict[str, type[BaseModel]]，键为类型名（live 验证为 dict 而非 list）。
ENTITY_TYPES: dict[str, type[BaseModel]] = {
    'Court': Court,
    'Judge': Judge,
    'Party': Party,
    'LawFirm': LawFirm,
    'Statute': Statute,
    'Case': Case,
    'Charge': Charge,
}
