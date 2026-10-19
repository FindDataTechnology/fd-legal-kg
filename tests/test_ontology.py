"""ontology：7 类实体齐全、schema 可序列化、描述到位、字段全 Optional。"""

from __future__ import annotations

from pydantic import BaseModel

from fd_legal_kg.ontology import ENTITY_TYPES

EXPECTED = {
    'Court': '法院',
    'Judge': '法官',
    'Party': '当事人',
    'LawFirm': '律所',
    'Statute': '法规',
    'Case': '案例',
    'Charge': '罪名',
}


def test_seven_entity_types_present():
    assert set(ENTITY_TYPES) == set(EXPECTED)
    for name, cls in ENTITY_TYPES.items():
        assert issubclass(cls, BaseModel), name


def test_docstrings_are_chinese_legal_definitions():
    for name, keyword in EXPECTED.items():
        doc = ENTITY_TYPES[name].__doc__ or ''
        assert keyword in doc, f'{name} docstring should mention {keyword}'


def test_schema_serializable_empty_object():
    for name, cls in ENTITY_TYPES.items():
        schema = cls.model_json_schema()
        assert schema.get('type') == 'object'
        # 空模型（无属性）：类型语义在 docstring，graphiti 会将 docstring 注入抽取 prompt
        assert schema.get('properties') in (None, {}), name


def test_models_are_attributeless_for_falkordb():
    """live 验证（zihan FalkorDB）：非空 attributes 写边触发 primitive-type 限制，
    故试点七类模型一律无字段、类型语义由 docstring 承载。"""
    for name, cls in ENTITY_TYPES.items():
        assert cls.model_fields == {}, name
        assert cls.__doc__ and len(cls.__doc__) > 10, name


def test_schema_has_no_required_fields():
    for name, cls in ENTITY_TYPES.items():
        assert cls.model_json_schema().get('required') is None, name


def test_statute_docstring_carries_temporal_semantics():
    assert '生效日期' in ENTITY_TYPES['Statute'].__doc__
