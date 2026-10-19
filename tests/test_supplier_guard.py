"""supplier：只读守卫拒绝写语句；连接参数带 read-only；行->Episode 映射正确。"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fd_legal_kg import supplier
from fd_legal_kg.supplier import (
    Episode,
    ReadOnlyViolation,
    amendment_row_to_episode,
    assert_readonly,
    build_connect_kwargs,
    guide_case_row_to_episode,
    load_episodes_jsonl,
    split_articles,
    statute_row_to_episodes,
)


@pytest.mark.parametrize('sql', [
    'SELECT * FROM guide_cases',
    "select 'insert into x' as note, id from laws",   # 字符串字面量不算写
    'SELECT 1 -- insert into t values (1)',           # 行注释不算写
    '/* delete from x */ SELECT version()',
    'WITH t AS (SELECT 1) SELECT * FROM t',
])
def test_readonly_guard_allows_reads(sql):
    assert_readonly(sql)  # 不抛即通过


@pytest.mark.parametrize('sql', [
    "INSERT INTO laws (title) VALUES ('x')",
    'update laws set title = 1',
    'DELETE FROM guide_cases',
    'delete guide_cases',
    'CREATE TABLE tmp (id int)',
    'DROP TABLE laws',
    'ALTER TABLE laws ADD COLUMN x int',
    'TRUNCATE guide_cases',
    'MERGE INTO laws USING t ON 1=1',
    'GRANT ALL ON laws TO public',
    'COPY laws FROM stdin',
    'insert into\nlaws (title) values (\'multi\nline\')',
])
def test_readonly_guard_rejects_writes(sql):
    with pytest.raises(ReadOnlyViolation):
        assert_readonly(sql)


def test_connect_kwargs_carry_readonly_session_option():
    kwargs = build_connect_kwargs('postgresql://ro:ro@h:5432/law_db')
    assert kwargs['options'] == '-c default_transaction_read_only=on'
    assert kwargs['autocommit'] is True
    assert kwargs['conninfo'] == 'postgresql://ro:ro@h:5432/law_db'
    # 断言服务端只读开关真的在 options 里（belt and braces 的第一层）
    assert 'default_transaction_read_only=on' in supplier.PSYCOPG_OPTIONS


def test_guide_case_row_to_episode():
    as_of = datetime(2023, 5, 1, tzinfo=timezone.utc)
    row = ('gc-24', 24, '指导案例24号', as_of, None, '', '', '', '', '', '正文' * 50)
    ep = guide_case_row_to_episode(row)
    assert ep.episode_id == 'gc:gc-24'
    assert ep.kind == 'guide_case'
    assert ep.group_id == 'pilot'
    assert ep.valid_at == as_of
    assert ep.content.startswith('【标题】指导案例24号')
    assert '【正文】' in ep.content
    assert ep.provenance['stable_id'] == 'guide_cases:gc-24'
    assert ep.provenance['doc_title'] == '指导案例24号'


def test_split_articles_spans_are_contiguous():
    text = '第一条 为了惩罚犯罪，根据宪法，制定本法。第二条 一切危害国家安全的行为都是犯罪。'
    arts = split_articles(text)
    assert [a[0] for a in arts] == ['第一条', '第二条']
    assert arts[0][1] == 0 and arts[-1][2] == len(text)


def test_statute_row_to_episodes():
    full_text = '第一条 为了惩罚犯罪，根据宪法，制定本法。\n第二条 一切危害国家安全的行为都是犯罪。'
    eps = statute_row_to_episodes((18726, '中华人民共和国刑法', '2020-12-26', None, '有效', full_text), cap=5)
    assert len(eps) == 2
    ep = eps[1]
    assert ep.episode_id == 'law:18726:第二条'
    assert ep.kind == 'statute_article'
    assert ep.provenance['span']['unit'] == 'laws.content'
    assert ep.valid_at is not None and ep.valid_at.year == 2020
    assert ep.content.startswith('《中华人民共和国刑法》（发布')


def test_statute_row_respects_cap():
    text = '\n'.join(f'第{c}条 ' + '内容' * 30 + '。' for c in '一二三四五')
    eps = statute_row_to_episodes((1, '某法', '2020-01-01', None, '有效', text), cap=2)
    assert len(eps) == 2


def test_amendment_row_to_episode():
    ep = amendment_row_to_episode((69878, '中华人民共和国刑法修正案', '1999-12-25', '正文' * 40))
    assert ep.kind == 'statute_amendment'
    assert ep.episode_id == 'law:69878:full'
    assert ep.valid_at is not None and ep.valid_at.year == 1999


def test_episode_validates_kind():
    with pytest.raises(ValueError):
        Episode(episode_id='x', kind='tweet', content='c')
    ep = Episode(episode_id='x', kind='statute_amendment', content='c', provenance={})
    assert ep.provenance['source_db'] == 'law_db'  # 缺省强制补上


def test_load_episodes_jsonl(tmp_path):
    p = tmp_path / 'eps.jsonl'
    p.write_text(
        '{"episode_id":"gc:x","kind":"guide_case","content":"正文","valid_at":"2012-01-11 00:00:00+08:00",'
        '"provenance":{"source_db":"law_db","stable_id":"guide_cases:x","as_of":"2012-01-11 00:00:00+08:00","doc_title":"t","span":null}}\n',
        encoding='utf-8',
    )
    eps = load_episodes_jsonl(str(p))
    assert len(eps) == 1 and eps[0].episode_id == 'gc:x'
    assert eps[0].valid_at is not None and eps[0].valid_at.year == 2012


def test_sql_constants_are_readonly():
    for sql in (supplier.GUIDE_CASE_SQL, supplier.STATUTE_SQL, supplier.AMENDMENT_SQL, supplier.INSPECT_SQL):
        assert_readonly(sql)


def test_corpus_index_exposes_the_three_provenance_fields():
    """语料索引必须给出 source_db / stable_id / as_of（+ span）供查询面回指原文。"""
    from datetime import datetime, timezone

    from fd_legal_kg.supplier import Episode, corpus_index_for

    eps = [
        Episode(
            episode_id='law:18726:第一条',
            kind='statute_article',
            content='第一条 为了惩罚犯罪……',
            provenance={
                'source_db': 'law_db',
                'stable_id': 'laws:18726',
                'as_of': datetime(2020, 12, 26, tzinfo=timezone.utc),
                'doc_title': '中华人民共和国刑法',
                'span': {'char_start': 1456, 'char_end': 1504},
            },
        )
    ]
    idx = corpus_index_for(eps)
    entry = idx['law:18726:第一条']
    assert entry['source_db'] == 'law_db'
    assert entry['stable_id'] == 'laws:18726'
    assert entry['as_of'].startswith('2020-12-26')
    assert entry['span'] == {'char_start': 1456, 'char_end': 1504}
    assert entry['doc_title'] == '中华人民共和国刑法'
