"""evalrunner：题集加载校验、双口径取证、逐题错误隔离、JSONL 往返与汇总计数。"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from fd_legal_kg.evalrunner import DualEvalRunner, EvalQuestion, load_questions


def _questions() -> list[EvalQuestion]:
    return [
        EvalQuestion(id='T1', kind='temporal', query='q-正常', expectation='e1',
                     verdict_rule='r1'),
        EvalQuestion(id='T2', kind='temporal', query='q-基线炸', expectation='e2',
                     verdict_rule='r2'),
        EvalQuestion(id='M1', kind='multihop', query='q-多跳', expectation='e3',
                     verdict_rule='r3'),
    ]


class FakeBaseline:
    """按查询子串决定成败的 fake 基线客户端。"""

    def __init__(self, fail_marker: str = ''):
        self.fail_marker = fail_marker
        self.calls: list[tuple[str, str]] = []

    async def call(self, tool, arguments):
        # 参数名依工具而异（query / title_query），统一取唯一值即可
        query = next(iter(arguments.values()))
        self.calls.append((tool, query))
        if self.fail_marker and self.fail_marker in query:
            raise RuntimeError('mcp down')
        return ['{"hits": [{"title": "民法典"}]}']


def _make_kg(fail_marker: str = ''):
    seen: list[tuple[str, str]] = []

    async def _fn(query, question):
        seen.append((query, question.id))
        if fail_marker and fail_marker in query:
            raise ValueError('kg 车道故障')
        return [{'fact': 'KG 事实', 'valid_at': '2021-01-01T00:00:00+00:00',
                 'provenance': [{'stable_id': 'law-1#第二条'}]}]

    return _fn, seen


def test_load_questions_ok_and_ignores_extra_fields(tmp_path):
    path = tmp_path / 'questions.json'
    path.write_text(json.dumps({
        '_comment': '金标题草稿',
        'questions': [
            {'id': 'T1', 'kind': 'temporal', 'query': '查询一', 'expectation': '口径',
             'verdict_rule': '判定', 'baseline_note': '草稿自带字段，应被忽略'},
            {'id': 'M1', 'kind': 'multihop', 'query': '查询二', 'expectation': '口径',
             'verdict_rule': '判定'},
        ],
    }, ensure_ascii=False), encoding='utf-8')
    questions = load_questions(path)
    assert [q.id for q in questions] == ['T1', 'M1']
    assert questions[0].kind == 'temporal'
    assert not hasattr(questions[0], 'baseline_note')


@pytest.mark.parametrize('bad', ['trivial', '', None])
def test_load_questions_rejects_bad_kind(tmp_path, bad):
    path = tmp_path / 'questions.json'
    path.write_text(json.dumps({
        'questions': [{'id': 'X', 'kind': bad, 'query': 'q', 'expectation': 'e',
                       'verdict_rule': 'r'}],
    }), encoding='utf-8')
    with pytest.raises(ValidationError):
        load_questions(path)


def test_load_questions_rejects_bad_top_level_shape(tmp_path):
    path = tmp_path / 'questions.json'
    path.write_text('{"items": []}', encoding='utf-8')
    with pytest.raises(ValueError, match='questions'):
        load_questions(path)


async def test_run_records_both_calibers_and_tool_mapping():
    baseline = FakeBaseline()
    kg_fn, seen = _make_kg()
    report = await DualEvalRunner(baseline, kg_fn).run(_questions())
    assert [r.question.id for r in report.results] == ['T1', 'T2', 'M1']
    # 基线工具映射：temporal 只调混合检索；multihop 叠加指导案例检索
    assert baseline.calls == [
        ('law_semantic_search', 'q-正常'),
        ('law_semantic_search', 'q-基线炸'),
        ('law_semantic_search', 'q-多跳'),
        ('guide_cases_search', 'q-多跳'),
    ]
    assert seen == [('q-正常', 'T1'), ('q-基线炸', 'T2'), ('q-多跳', 'M1')]
    for result in report.results:
        hit = '{"hits": [{"title": "民法典"}]}'
        # multihop 叠加了 guide_cases_search，基线证据是两次调用的拼接
        expected = [hit, hit] if result.question.kind == 'multihop' else [hit]
        assert result.baseline_evidence == expected
        assert result.kg_evidence[0]['fact'] == 'KG 事实'
        assert result.verdict is None
        assert result.ts is not None


async def test_baseline_error_isolated_per_question():
    kg_fn, _ = _make_kg()
    report = await DualEvalRunner(FakeBaseline(fail_marker='基线炸'), kg_fn).run(_questions())
    t2 = report.results[1]
    # 故障车道以 {"error": ...} 落证；另一车道与后续题目不受影响
    assert len(t2.baseline_evidence) == 1
    payload = json.loads(t2.baseline_evidence[0])
    # 错误带工具名：一次跑分含多个基线工具，不带名就分不清是哪个挂
    assert payload['error'].startswith('law_semantic_search: RuntimeError: mcp down')
    assert t2.kg_evidence[0]['fact'] == 'KG 事实'
    assert report.results[0].baseline_evidence[0].startswith('{"hits"')
    assert report.results[2].baseline_evidence[0].startswith('{"hits"')


async def test_kg_error_isolated_per_question():
    kg_fn, _ = _make_kg(fail_marker='多跳')
    report = await DualEvalRunner(FakeBaseline(), kg_fn).run(_questions())
    m1 = report.results[2]
    assert m1.kg_evidence == [{'error': 'ValueError: kg 车道故障'}]
    assert m1.baseline_evidence[0].startswith('{"hits"')
    assert report.results[0].kg_evidence[0]['fact'] == 'KG 事实'


async def test_run_single_caliber_modes():
    baseline = FakeBaseline()
    kg_fn, seen = _make_kg()
    report = await DualEvalRunner(baseline, kg_fn).run(_questions(), baseline=False, kg=True)
    assert baseline.calls == [] and seen
    assert all(r.baseline_evidence == [] for r in report.results)

    baseline = FakeBaseline()
    kg_fn, seen = _make_kg()
    report = await DualEvalRunner(baseline, kg_fn).run(_questions(), baseline=True, kg=False)
    assert seen == []
    assert all(r.kg_evidence == [] for r in report.results)


async def test_run_requires_kg_fn_when_kg_enabled():
    runner = DualEvalRunner(FakeBaseline(), kg_query_fn=None)
    with pytest.raises(ValueError, match='kg_query_fn'):
        await runner.run(_questions()[:1])
    # 关闭 kg 车道时允许缺省
    report = await runner.run(_questions()[:1], kg=False)
    assert report.results[0].kg_evidence == []


async def test_jsonl_round_trip(tmp_path):
    report = await DualEvalRunner(FakeBaseline(), _make_kg()[0]).run(_questions())
    report.results[0].verdict = 'KG 命中 / 基线未命中'
    out = tmp_path / 'report.jsonl'
    report.to_jsonl(out)
    lines = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines()]
    assert [line['id'] for line in lines] == ['T1', 'T2', 'M1']
    first = lines[0]
    assert first['kind'] == 'temporal' and first['verdict_rule'] == 'r1'
    assert first['query'] == 'q-正常' and first['expectation'] == 'e1'
    assert first['verdict'] == 'KG 命中 / 基线未命中'
    assert first['baseline_evidence'] == ['{"hits": [{"title": "民法典"}]}']
    assert first['kg_evidence'][0]['valid_at'] == '2021-01-01T00:00:00+00:00'
    assert 'ts' in first and lines[1]['verdict'] is None


async def test_summary_counts():
    # T1 双中 / T2 基线炸 / M1 KG 炸
    report = await DualEvalRunner(
        FakeBaseline(fail_marker='基线炸'), _make_kg(fail_marker='多跳')[0]
    ).run(_questions())
    summary = report.summary()
    assert summary['total'] == 3
    assert summary['baseline_hits'] == 2 and summary['kg_hits'] == 2
    assert summary['by_kind']['temporal'] == {'total': 2, 'baseline_hits': 1, 'kg_hits': 2}
    assert summary['by_kind']['multihop'] == {'total': 1, 'baseline_hits': 1, 'kg_hits': 0}


async def test_baseline_passes_each_tool_its_own_argument_name():
    """实测 4.1.1 签名：law_semantic_search 收 query，guide_cases_search 收 title_query。

    统一传 query 会让后者报 missing_argument，把「参数传错」伪装成「基线答不出」。
    """
    seen: list[tuple[str, dict]] = []

    class Recording:
        async def call(self, tool, arguments):
            seen.append((tool, dict(arguments)))
            return ['{"hits": []}']

    kg_fn, _ = _make_kg()
    await DualEvalRunner(Recording(), kg_fn).run(_questions()[2:])  # 只有 multihop
    assert seen == [
        ('law_semantic_search', {'query': 'q-多跳'}),
        ('guide_cases_search', {'title_query': 'q-多跳'}),
    ]
