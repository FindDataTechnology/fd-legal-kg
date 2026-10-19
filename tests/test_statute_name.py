"""statute_name：法规名确定性拆解。用例全部取自试点图 340 个 Statute 节点的真实形态。"""

from __future__ import annotations

from fd_legal_kg.statute_name import group_by_statute, parse_statute_name


def test_plain_statute_name_is_left_alone():
    p = parse_statute_name('中华人民共和国刑法')
    assert p.kind == 'statute'
    assert p.statute == '中华人民共和国刑法'
    assert p.articles == ()
    assert p.version is None


def test_article_number_is_split_off():
    p = parse_statute_name('中华人民共和国合同法第四百二十四条')
    assert p.kind == 'article'
    assert p.is_article
    assert p.statute == '中华人民共和国合同法'
    assert p.articles == ('第四百二十四条',)


def test_book_title_marks_are_stripped_from_the_subject():
    """书名号有无造成分裂（《…刑法》vs 刑法）——主体名必须收敛成同一个。"""
    marked = parse_statute_name('《中华人民共和国刑法》第313条')
    bare = parse_statute_name('中华人民共和国刑法第303条')
    assert marked.statute == '中华人民共和国刑法'
    assert bare.statute == '中华人民共和国刑法'
    # 与纯法规名节点三方一致，才能归并
    plain = parse_statute_name('中华人民共和国刑法')
    assert marked.statute == plain.statute == bare.statute


def test_paragraph_and_item_levels():
    p = parse_statute_name('《中华人民共和国刑法》第五十条第二款')
    assert p.statute == '中华人民共和国刑法'
    assert p.articles == ('第五十条第二款',)

    p2 = parse_statute_name('《中华人民共和国反垄断法》第十七条第一款第五项')
    assert p2.statute == '中华人民共和国反垄断法'
    assert p2.articles == ('第十七条第一款第五项',)


def test_multiple_articles_in_one_name():
    p = parse_statute_name('《中华人民共和国土地管理法》第二条、第九条')
    assert p.statute == '中华人民共和国土地管理法'
    assert p.articles == ('第二条', '第九条')


def test_ellided_coordination_is_not_left_in_the_subject():
    """「第285条第1款、第2款」这种承前省略并列：不能只拆掉第一段，留下「、第2款」。"""
    p = parse_statute_name('《中华人民共和国刑法》第285条第1款、第2款')
    assert p.statute == '中华人民共和国刑法'
    assert p.articles == ('第285条第1款、第2款',)


def test_zhi_article_form():
    p = parse_statute_name('《中华人民共和国刑法》第253条之一')
    assert p.statute == '中华人民共和国刑法'
    assert p.articles == ('第253条之一',)


def test_version_marker_is_extracted_not_merged():
    """版本标注必须单独抽出：同法不同版本在双时态图谱里要能区分，
    粗暴归并会毁掉版本语义（试点语料里专利法就有 2000/2020 两版）。"""
    p = parse_statute_name('中华人民共和国专利法（2020年修正）第22条第3款')
    assert p.statute == '中华人民共和国专利法'
    assert p.version == '2020'
    assert p.articles == ('第22条第3款',)

    p2000 = parse_statute_name('中华人民共和国专利法（2000年修正）第22条第3款')
    assert p2000.statute == p.statute          # 同一部法
    assert p2000.version == '2000'             # 但版本不同
    assert p2000.version != p.version


def test_legislative_interpretation_is_not_split():
    """立法解释标题里的条号是**被解释对象**，拆了就丢了指代。

    例：「全国人大常委会关于《刑法》第二百二十八条…的解释」——
    第二百二十八条 属于刑法，不属于这个解释文件本身。
    """
    raw = ('全国人民代表大会常务委员会关于《中华人民共和国刑法》'
           '第二百二十八条、第三百四十二条、第四百一十条的解释')
    p = parse_statute_name(raw)
    assert p.kind == 'interpretation'
    assert not p.is_article
    assert p.articles == ()


def test_judicial_interpretation_with_article_still_splits():
    """司法解释名后跟自己的条号时仍要拆（条号在书名号之外，属该解释自身）。"""
    p = parse_statute_name(
        '《最高人民法院关于适用〈中华人民共和国民事诉讼法〉的解释》第一百八十八条'
    )
    assert p.kind == 'article'
    assert p.articles == ('第一百八十八条',)
    # 尖括号书名号也一并去掉
    assert '〈' not in p.statute and '》' not in p.statute


def test_empty_and_odd_input_does_not_raise():
    # raw 存的是 strip 后的值（空白输入收敛为空串，这是有意的）
    for raw, expected in (('', ''), ('   ', ''), ('某无名实体', '某无名实体'),
                          ('第384条', '第384条')):
        p = parse_statute_name(raw)
        assert p.raw == expected
    # 不以条号开头的普通实体名不被误当作条号拆掉
    assert parse_statute_name('第384条').kind == 'statute'


def test_group_by_statute_merges_article_nodes_into_their_statute():
    names = [
        '中华人民共和国刑法',
        '《中华人民共和国刑法》第313条',
        '中华人民共和国刑法第303条',
        '《中华人民共和国刑法》第20条',
        '中华人民共和国专利法（2020年修正）第22条第3款',
        '中华人民共和国专利法（2000年修正）第22条第3款',
        '全国人民代表大会常务委员会关于《中华人民共和国刑法》第二百二十八条的解释',
    ]
    groups = group_by_statute(names)
    # 刑法的三个条级节点并入刑法主体
    assert len(groups['中华人民共和国刑法'].members) == 4
    assert groups['中华人民共和国刑法'].article_count == 3
    # 专利法两版本主体同名，故并入同一组（版本信息保留在 member.version 上）
    assert len(groups['中华人民共和国专利法'].members) == 2
    assert {m.version for m in groups['中华人民共和国专利法'].members} == {'2020', '2000'}
    # 解释类不并入刑法组，自成一组
    assert not any(m.kind == 'interpretation'
                   for m in groups['中华人民共和国刑法'].members)
    assert any(k.endswith('#interpretation') for k in groups)
