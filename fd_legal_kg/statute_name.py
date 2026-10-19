"""法规名确定性拆解：把「法规名+条号」的实体名拆回 (法规主体, 条号, 版本)。

为什么需要这一层（试点实测发现，见 reports/2026-10-19-spike1.3-dedup.md）：
抽取环节常把「某法第 N 条」整个当成一个 Statute 实体名，于是
`中华人民共和国刑法` 与 `《中华人民共和国刑法》第313条` 成了两个互不相连的节点，
「哪些案例引用了刑法」这类聚合会漏掉全部条级引用。340 个 Statute 节点里
55 个（16.2%）是这个形态。

为什么用确定性解析而不是交给 LLM：条号是**正则可解析的结构**，
交给模型只会引入幻觉，且这层要在全量数据上跑，成本不可接受。

设计原则：
  - 只拆「法规名 + 条号」；**立法解释类不拆**（其标题里的条号是被解释对象，
    如「关于《刑法》第二百二十八条…的解释」，拆了就丢了指代对象）；
  - 版本标注（（2020年修正））单独抽出，**不并入主体名**——
    同一部法的不同版本在双时态图谱里必须保留区分，粗暴归并会毁掉版本语义；
  - 解析不出来就原样返回，绝不猜测。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 条号：第X条 [之Y] [第Z款] [第W项]
ARTICLE_RE = re.compile(
    r'第[一二三四五六七八九十百千零〇\d]+条'
    r'(?:之[一二三四五六七八九十\d]+)?'
    r'(?:第[一二三四五六七八九十\d]+款)?'
    r'(?:第[一二三四五六七八九十\d]+项)?'
    # 仅允许承前省略的「款/项」并列（第285条第1款、第2款）。
    # 不能吞掉「第N条」——否则「第二条、第九条」会被误判为一条。
    r'(?:\s*[、及和]\s*第[一二三四五六七八九十\d]+[款项]'
    r'(?:第[一二三四五六七八九十\d]+项)?)*'
)

# 版本标注：（2020年修正）/ （2000年修订）
VERSION_RE = re.compile(r'[（(]\s*(\d{4})\s*年(?:修正|修订|修改|修正草案)?\s*[)）]')

# 书名号（含尖括号书名号）
BRACKET_CHARS_RE = re.compile(r'[《》〈〉]')

# 书名号内的内容（《…》/〈…〉）
BRACKETED_RE = re.compile(r'[《〈]([^》〉]*)[》〉]')

# 解释性后缀：条号之后若还跟着这些词，说明这份文件是「关于某法某条的解释」，
# 条号属被引用的对象而非自身（见 parse_statute_name 里的说明）。
INTERPRETATION_SUFFIX_RE = re.compile(r'^\s*[、,，]?\s*(?:的)?(?:解释|规定|批复|答复|决定|通知|意见)')

# 拆掉条号/版本后可能残留在两端的分隔符
EDGE_JUNK_RE = re.compile(r'^[、,，;；\s及和]+|[、,，;；\s及和]+$')


@dataclass(frozen=True)
class ParsedStatuteName:
    """一个法规实体名的拆解结果。

    kind:
      - ``article``        法规名 + 条号（可归并到 statute 主体）
      - ``statute``        纯法规名，无条号
      - ``interpretation`` 立法/司法解释：条号是解释对象，**不拆**
    """

    raw: str
    kind: str
    statute: str
    articles: tuple[str, ...] = ()
    version: str | None = None

    @property
    def is_article(self) -> bool:
        return self.kind == 'article'

    def canonical_statute(self) -> str:
        """法规主体的规范名（去书名号、去条号、去版本）。

        同一部法的不同写法（《中华人民共和国刑法》/ 中华人民共和国刑法）
        在这里收敛成同一个名字，是归并的锚点。
        """
        return self.statute


def _clean(s: str) -> str:
    """去书名号、修掉条号移除后残留的边界分隔符。"""
    s = BRACKET_CHARS_RE.sub('', s)
    s = EDGE_JUNK_RE.sub('', s)
    return s.strip()


def parse_statute_name(name: str) -> ParsedStatuteName:
    """确定性拆解一个法规实体名。不猜测：拆不出来就原样返回。"""
    raw = (name or '').strip()

    version_match = VERSION_RE.search(raw)
    version = version_match.group(1) if version_match else None

    articles = ARTICLE_RE.findall(raw)
    if not articles:
        # 没有任何条号：纯法规名（或解释文件名本身），不拆
        return ParsedStatuteName(
            raw=raw,
            kind='statute',
            statute=_clean(VERSION_RE.sub('', raw)),
            articles=(),
            version=version,
        )

    # 立法/司法解释：这是一份**解释文件**，其名中的条号是被解释的对象，
    # 而不是这份文件自身的条号 → 整体保留，不拆。
    # 判据是「条号后面还跟着解释性后缀」：
    #   「关于《刑法》第二百二十八条…的解释」  → 条号后接 的解释 ⇒ 被解释对象，不拆
    #   「《…的解释》第一百八十八条」          → 条号在末尾、后面无后缀 ⇒ 自身条号，要拆
    # （试过「条号是否落在书名号内」，不成立：真实数据里书名号只包法规名，
    #   条号紧跟在书名号之后。）
    tail = ARTICLE_RE.split(raw)[-1] or ''
    if INTERPRETATION_SUFFIX_RE.search(tail):
        return ParsedStatuteName(
            raw=raw,
            kind='interpretation',
            statute=_clean(VERSION_RE.sub('', raw)),
            articles=(),
            version=version,
        )

    # 主体 = 原串去掉条号段，再去掉版本段
    statute = ARTICLE_RE.sub('', raw)
    statute = VERSION_RE.sub('', statute)
    statute = _clean(statute)
    if not statute:
        # 拆完主体为空（如「第384条」这种孤立条号）：不拆，原样保留。
        # 宁可留着一个不完美的名字，也不产生一个空主体的假归并。
        return ParsedStatuteName(
            raw=raw,
            kind='statute',
            statute=_clean(VERSION_RE.sub('', raw)),
            articles=(),
            version=version,
        )
    return ParsedStatuteName(
        raw=raw,
        kind='article',
        statute=_clean(statute),
        articles=tuple(articles),
        version=version,
    )


@dataclass
class MergeGroup:
    """同一部法规下的归并组：主体名 + 归入它的条级节点。"""

    statute: str
    members: list[ParsedStatuteName] = field(default_factory=list)

    @property
    def article_count(self) -> int:
        return sum(1 for m in self.members if m.is_article)


def group_by_statute(names: list[str]) -> dict[str, MergeGroup]:
    """把一批法规实体名按「法规主体」归并。

    返回 {规范主体名: MergeGroup}。interpretation 类不并入任何组（各自独立），
    因为它们的条号是被解释对象，混进来会污染聚合。
    """
    groups: dict[str, MergeGroup] = {}
    for raw in names:
        parsed = parse_statute_name(raw)
        if parsed.kind == 'interpretation':
            # 解释类自成一个组，key 用它自己的规范名，避免与主体法混淆
            key = f'{parsed.statute}#interpretation'
        else:
            key = parsed.canonical_statute()
        groups.setdefault(key, MergeGroup(statute=key)).members.append(parsed)
    return groups
