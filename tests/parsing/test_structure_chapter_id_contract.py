"""S2/S4 承重契约锁(CI,非运行时门):chapter_id = 'ch'+section_id 确定性
派生,且与 Chunk.chapter_id / 后端 _ck(routes_domain:216-220)同形;S4
content payload 解析锁形与 force 语义。锁规则防未来误改 + 锁 S2/S4 一致
(S4 写 Chunk.chapter_id 必须同 S2 派生)。依
memory:contract-lock-test-vs-runtime-gate —— 派生无运行时方差,故契约锁
而非运行时门。纯函数,无 API/DB。
"""
import re

import pytest

from sla.parsing.content_extract import parse_content_payload
from sla.parsing.structure_extract import derive_chapter_id, parse_toc_payload

# Chunk.chapter_id / detect_sections 既有形 + 后端 _ck(c[2:].split('.') 全 int)
_CH_FORM = re.compile(r"^ch\d+(\.\d+)*$")


def _ck_parses(cid: str) -> bool:
    """镜像 routes_domain._ck:c[2:].split('.') 全 int 才可正确排序。"""
    try:
        [int(x) for x in cid[2:].split(".")]
        return True
    except Exception:
        return False


@pytest.mark.parametrize("sid,expect", [
    ("1", "ch1"), ("1.3", "ch1.3"), ("6.1.1", "ch6.1.1"),
    ("12.10", "ch12.10"), ("  2.4  ", "ch2.4"),
])
def test_derive_keeps_levels_and_matches_chunk_form(sid, expect):
    cid = derive_chapter_id(sid)
    assert cid == expect
    assert _CH_FORM.match(cid), f"{cid!r} 不匹配 Chunk.chapter_id 形"
    assert _ck_parses(cid), f"{cid!r} 后端 _ck 不可解析 → S3 排序坏"


@pytest.mark.parametrize("sid", [
    "", "   ", "A.1", "前言", "1.", ".1", "1..2", "3-1", "ch1",
])
def test_derive_unkeyed_returns_none(sid):
    # 非点分整数 → None(调用方跳过+计数,不让坏键污染 join 脊)
    assert derive_chapter_id(sid) is None


def test_derive_is_pure_ch_prefix_rule():
    # 承重:就是 'ch'+strip(section_id),无其它变换 —— S4 必须复用此派生
    assert derive_chapter_id("2.4") == "ch2.4"
    assert derive_chapter_id("10") == "ch10"


@pytest.mark.parametrize("payload,pages,force", [
    ("toc:7-10", [7, 8, 9, 10], False),
    ("toc:7,8,9", [7, 8, 9], False),
    ("toc:7-10,15", [7, 8, 9, 10, 15], False),
    ("toc:7-10!force", [7, 8, 9, 10], True),
    ("toc:5", [5], False),
    ("toc:9-9", [9], False),
    ("toc: 7 , 8 ", [7, 8], False),
])
def test_parse_toc_payload_ok(payload, pages, force):
    assert parse_toc_payload(payload) == (pages, force)


@pytest.mark.parametrize("bad", [
    "7-10", "toc:", "toc:0", "toc:abc", "toc:10-3", "toc:-5", "", "  ",
])
def test_parse_toc_payload_rejects(bad):
    with pytest.raises(ValueError):
        parse_toc_payload(bad)


def test_bite_missing_ch_prefix_fails_form():
    # bite:若有人把派生改成漏 'ch' 前缀 / 翻译 section_id,本测应红
    assert not _CH_FORM.match("1.3")
    assert _CH_FORM.match("ch1.3")


# ---------- S4 parse_content_payload(锁形 + force 语义)----------

@pytest.mark.parametrize("payload,ch,pages,force", [
    ("ch1.2|p=50-65", "ch1.2",
     [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65], False),
    ("ch1.2|p=50-65!force", "ch1.2",
     [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65], True),
    ("ch6.1.1|p=200,202-205", "ch6.1.1", [200, 202, 203, 204, 205], False),
    ("ch22|p=440", "ch22", [440], False),
    ("ch1|p=3-4", "ch1", [3, 4], False),
])
def test_parse_content_payload_ok(payload, ch, pages, force):
    assert parse_content_payload(payload) == (ch, pages, force)


@pytest.mark.parametrize("bad", [
    "ch1.2",                # 无 |p=
    "|p=50",                # 无 chapter
    "ch1.2|p=",             # 空 pages
    "ch1.2|p=abc",          # 非整数
    "ch1.2|p=10-3",         # 反序
    "ch1.2|p=0",            # 0
    "1.2|p=50",             # 无 ch 前缀
    "",
    "ch|p=50",              # ch 无数字
])
def test_parse_content_payload_rejects(bad):
    with pytest.raises(ValueError):
        parse_content_payload(bad)


def test_parse_content_payload_preserves_chapter_id_form():
    # 承重:解出的 chapter_id 形与 derive_chapter_id 输出形同(S4 用之写
    # Chunk.chapter_id,必须 _CH_FORM 匹配 + _ck 可解析)
    for p in ("ch1|p=3", "ch1.2|p=50", "ch6.1.1|p=200"):
        ch, _, _ = parse_content_payload(p)
        assert _CH_FORM.match(ch), f"{ch!r} 不匹配 Chunk.chapter_id 形"
        assert _ck_parses(ch), f"{ch!r} 后端 _ck 不可解析"
