"""section-detection gate 回归(O1/O2)。纯 stdlib+pytest,无网络无 LLM。

反自欺纪律编入回归:真 PDF fixture + 显式断言 "B1 在真实污染态不 fire"
+ "clean ch1/ch2 不被误杀"(§29 幸存者偏差防护)。
"""
import pytest

from sla.parsing.pdf import extract_pages
from sla.parsing.chapter_detect import (
    Section,
    SectionDetectionError,
    detect_sections,
    recover_toc_section_ids,
    validate_sections,
)

PDF = "/Users/liutao/Books/SuttonBartoIPRLBook2ndEd.pdf"

# §29 真实受害者:旧 buggy parse_toc_text 漏掉的 9 个(全书)
NINE_VICTIMS = {(3, 5), (3, 10), (5, 8), (7, 9), (7, 10), (11, 1), (11, 2), (11, 3), (14, 1)}


@pytest.fixture(scope="session")
def real_pages():
    return extract_pages(PDF)


def _sec(sid, ps, pe):
    """造 Section 测试桩(module-level helper,#5:勿再缝进 fixture)。"""
    return Section(
        chapter_id=f"ch{sid}", section_id=sid, title="x",
        book_page_start=ps, book_page_end=pe,
        pdf_page_start=ps, pdf_page_end=pe,
    )


def test_Dprime_recovers_ch3_incl_starred_and_inline(real_pages):
    ids = recover_toc_section_ids(real_pages)
    assert ids is not None, "真书有 TOC,D′ 不应返回 None"
    assert {b for a, b in ids if a == 3} == set(range(1, 11))   # ∗3.5 + 3.10


def test_Dprime_would_have_caught_all_9_victims(real_pages):
    allid = recover_toc_section_ids(real_pages)
    assert allid is not None
    buggy = allid - NINE_VICTIMS
    for maj in {a for a, _ in NINE_VICTIMS}:
        toc = {b for a, b in allid if a == maj}
        det = {b for a, b in buggy if a == maj}
        assert toc - det, f"ch{maj}: D′ 应抓到缺口(否则旧 bug 不会被拦)"


def test_A_catches_interior_gap(real_pages):
    polluted = [_sec(f"3.{i}", 0, 0) for i in (1, 2, 3, 4, 6, 7, 8, 9)]   # 缺 3.5
    with pytest.raises(SectionDetectionError, match=r"缺 3\.5"):
        validate_sections(polluted, real_pages)


def test_B1_does_NOT_fire_on_real_pollution(real_pages):
    # spans 源自 §29 实测污染态:3.4 吞 3.5=6页、3.9 吞 3.10=10页、中位 4(非拍脑袋)
    polluted = [
        _sec("3.1", 67, 70), _sec("3.2", 71, 72), _sec("3.3", 73, 74),
        _sec("3.4", 75, 80), _sec("3.6", 81, 83), _sec("3.7", 84, 88),
        _sec("3.8", 89, 92), _sec("3.9", 93, 102),
    ]
    try:
        validate_sections(polluted, real_pages)
    except SectionDetectionError as e:
        assert "B1" not in str(e)        # 证 B1 救不了场,A/D′ 才是拦截者


def test_clean_ch1_ch2_pass(real_pages):
    # 防误报:§29 说 ch1/2"全绿"是幸存者偏差,gate 不能误杀幸存者
    for ch in ("1", "2"):
        validate_sections(detect_sections(real_pages, chapter_filter=ch), real_pages)


def test_clean_ch3_postfix_pass(real_pages):
    # ch3 = §29 relabel A' 手术章。接 ingest live path 前必须确认 gate 不误杀
    # 已修复语料的 happy-path(否则一接线就 hard-fail 在自己修过的章)。
    validate_sections(detect_sections(real_pages, chapter_filter="3"), real_pages)
