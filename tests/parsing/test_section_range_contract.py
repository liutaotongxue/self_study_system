r"""O3:section 页范围【契约回归锁】(CI,非 runtime gate)。

修正-C1 谓词对正确 detect_sections 输出恒绿(全书 11 章实测零-FP:
ch1-9,11,14)。它本质是契约锁:未来 chapter_detect 页算术回归 → CI 在
ship 前红,比 runtime gate 早且便宜;不接 ingest 故零 §29-误杀-生产风险。

§29(漏节)归 Eval-1(validate_sections),不在此 fire —— 正交锁测试钉死。

【已命名残留 / O9-class trigger,未由本测试覆盖,勿当已解决】
全局 pdf_offset 误算:chapter_detect 的 `^Chapter\s+1\s*$` 被前言/分卷页
误匹配 → 所有节均匀偏移 → 相对连续性不变 → 修正-C1 盲、Eval-1 也盲。
原 C2(章跨度独立交叉核)是唯一能抓它的,因被空中间章 ch10 证伪而删
(见 RETROSPECTIVE §30)。chapter_detect:169 仅挡 ch1 完全检测失败(None),
不挡 ch1 错误检测(匹配错页)。重访 trigger = 灌结构不同 PDF / 改 ch1-header 匹配。
"""
import pytest

from sla.parsing.pdf import extract_pages
from sla.parsing.chapter_detect import Section, detect_sections

PDF = "/Users/liutao/Books/SuttonBartoIPRLBook2ndEd.pdf"


@pytest.fixture(scope="session")
def real_pages():
    return extract_pages(PDF)


def _c1_violations(sections):
    """修正-C1:违规 iff 深重叠(a.end>b.start) 或 空洞(a.end+1<b.start);
    放行 a.end==b.start(短 starred 节合法共享边界,如 7.9/7.10、8.8/8.9)。"""
    by_major = {}
    for s in sections:
        by_major.setdefault(int(s.section_id.split(".")[0]), []).append(s)
    out = []
    for major, secs in sorted(by_major.items()):
        secs = sorted(secs, key=lambda s: int(s.section_id.split(".")[1]))
        for a, b in zip(secs, secs[1:]):
            if a.pdf_page_end > b.pdf_page_start:
                out.append(f"ch{major}: {a.section_id}>{b.section_id} 深重叠")
            elif a.pdf_page_end + 1 < b.pdf_page_start:
                out.append(f"ch{major}: {a.section_id}↮{b.section_id} 空洞")
    return out


def _s(sid, ps, pe):
    return Section(chapter_id=f"ch{sid}", section_id=sid, title="x",
                   book_page_start=ps, book_page_end=pe,
                   pdf_page_start=ps, pdf_page_end=pe)


# --- 契约锁:全书正确 detect 输出必须零违规(改坏页算术 → CI 红)---
def test_contract_corrected_C1_zero_violation_full_book(real_pages):
    assert _c1_violations(detect_sections(real_pages)) == []


def test_contract_span_ge_1_full_book(real_pages):
    # 直接断言(honest-A:不靠修正-C1 间接覆盖,边界对齐 inverted span 会溜过它)
    bad = [s.section_id for s in detect_sections(real_pages)
           if s.pdf_page_end < s.pdf_page_start]
    assert bad == []


# --- 失败类:证明这锁真能抓回归(非 green-always)---
def test_detects_deep_overlap(real_pages):
    assert _c1_violations([_s("3.4", 75, 80), _s("3.5", 76, 80)])  # 80>76


def test_detects_gap(real_pages):
    assert _c1_violations([_s("3.4", 75, 75), _s("3.5", 78, 80)])  # 76<78


# --- 正交锁:§29 漏节态不 fire(那是 Eval-1 的活,O3 不抢、不误归因)---
def test_does_NOT_fire_on_sec29_missing_section_state(real_pages):
    sec29 = [_s("3.1", 67, 70), _s("3.2", 71, 72), _s("3.3", 73, 74),
             _s("3.4", 75, 80), _s("3.6", 81, 83), _s("3.7", 84, 88),
             _s("3.8", 89, 92), _s("3.9", 93, 102)]   # 3.5/3.10 缺;3.4.end+1=81==3.6.start
    assert _c1_violations(sec29) == []
