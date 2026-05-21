"""章节检测 —— Phase 2-W1-3.

策略(no embedded TOC 时):
  1. 扫前 15 页找 TOC 页(含 'Contents' header 或 section_id 密集)
  2. 解析 TOC text 抽 section_id + title + book_page
  3. 找 'Chapter N' 在 PDF 哪页 → 算 pdf_offset
  4. 算每个 section 的 PDF 页范围(start = book_page + offset, end = 下一节 start - 1)

输出 Section 列表,page 字段语义跟 fixture 一致(book page,不是 PDF page)。
"""
import logging
import re
from dataclasses import dataclass

from sla.parsing.pdf import Page

log = logging.getLogger(__name__)


# --- section-detection gate 常量(单点定义,防 silent drift;依据 RETROSPECTIVE §29/§30 O1/O2)---

# D′ workhorse:对 raw toc_text 的【独立宽松】节号扫描。与 parse_toc_text 的关键区别 = 锚定方式:
#   parse_toc_text:逐行 re.fullmatch(r"\d+(?:\.\d+)?", line) —— 要求【整行只有】节号,
#                   故 `∗3.5`(星前缀)和 `3.10 Summary`(节号+标题同行)fullmatch 失败 → §29 漏检
#   D′           :全局 finditer,行首允许【空白/星号前缀】,节号后只需【词边界】 ——
#                   不要求整行、不要求无星号,starred / inline 两种形态都命中
# 唯一共享依赖 = extract_toc_text(定位+取 Contents 文本),§29 bug 不在那(在 entry 抽取
# 的 strict fullmatch),故 D′ 对 §29 bug 类是真正交校验,非"循环自证"。
# `∗` = U+2217 ASTERISK OPERATOR(教材 advanced-section 标记),非 ASCII `*`;两个都收。
_SECTION_TOKEN_RE = re.compile(r"(?m)^[ \t∗*]*(\d+)\.(\d+)\b")

# B1 跨度离群:§29 实测对真实污染失效(ch3.9 吞 3.10 仅 2.5×中位、ch3.4 吞 3.5 仅 1.5×,
# 均 < 3;小样本+多节污染时污染节自抬中位 → 自我失效)。**WARN-only 软提示,勿再升承重位。**
TRAILING_SPAN_FACTOR = 3
MIN_SECS_FOR_SPAN_STAT = 4

# B2 末节标题:本书 ch1 末节是 "Bibliographical Remarks" 非 Summary(实证 TOC),
# 故 TERMINAL 是【集合】;且本书该信号只能 WARN(见 GATE_STRICT_TERMINAL)。
TERMINAL_TITLE_RE = re.compile(r"^(summary|conclusion|bibliographical)\b", re.I)
GATE_STRICT_TERMINAL = False


class SectionDetectionError(ValueError):
    """章节检测 gate 硬违规。ingest 阶段早停用,不是 graph 异常。"""


@dataclass
class Section:
    """单节(1.3 / 2.1 等)的检测结果。"""
    chapter_id: str          # 'ch1.3'(给 DB Chunk.chapter_id 用,前缀 'ch')
    section_id: str          # '1.3'(原始 TOC 编号)
    title: str               # 'Elements of Reinforcement Learning'
    book_page_start: int     # 节起始书页(TOC 给的页码)
    book_page_end: int       # 下一节起始 - 1
    pdf_page_start: int      # book_page_start + pdf_offset
    pdf_page_end: int        # book_page_end + pdf_offset


# --------------------------------------------------------------------------- #
# 子步骤
# --------------------------------------------------------------------------- #

def find_toc_pages(pages: list[Page], max_scan: int = 15) -> list[int]:
    """找 PDF 里的 TOC 页(1-based pdf_page 列表)。"""
    indices: list[int] = []
    started = False
    for p in pages[:max_scan]:
        has_contents = "Contents" in p.text[:200]
        section_count = len(re.findall(r"\n\d+\.\d+\n", p.text))
        if has_contents:
            indices.append(p.pdf_page)
            started = True
        elif started:
            # 已进入 TOC,后续 section_id 密集 → TOC 续页;否则 TOC 结束
            if section_count >= 3:
                indices.append(p.pdf_page)
            else:
                break
    return indices


def parse_toc_text(toc_text: str) -> list[dict]:
    """从 TOC 文本提取条目列表。

    TOC 条目在文本里跨多行,典型形态:
        section_id (e.g. '1.3' 或 '1' for chapter heading)
        title (e.g. 'Elements of Reinforcement Learning')
        dot leader(可能,e.g. '. . . . .')
        page number (e.g. '7')

    解析规则:
      - 锚点:行(去前导 ∗/* 后)以 \\d+(\\.\\d+)? 开头
        · 纯 section_id 一行(常态,如 '3.1')
        · 带前导星号(教材标记 advanced/可跳过节,如 '∗3.5')
        · section_id + title 同一行(两位数节号/章号挤同行,如 '3.10 Summary'
          '11.1 Actor–Critic Methods')
      - 锚点后累积 title 行,直到遇到:纯数字(=page),或下一个 section_id
      - 纯 page 必须 < 1000(防止误吞下章标题数字)
      - title 行去尾点
      - Roman numeral 行(前 matter)中断

    返回 [{section_id, title, book_page}] (chapter 自身和子节都返回)。
    """
    lines = toc_text.split("\n")
    entries: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # 去前导星号:教材用 ∗(U+2217)或 * 标记 advanced section(∗3.5 / ∗5.8)
        stripped = re.sub(r"^[∗*]+\s*", "", line)
        anchor = re.match(r"^(\d+(?:\.\d+)?)(?:\s+(\S.*))?$", stripped)
        if not anchor:
            i += 1
            continue
        section_id = anchor.group(1)
        inline_rest = anchor.group(2)

        title_parts: list[str] = []
        page_num: int | None = None
        # section_id + title 挤同行:仅对子节(含点)接受 inline title,
        # 避免把纯 page 行 '53' 或 chapter 行误判
        if inline_rest and "." in section_id:
            cleaned = re.sub(r"[\s.]+$", "", inline_rest).strip()
            if cleaned and not re.fullmatch(r"\d+", cleaned):
                title_parts.append(cleaned)
        j = i + 1
        while j < len(lines):
            cand = lines[j].strip()
            if not cand:
                j += 1
                continue
            # 下一个子节锚点(去星号 + 兼容 inline title);含点 = 必然子节
            cand_stripped = re.sub(r"^[∗*]+\s*", "", cand)
            if re.match(r"^\d+\.\d+(?:\s+\S.*)?$", cand_stripped):
                break          # 下一个子节,这条结束(没读到 page,放弃)
            # 纯整数:可能是下一个 chapter id 或当前节的 page
            if re.fullmatch(r"\d+", cand):
                if title_parts and int(cand) < 1000:
                    page_num = int(cand)
                    j += 1
                break              # 无 title 碰到下一 chapter id 也在此放弃
            # 罗马数字(前 matter 的 page) → 这条不计入
            if re.fullmatch(r"[ivxlcdm]+", cand.lower()) and len(cand) < 6:
                break
            # 纯 dot leader
            if re.fullmatch(r"[\s.]+", cand):
                j += 1
                continue
            # 否则是 title 一部分,去尾部 dots
            cleaned = re.sub(r"[\s.]+$", "", cand).strip()
            if cleaned:
                title_parts.append(cleaned)
            j += 1

        if page_num is not None and title_parts:
            entries.append({
                "section_id": section_id,
                "title": " ".join(title_parts),
                "book_page": page_num,
            })
        i = j

    return entries


def find_first_chapter_pdf_page(pages: list[Page], chapter_num: int = 1) -> int | None:
    """在 PDF 正文里找 'Chapter N' 标题,返回 PDF 页号。"""
    pattern = re.compile(rf"^Chapter\s+{chapter_num}\s*$", re.MULTILINE)
    for p in pages:
        # 标题应该在页首 ~200 字符内,避免 match 引用提及
        if pattern.search(p.text[:200]):
            return p.pdf_page
    return None


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #

def detect_sections(
    pages: list[Page],
    chapter_filter: str | None = None,
) -> list[Section]:
    """主入口:返回 Section 列表(只含子节,不含 chapter 标题自身)。

    chapter_filter: 形如 '1',只返回 ch1.x;None 返回全部。
    """
    # 与 recover_toc_section_ids(D′)共用唯一一份 toc_text(#2:保证逐字节同输入)。
    # not toc_text.strip() 是原 `not toc_page_ids` 的 superset(也覆盖"页在但文本空")。
    toc_text = extract_toc_text(pages)
    if not toc_text.strip():
        raise ValueError("no TOC pages found")

    raw_entries = parse_toc_text(toc_text)
    if not raw_entries:
        raise ValueError("no TOC entries parsed")   # 第二道 guard,保留:页在文本在但抽不出条目

    # 算 pdf_offset:Chapter 1 在 PDF 哪页 → offset = pdf - 1 (因为 ch1 总是 book page 1)
    ch1_pdf = find_first_chapter_pdf_page(pages, 1)
    if ch1_pdf is None:
        raise ValueError("can't find 'Chapter 1' header in PDF body")
    pdf_offset = ch1_pdf - 1

    # 过滤出子节(含点的 section_id)
    sub_entries = [e for e in raw_entries if "." in e["section_id"]]
    if chapter_filter is not None:
        sub_entries = [
            e for e in sub_entries
            if e["section_id"].split(".")[0] == chapter_filter
        ]

    sections: list[Section] = []
    for idx, e in enumerate(sub_entries):
        # end page:下一节(按全局顺序)起始 - 1。需要从原始 raw_entries 里找下一个
        # 任意 section/chapter 锚点(不能只看 sub_entries,否则跨 chapter 时算错)
        global_idx = raw_entries.index(e)
        if global_idx + 1 < len(raw_entries):
            next_book_page = raw_entries[global_idx + 1]["book_page"]
            book_page_end = max(e["book_page"], next_book_page - 1)
        else:
            book_page_end = e["book_page"]

        sections.append(Section(
            chapter_id=f"ch{e['section_id']}",
            section_id=e["section_id"],
            title=e["title"],
            book_page_start=e["book_page"],
            book_page_end=book_page_end,
            pdf_page_start=e["book_page"] + pdf_offset,
            pdf_page_end=book_page_end + pdf_offset,
        ))
    return sections


# --------------------------------------------------------------------------- #
# section-detection gate(O1/O2:确定性闸门,非 graph node 非 hook)
# --------------------------------------------------------------------------- #

def extract_toc_text(pages: list[Page]) -> str:
    """Contents 区原始文本。detect_sections 与 recover_toc_section_ids 共用【唯一】一份,
    保证 D′ 校验的输入与 parse_toc_text 消费的逐字节相同——否则交叉校验静默失效(#2)。

    D′ soundness 依赖 find_toc_pages 为 pages 的纯确定函数(同输入→同输出),
    故两调用点 toc_text 必逐字节同;谁将来给 find_toc_pages 加状态,这条不变量即断。
    """
    toc_page_ids = find_toc_pages(pages)
    return "\n".join(p.text for p in pages if p.pdf_page in toc_page_ids)


def recover_toc_section_ids(pages: list[Page]) -> set[tuple[int, int]] | None:
    """D′:独立宽松正则复原节号集合,作 parse_toc_text 的正交校验。

    返回 None = TOC 文本为空,D′ 不可用 —— 调用方须【跳过 D′】,不得把空集当成
    "所有节都缺"→灾难性误报(#2 镜像)。
    FP/FN 面仅在 SuttonBarto 本书 toc_text 实测过(全书 80 节 0/0);
    结构不同的 PDF 需重验(归 §30 O9 同类)。独立性论证见 _SECTION_TOKEN_RE 注释。
    """
    toc_text = extract_toc_text(pages)
    if not toc_text.strip():
        return None
    return {
        (int(m.group(1)), int(m.group(2)))
        for m in _SECTION_TOKEN_RE.finditer(toc_text)
    }


def validate_sections(
    sections: list[Section],
    pages: list[Page],
    *,
    strict_terminal: bool = GATE_STRICT_TERMINAL,
) -> None:
    """A + D′ 为 FAIL 双主力,B1 仅 WARN,B2 本书 WARN。任一 FAIL → raise。

    A:编号连续性,稳覆盖 interior 漏(3.5:3.4→3.6 跳号)。
    D′:独立宽松 TOC 复原,覆盖 interior + trailing 漏(3.10:A 盲区)。
    """
    violations: list[tuple[str, str, str]] = []
    toc_ids_all = recover_toc_section_ids(pages)      # #3:循环外算一次
    if toc_ids_all is None:
        log.warning("[section-gate] D′ 跳过:TOC 文本为空,本次无 D′ 交叉校验")

    by_major: dict[int, list[Section]] = {}
    for s in sections:
        by_major.setdefault(int(s.section_id.split(".")[0]), []).append(s)

    for major, secs in sorted(by_major.items()):
        secs = sorted(secs, key=lambda s: int(s.section_id.split(".")[1]))
        minors = [int(s.section_id.split(".")[1]) for s in secs]

        gap = sorted(set(range(1, minors[-1] + 1)) - set(minors))
        if gap:
            violations.append(("FAIL", "A.gap",
                f"ch{major}: 缺 " + ", ".join(f"{major}.{m}" for m in gap)))

        if toc_ids_all is not None:                   # #3:None 则整体跳过 D′
            toc_minors = {b for (a, b) in toc_ids_all if a == major}
            miss = sorted(toc_minors - set(minors))
            if miss:
                violations.append(("FAIL", "D'.toc_recover",
                    f"ch{major}: 独立 TOC 扫描见 "
                    f"{major}.{{{','.join(map(str, miss))}}} 但未产出(§29 同症)"))

        # B1 WARN-only:§29 实测 ch3.9 吞3.10=2.5×中位、ch3.4 吞3.5=1.5×,均<3 故失效,仅参考
        spans = [s.pdf_page_end - s.pdf_page_start + 1 for s in secs]
        if len(secs) >= MIN_SECS_FOR_SPAN_STAT:
            med = sorted(spans)[len(spans) // 2]
            for s, sp in zip(secs, spans):
                if med > 0 and sp >= TRAILING_SPAN_FACTOR * med:
                    log.warning(
                        f"[section-gate WARN B1.span] ch{major}: {s.section_id} "
                        f"{sp}页={sp/med:.1f}×中位(此信号 §29 实测会漏,仅参考)")

        if not TERMINAL_TITLE_RE.match(secs[-1].title.strip().lower()):
            violations.append(
                ("FAIL" if strict_terminal else "WARN", "B2.terminal",
                 f"ch{major}: 末节 {secs[-1].section_id}={secs[-1].title!r} 非 terminal 集"))

    for lvl, code, msg in violations:
        if lvl == "WARN":
            log.warning(f"[section-gate WARN {code}] {msg}")
    fails = [v for v in violations if v[0] == "FAIL"]
    if fails:
        raise SectionDetectionError(
            "\n".join(f"  [{c}] {m}" for l, c, m in violations if l == "FAIL")
            + f"\n→ {len(fails)} 项硬违规")
