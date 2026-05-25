"""Chapter detection -- Phase 2-W1-3.

Strategy (when no embedded TOC):
  1. Scan first 15 pages for TOC pages (containing 'Contents' header or dense section_id)
  2. Parse TOC text to extract section_id + title + book_page
  3. Find which PDF page 'Chapter N' is on -> compute pdf_offset
  4. Compute PDF page range for each section (start = book_page + offset, end = next section start - 1)

Outputs a list of Section; page fields use the same semantics as fixtures (book page, not PDF page).
"""
import logging
import re
from dataclasses import dataclass

from sla.parsing.pdf import Page

log = logging.getLogger(__name__)


# --- section-detection gate constants (single-point definition to prevent silent drift; per RETROSPECTIVE §29/§30 O1/O2) ---

# D' workhorse: an [independent, loose] section-number scan over raw toc_text. Key difference from parse_toc_text = anchoring:
#   parse_toc_text: line-by-line re.fullmatch(r"\d+(?:\.\d+)?", line) -- requires the [entire line to be only] a section number,
#                   so `∗3.5` (star prefix) and `3.10 Summary` (section-id + title on same line) fail fullmatch -> §29 miss
#   D'            : global finditer, allows [whitespace/star prefix] at line start, only [word boundary] required after the
#                   section number -- no full-line / no-asterisk requirement; both starred and inline forms hit
# The only shared dependency = extract_toc_text (locate + fetch Contents text); the §29 bug is not there (it's in the entry
# extraction's strict fullmatch), so D' is a true cross-check against the §29 bug class, not "circular self-validation".
# `∗` = U+2217 ASTERISK OPERATOR (textbook advanced-section marker), not ASCII `*`; we accept both.
_SECTION_TOKEN_RE = re.compile(r"(?m)^[ \t∗*]*(\d+)\.(\d+)\b")

# B1 span outlier: §29 empirically fails on real contamination (ch3.9 swallowing 3.10 was only 2.5x median, ch3.4 swallowing
# 3.5 only 1.5x, both < 3; under small samples + multi-section contamination, the polluted section inflates the median ->
# self-defeats). **WARN-only soft hint, do NOT promote back to a load-bearing gate.**
TRAILING_SPAN_FACTOR = 3
MIN_SECS_FOR_SPAN_STAT = 4

# B2 terminal-section title: in this textbook, ch1's terminal section is "Bibliographical Remarks", not Summary (empirical
# TOC), so TERMINAL is a [set]; and for this book the signal can only be WARN (see GATE_STRICT_TERMINAL).
TERMINAL_TITLE_RE = re.compile(r"^(summary|conclusion|bibliographical)\b", re.I)
GATE_STRICT_TERMINAL = False


class SectionDetectionError(ValueError):
    """Hard violation from the chapter-detection gate. Used to early-stop the ingest phase, not a graph exception."""


@dataclass
class Section:
    """Detection result for a single section (e.g. 1.3 / 2.1)."""
    chapter_id: str          # 'ch1.3' (used for DB Chunk.chapter_id, prefixed with 'ch')
    section_id: str          # '1.3' (raw TOC numbering)
    title: str               # 'Elements of Reinforcement Learning'
    book_page_start: int     # Section's starting book page (page number given by TOC)
    book_page_end: int       # Next section start - 1
    pdf_page_start: int      # book_page_start + pdf_offset
    pdf_page_end: int        # book_page_end + pdf_offset


# --------------------------------------------------------------------------- #
# Sub-steps
# --------------------------------------------------------------------------- #

def find_toc_pages(pages: list[Page], max_scan: int = 15) -> list[int]:
    """Find TOC pages in the PDF (list of 1-based pdf_page values)."""
    indices: list[int] = []
    started = False
    for p in pages[:max_scan]:
        has_contents = "Contents" in p.text[:200]
        section_count = len(re.findall(r"\n\d+\.\d+\n", p.text))
        if has_contents:
            indices.append(p.pdf_page)
            started = True
        elif started:
            # Already inside TOC; if section_ids stay dense -> TOC continuation page; otherwise TOC has ended
            if section_count >= 3:
                indices.append(p.pdf_page)
            else:
                break
    return indices


def parse_toc_text(toc_text: str) -> list[dict]:
    """Extract entry list from TOC text.

    TOC entries span multiple lines in the text; typical shape:
        section_id (e.g. '1.3' or '1' for chapter heading)
        title (e.g. 'Elements of Reinforcement Learning')
        dot leader (optional, e.g. '. . . . .')
        page number (e.g. '7')

    Parsing rules:
      - Anchor: line (after stripping leading ∗/*) starts with \\d+(\\.\\d+)?
        - Pure section_id on its own line (the norm, e.g. '3.1')
        - With leading asterisk (textbook marker for advanced/skippable sections, e.g. '∗3.5')
        - section_id + title on the same line (two-digit section/chapter numbers crammed together,
          e.g. '3.10 Summary', '11.1 Actor-Critic Methods')
      - After the anchor, accumulate title lines until we hit: a bare number (= page), or the next section_id
      - Bare page must be < 1000 (prevents accidentally swallowing the next chapter's title number)
      - Strip trailing dots from title lines
      - Roman numeral lines (front matter) terminate the entry

    Returns [{section_id, title, book_page}] (returns both the chapter itself and its subsections).
    """
    lines = toc_text.split("\n")
    entries: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Strip leading asterisks: textbook uses ∗ (U+2217) or * to mark advanced sections (∗3.5 / ∗5.8)
        stripped = re.sub(r"^[∗*]+\s*", "", line)
        anchor = re.match(r"^(\d+(?:\.\d+)?)(?:\s+(\S.*))?$", stripped)
        if not anchor:
            i += 1
            continue
        section_id = anchor.group(1)
        inline_rest = anchor.group(2)

        title_parts: list[str] = []
        page_num: int | None = None
        # section_id + title crammed onto same line: only accept inline title for subsections (those with a dot),
        # to avoid misreading a bare-page line '53' or a chapter line
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
            # Next subsection anchor (strip asterisk + tolerate inline title); presence of a dot = definitely a subsection
            cand_stripped = re.sub(r"^[∗*]+\s*", "", cand)
            if re.match(r"^\d+\.\d+(?:\s+\S.*)?$", cand_stripped):
                break          # Next subsection encountered; this entry ends (no page read, give up)
            # Bare integer: could be the next chapter id, or the current section's page
            if re.fullmatch(r"\d+", cand):
                if title_parts and int(cand) < 1000:
                    page_num = int(cand)
                    j += 1
                break              # If no title and we hit the next chapter id, also give up here
            # Roman numeral (front matter page) -> do not record this entry
            if re.fullmatch(r"[ivxlcdm]+", cand.lower()) and len(cand) < 6:
                break
            # Pure dot leader
            if re.fullmatch(r"[\s.]+", cand):
                j += 1
                continue
            # Otherwise it's part of the title; strip trailing dots
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
    """Find the 'Chapter N' heading in the PDF body and return its PDF page number."""
    pattern = re.compile(rf"^Chapter\s+{chapter_num}\s*$", re.MULTILINE)
    for p in pages:
        # The heading should be within the first ~200 chars of the page, to avoid matching reference mentions
        if pattern.search(p.text[:200]):
            return p.pdf_page
    return None


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def detect_sections(
    pages: list[Page],
    chapter_filter: str | None = None,
) -> list[Section]:
    """Main entry: returns a list of Section (subsections only, excluding the chapter heading itself).

    chapter_filter: e.g. '1' returns only ch1.x; None returns all.
    """
    # Shares the single toc_text with recover_toc_section_ids (D') (#2: guarantees byte-for-byte identical input).
    # `not toc_text.strip()` is a superset of the original `not toc_page_ids` (also covers "pages present but text empty").
    toc_text = extract_toc_text(pages)
    if not toc_text.strip():
        raise ValueError("no TOC pages found")

    raw_entries = parse_toc_text(toc_text)
    if not raw_entries:
        raise ValueError("no TOC entries parsed")   # Second guard, kept: pages and text are present but no entries extractable

    # Compute pdf_offset: which PDF page is Chapter 1 on -> offset = pdf - 1 (since ch1 is always book page 1)
    ch1_pdf = find_first_chapter_pdf_page(pages, 1)
    if ch1_pdf is None:
        raise ValueError("can't find 'Chapter 1' header in PDF body")
    pdf_offset = ch1_pdf - 1

    # Filter to subsections (section_id containing a dot)
    sub_entries = [e for e in raw_entries if "." in e["section_id"]]
    if chapter_filter is not None:
        sub_entries = [
            e for e in sub_entries
            if e["section_id"].split(".")[0] == chapter_filter
        ]

    sections: list[Section] = []
    for idx, e in enumerate(sub_entries):
        # end page: next section's start (in global order) - 1. Must look up the next any-section/chapter anchor in the
        # raw_entries list (cannot only look at sub_entries, otherwise cross-chapter boundaries miscompute)
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
# section-detection gate (O1/O2: deterministic gate, not a graph node, not a hook)
# --------------------------------------------------------------------------- #

def extract_toc_text(pages: list[Page]) -> str:
    """Raw text of the Contents region. detect_sections and recover_toc_section_ids share this [single] copy,
    guaranteeing D''s validation input is byte-identical to what parse_toc_text consumes -- otherwise the cross-check
    silently breaks (#2).

    D' soundness depends on find_toc_pages being a pure deterministic function of pages (same input -> same output),
    so the toc_text at both call sites must be byte-identical; whoever later adds state to find_toc_pages will break
    this invariant.
    """
    toc_page_ids = find_toc_pages(pages)
    return "\n".join(p.text for p in pages if p.pdf_page in toc_page_ids)


def recover_toc_section_ids(pages: list[Page]) -> set[tuple[int, int]] | None:
    """D': recover the section-number set via an independent loose regex, used as orthogonal validation of parse_toc_text.

    Returns None = TOC text is empty, D' is unavailable -- caller MUST [skip D'], must not treat the empty set as
    "all sections missing" -> catastrophic false alarm (#2 mirror).
    The FP/FN surface has only been empirically validated on the SuttonBarto book's toc_text (full 80-section book, 0/0);
    PDFs with different structure need re-validation (same class as §30 O9). Independence argument is in the
    _SECTION_TOKEN_RE comment.
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
    """A + D' are the FAIL workhorses, B1 is WARN-only, B2 is WARN for this book. Any FAIL -> raise.

    A: numbering continuity, reliably catches interior misses (3.5: 3.4 -> 3.6 gap).
    D': independent loose TOC recovery, covers interior + trailing misses (3.10: A's blind spot).
    """
    violations: list[tuple[str, str, str]] = []
    toc_ids_all = recover_toc_section_ids(pages)      # #3: compute once outside the loop
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

        if toc_ids_all is not None:                   # #3: if None, skip D' entirely
            toc_minors = {b for (a, b) in toc_ids_all if a == major}
            miss = sorted(toc_minors - set(minors))
            if miss:
                violations.append(("FAIL", "D'.toc_recover",
                    f"ch{major}: 独立 TOC 扫描见 "
                    f"{major}.{{{','.join(map(str, miss))}}} 但未产出(§29 同症)"))

        # B1 WARN-only: §29 empirically ch3.9 swallowing 3.10 = 2.5x median, ch3.4 swallowing 3.5 = 1.5x, both < 3, so it
        # self-defeats; reference only
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
