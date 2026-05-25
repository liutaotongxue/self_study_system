"""1c: deterministic structure gate outside the agent (pure function; no DB, no network, no LLM).

Independence lifeline: this module consumes only (pages, proposed). pages = body re-scanned by 1c's own extract_pages
([NOT] the agent's probe_body output); proposed = read byte-for-byte from ToolCall.input. The validation signal
_SECTION_TOKEN_RE (regex-on-body) and the extraction (agent-on-TOC) never share a source -> not green-by-construction.

Layer A hard gate = well-formedness. Explicitly noted: this is green-by-construction -- it only catches "proposal
self-contradictions", does not prove "structure is correct"; necessary not sufficient (see memory
chapter_detect-page-invariants-are-circular).
Layer B = the only truly independent signal, bidirectional:
  B-1 proposal subset-of body (each proposed section's token appears within ±tol of its claimed pdf page) -> hit rate;
  B-2 independent forward scan of body S_body, proposal superset-of S_body -> §29 guard [proper] (catches missed sections).
Layer C constant offset = WARN-only (B1-veto: this family self-defeated on real §29 contamination,
  chapter_detect:31-33 "do not promote back to a load-bearing gate"; promotion requires S&B + §29 empirical zero false-fire).

No hard equalities / hardcoded counts: everything is reported numerically; B/C thresholds and tol are set later from
real data (fork (1)(4)).
"""
import re

# Reuse the repo's already-validated independent token matcher (user hard constraint: do not write another regex)
from sla.parsing.chapter_detect import _SECTION_TOKEN_RE  # noqa: PLC2701
from sla.parsing.pdf import Page

_TWO_LEVEL = re.compile(r"^\d+\.\d+$")
_SEVEN = (
    "section_id", "chapter_id", "title",
    "book_page_start", "book_page_end", "pdf_page_start", "pdf_page_end",
)


def _skey(sid: str):
    """Two-level 'a.b' -> [a,b] sort key; non-two-level -> huge value sinks to bottom (no crash)."""
    if _TWO_LEVEL.match(sid):
        return [int(t) for t in sid.split(".")]
    return [10 ** 9]


class GateReport:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    @property
    def hard_pass(self) -> bool:
        # v1: only Layer A is a hard gate. B/C are measurement + WARN (thresholds not yet set from real data -> no gating, fork (1)).
        return not self.layer_a_violations


def _body_scan(pages: list[Page]) -> dict:
    """Independent forward scan: run _SECTION_TOKEN_RE across all PDF line starts -> {section_id: [(pdf_page, line)]}"""
    found: dict = {}
    for p in pages:
        for m in _SECTION_TOKEN_RE.finditer(p.text):
            sid = f"{m.group(1)}.{m.group(2)}"
            line = p.text[m.start():m.start() + 90].splitlines()[0].strip()
            found.setdefault(sid, []).append((p.pdf_page, line))
    return found


def verify_structure_proposal(pages, proposed, pdf_tol: int = 1) -> GateReport:
    proposed = proposed or []

    # ---- Layer A hard gate (green-by-construction: necessary not sufficient) ----
    viol = []
    for i, s in enumerate(proposed):
        if not isinstance(s, dict):
            viol.append((i, "not-a-dict"))
            continue
        miss = [k for k in _SEVEN if k not in s]
        if miss:
            viol.append((s.get("section_id", i), f"missing {miss}"))
            continue
        sid = str(s["section_id"])
        if not _TWO_LEVEL.match(sid):
            viol.append((sid, "section_id 非两级十进制 N.M(Hole A 范围墙)"))
        if str(s["chapter_id"]) != f"ch{sid}":
            viol.append((sid, f"chapter_id={s['chapter_id']!r} != 'ch{sid}'"))
        try:
            bs, be = int(s["book_page_start"]), int(s["book_page_end"])
            ps, pe = int(s["pdf_page_start"]), int(s["pdf_page_end"])
        except (TypeError, ValueError):
            viol.append((sid, "页字段非整数"))
            continue
        if not (1 <= bs <= be) or not (1 <= ps <= pe):
            viol.append((sid, f"页序非法 book[{bs},{be}] pdf[{ps},{pe}]"))

    # ---- Layer B-2 independent forward scan (§29 guard proper) ----
    body = _body_scan(pages)
    s_body = sorted(body.keys(), key=_skey)
    prop_ids = {str(s.get("section_id")) for s in proposed if isinstance(s, dict)}
    missing = [sid for sid in s_body if sid not in prop_ids]   # Contains noise -> human review
    extra = sorted(prop_ids - set(s_body), key=_skey)
    samples = [(sid, body[sid][0][0], body[sid][0][1]) for sid in s_body[:25]]

    # ---- Layer B-1 proposal subset-of body hit (claimed pdf page ±tol) ----
    by_page: dict = {}
    for p in pages:
        for m in _SECTION_TOKEN_RE.finditer(p.text):
            by_page.setdefault(
                f"{m.group(1)}.{m.group(2)}", set()).add(p.pdf_page)
    b1_hit, b1_misses, b1_total = 0, [], 0
    for s in proposed:
        if not isinstance(s, dict) or "pdf_page_start" not in s:
            continue
        b1_total += 1
        sid = str(s.get("section_id"))
        try:
            claim = int(s["pdf_page_start"])
        except (TypeError, ValueError):
            b1_misses.append((sid, s.get("pdf_page_start")))
            continue
        if any(abs(pp - claim) <= pdf_tol for pp in by_page.get(sid, ())):
            b1_hit += 1
        else:
            b1_misses.append((sid, claim))

    # ---- Layer C offset (WARN-only, B1-vetoed) ----
    offsets = {}
    for s in proposed:
        if not isinstance(s, dict):
            continue
        try:
            offsets[str(s["section_id"])] = (
                int(s["pdf_page_start"]) - int(s["book_page_start"]))
        except (TypeError, ValueError, KeyError):
            pass
    offset_constant = len(set(offsets.values())) <= 1

    return GateReport(
        n_proposed=len(proposed),
        layer_a_violations=viol,
        b1_hit=b1_hit, b1_total=b1_total, b1_misses=b1_misses,
        s_body=s_body, missing_vs_body=missing, extra_vs_body=extra,
        s_body_samples=samples,
        offsets=offsets, offset_constant=offset_constant,
    )
