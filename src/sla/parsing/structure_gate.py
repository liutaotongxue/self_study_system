"""1c:agent 外确定性结构 gate(纯函数;无 DB、无网络、无 LLM)。

独立性命脉:本模块只吃 (pages, proposed)。pages = 1c 自己 extract_pages
重扫的正文(【不】吃 agent 的 probe_body 输出);proposed = 逐字读自
ToolCall.input。校验信号 _SECTION_TOKEN_RE(regex-on-body)与抽取
(agent-on-TOC)永不同源 → 非 green-by-construction。

层 A 硬门 = well-formedness。明确标注:这是 green-by-construction —— 只
抓"提议自相矛盾",不证"结构正确",必要非充分(见记忆
chapter_detect-page-invariants-are-circular)。
层 B = 唯一真·独立信号,双向:
  B-1 提议⊆正文(每提议节 token 在其声称 pdf 页±tol 出现)→ 命中率;
  B-2 正文独立正扫 S_body,提议⊇S_body → §29 守卫【本体】(抓漏节)。
层 C offset 恒定 = WARN-only(B1-否决:此族 §29 真实污染失效,
  chapter_detect:31-33"勿再升承重位";升硬门须 S&B+§29 实测零误 fire)。

无任何硬等值 / 写死计数:全报数,B/C 阈值与 tol 由真数据后定(fork①④)。
"""
import re

# 复用仓库已验的独立 token 匹配器(用户硬约束:不另写 regex)
from sla.parsing.chapter_detect import _SECTION_TOKEN_RE  # noqa: PLC2701
from sla.parsing.pdf import Page

_TWO_LEVEL = re.compile(r"^\d+\.\d+$")
_SEVEN = (
    "section_id", "chapter_id", "title",
    "book_page_start", "book_page_end", "pdf_page_start", "pdf_page_end",
)


def _skey(sid: str):
    """两级 'a.b' → [a,b] 排序键;非两级 → 巨值沉底(不崩)。"""
    if _TWO_LEVEL.match(sid):
        return [int(t) for t in sid.split(".")]
    return [10 ** 9]


class GateReport:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    @property
    def hard_pass(self) -> bool:
        # v1:仅层 A 是硬门。B/C 为测量+WARN(阈值未由真数据定 → 不 gate,fork①)。
        return not self.layer_a_violations


def _body_scan(pages: list[Page]) -> dict:
    """独立正扫:_SECTION_TOKEN_RE 扫全 PDF 行首 → {section_id: [(pdf_page, line)]}"""
    found: dict = {}
    for p in pages:
        for m in _SECTION_TOKEN_RE.finditer(p.text):
            sid = f"{m.group(1)}.{m.group(2)}"
            line = p.text[m.start():m.start() + 90].splitlines()[0].strip()
            found.setdefault(sid, []).append((p.pdf_page, line))
    return found


def verify_structure_proposal(pages, proposed, pdf_tol: int = 1) -> GateReport:
    proposed = proposed or []

    # ---- 层 A 硬门(green-by-construction:必要非充分) ----
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

    # ---- 层 B-2 独立正扫(§29 守卫本体) ----
    body = _body_scan(pages)
    s_body = sorted(body.keys(), key=_skey)
    prop_ids = {str(s.get("section_id")) for s in proposed if isinstance(s, dict)}
    missing = [sid for sid in s_body if sid not in prop_ids]   # 含噪 → 人工查
    extra = sorted(prop_ids - set(s_body), key=_skey)
    samples = [(sid, body[sid][0][0], body[sid][0][1]) for sid in s_body[:25]]

    # ---- 层 B-1 提议⊆正文 命中(声称 pdf 页 ±tol) ----
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

    # ---- 层 C offset(WARN-only,B1-否决) ----
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
