"""KG extraction (Phase 2-W3-4) -- extract concepts + relations from a single Note.

**Architecture**: standalone service, does not go through the run_task pipeline. Rationale:
  - Single-turn LLM call, no multi-turn decision-making needed
  - run_task's Step/ToolCall trace + rule_evaluate are meaningless for KG extraction
  - Use ChatAnthropic.with_structured_output(KGExtraction) directly to get Pydantic objects

**LLM output contract**:
  - concepts: list[{label, type in {concept,method,example}, description}]
  - relations: list[{source_label, target_label, type in {requires,related_to,contrasts_with,example_of}, notes?}]
  - **LLM only provides label**; server uses slugify to generate external_id (avoids LLM inconsistency at slug level)

**Context window**: when extracting a Note, optionally feed "existing concepts label list" as dedup hint,
limited to most recent 30 labels to prevent prompt token blow-up.
"""
import json
import re
from dataclasses import dataclass
from typing import Annotated, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, BeforeValidator, Field

from sla.config import settings
from sla.db import SessionLocal
from sla.models.domain import Note
from sla.models.kg import KGNode


ConceptType = Literal["concept", "method", "example"]
RelationType = Literal["requires", "related_to", "contrasts_with", "example_of"]


# --------------------------------------------------------------------------- #
# Pydantic schemas for LLM structured output
# --------------------------------------------------------------------------- #

def _parse_if_json_string(v):
    """Same defense as Phase 1B-1 + tolerance for LLM trailing garbage (observed on ch3.10 rebuild job2:
    after a valid array, an extra '[\\n]' (=[]) caused json.loads "Extra data" -> build_kg rc=1).
    Strict failure -> raw_decode takes the first complete JSON value; tail guard: only re-raise when
    the trailing value is a **non-empty** list/dict (a real second value with content), keeping it loud
    instead of silently dropping extracted data; empty []/{}/non-JSON garbage -> ignore.
    Well-formed path is unchanged."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            obj, end = json.JSONDecoder().raw_decode(v.lstrip())
            tail = v.lstrip()[end:].strip()
            if tail:
                try:
                    t = json.loads(tail)
                except json.JSONDecodeError:
                    t = None
                if isinstance(t, (list, dict)) and len(t) > 0:
                    raise   # trailing non-empty second value: stay loud, do not silently drop data
            return obj
    return v


class ConceptItem(BaseModel):
    """A single concept emitted by the LLM."""
    label: str = Field(
        description="人类可读的名称,**与教材原文语言一致**(英文教材出英文 label,"
        "中文教材出中文 label),2-4 个词,直接复用源文本术语。"
        "英文 label 用单数形式不带冠词(如 'policy' 不是 'policies' 或 'the value function');"
        "中文 label 直接照搬术语(如 '策略'、'价值函数'、'监督学习')。"
    )
    type: ConceptType = Field(
        description="concept(定义/定理)| method(算法/技术)| example(具体实例)"
    )
    description: str | None = Field(
        default=None,
        description="1-2 句话定义,与教材原文语言一致(中文教材就中文,英文教材就英文),"
        "复用原文用词不翻译漂移",
    )


class RelationItem(BaseModel):
    """A single relation emitted by the LLM."""
    source_label: str = Field(description="边的起点 label,**必须**出现在 concepts 列表里")
    target_label: str = Field(description="边的终点 label,**必须**出现在 concepts 列表里")
    type: RelationType = Field(
        description="requires(有向前置)| related_to(无向弱关联)| "
        "contrasts_with(无向对比)| example_of(有向实例)"
    )
    notes: str | None = Field(
        default=None,
        description="可选 1 句话说明关系语境",
    )


class KGExtraction(BaseModel):
    """Top-level LLM output structure."""
    concepts: Annotated[
        list[ConceptItem], BeforeValidator(_parse_if_json_string)
    ] = Field(description="本 Note 抽出的 5-10 个 concepts/methods/examples")
    relations: Annotated[
        list[RelationItem], BeforeValidator(_parse_if_json_string)
    ] = Field(description="本 Note 抽出的 3-8 条关系")


# --------------------------------------------------------------------------- #
# Server-side helpers: slug + external_id
# --------------------------------------------------------------------------- #

_ARTICLE_RE = re.compile(r"\b(?:the|a|an)\b", flags=re.IGNORECASE)


def _singularize_word(word: str) -> str:
    """Lightweight English singularization: covers most common plural patterns, not exhaustive.
    **Non-ASCII words (e.g. Chinese '策略' / '价值函数') are returned as-is** --
    other languages have no English plural concept; applying rules causes false strips
    (2026-05-21 multilingual KG fix).

    English rules:
      - length <= 3: leave alone (avoid short-word false strips like 'is', 'as')
      - 'ies' ending and len > 4: replace with 'y' (policies -> policy)
      - 'sses' ending: drop 'es' (processes -> process)
      - 'ss' ending: leave alone (process, class -- already singular)
      - 's' ending: drop 's' (values -> value, actions -> action)
      - otherwise: leave alone

    Known false strips: 'axis' -> 'axi', 'iris' -> 'iri'. Not frequent in the RL domain; revisit in Phase 2+.
    """
    if not word.isascii():
        return word                       # Chinese/CJK/other non-ASCII: skip singularization
    w = word.lower()
    if len(w) <= 3:
        return word
    if w.endswith("ies") and len(w) > 4:
        return word[:-3] + "y"
    if w.endswith("sses"):
        return word[:-2]
    if w.endswith("ss"):
        return word
    if w.endswith("s"):
        return word[:-1]
    return word


def parse_markdown_sections(md: str) -> list[tuple[str, str]]:
    """Split a Markdown text into [(heading, content), ...] segments.

    A section = one #/##/###/#### heading line + everything that follows up to the next heading.
    Content before any heading (preamble) is ignored -- Notes usually start with # so there is no preamble.

    Example:
        '''
        # Ch1.3 Elements
        intro
        ## 1. Policy
        a policy is ...
        ## 2. Reward
        reward is ...
        '''
        -> [('Ch1.3 Elements', 'intro\\n'),
            ('1. Policy', 'a policy is ...\\n'),
            ('2. Reward', 'reward is ...\\n')]
    """
    sections: list[tuple[str, str]] = []
    cur_heading: str | None = None
    cur_lines: list[str] = []
    for line in md.split("\n"):
        m = re.match(r"^(#{1,4})\s+(.+)$", line)
        if m:
            if cur_heading is not None:
                sections.append((cur_heading, "\n".join(cur_lines)))
            cur_heading = m.group(2).strip()
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_heading is not None:
        sections.append((cur_heading, "\n".join(cur_lines)))
    return sections


def slugify_heading(heading: str) -> str:
    """Markdown heading -> URL slug (GitHub-style). **Accepts both raw and already-stripped ## forms.**

    Examples:
      '## 2. Action-Value Methods'   -> 'action-value-methods'
      '2. Action-Value Methods'      -> 'action-value-methods'  (form produced by parse_markdown_sections)
      '### Reward vs. Value'         -> 'reward-vs-value'
      'TD Update Rule (Eq. 2.4)'     -> 'td-update-rule-eq-2-4'
      '## 1.1 Reinforcement Learning'-> 'reinforcement-learning'

    Rules:
      1. lowercase + strip
      2. drop leading markdown ## prefix (if still present)
      3. drop leading numeric prefix (e.g. '1.', '2.3.')
      4. non [a-z0-9] -> hyphen, collapse, trim ends
      5. if result is empty (purely numeric heading), fallback to a hash of the original string
    """
    s = heading.lower().strip()
    s = re.sub(r"^#+\s+", "", s)            # accept '## xxx' form
    s = re.sub(r"^[\d.]+\s*", "", s)        # drop leading '2.' / '1.1.'
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    if not s:
        # Edge case (purely numeric heading): fallback to simple normalize of original heading
        s = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-") or "unnamed"
    return s


def slugify_headings(headings: list[str]) -> list[str]:
    """Batch-slugify a group of headings; collisions get GitHub-style -2 / -3 suffixes.

    A single slugify_heading cannot detect collisions like "within one Note: epsilon-Greedy vs Greedy"
    where Unicode differences get eaten and produce the same slug. The batch entry point has the
    global view and can rename on the second occurrence.

    Example:
      ['Greedy Action Selection', 'epsilon-Greedy Action Selection']
      -> ['greedy-action-selection', 'greedy-action-selection-2']
    """
    seen: dict[str, int] = {}
    result: list[str] = []
    for h in headings:
        base = slugify_heading(h)
        n = seen.get(base, 0) + 1
        seen[base] = n
        result.append(base if n == 1 else f"{base}-{n}")
    return result


def slugify(label: str) -> str:
    """label -> slug: Phase 2-W3-4 upgrade, **aggressive normalization for stable dedup**.

    Steps:
      1. lowercase + trim
      2. drop articles the/a/an (but **not** prepositions of/in/on/at -- those carry semantics)
      3. simple singularization (per-word)
      4. non [a-z0-9] -> hyphen, collapse, trim ends

    Examples:
      'Markov Property'             -> 'markov-property'
      'Markov_Property'             -> 'markov-property'
      'value function'              -> 'value-function'
      'value functions'             -> 'value-function'   (singularized)
      'model of environment'        -> 'model-of-environment'
      'model of the environment'    -> 'model-of-environment'  (drops 'the', merges!)
      'policies'                    -> 'policy'           (ies -> y)
      'Q-Learning'                  -> 'q-learning'
      'process'                     -> 'process'          (ss ending preserved)
    """
    s = label.strip().lower()
    # Drop articles (keep prepositions of/in/on/at -- they are semantic)
    s = _ARTICLE_RE.sub("", s)
    # Per-word singularization (_singularize_word already skips non-ASCII)
    words = [w for w in s.split() if w]
    words = [_singularize_word(w) for w in words]
    s = " ".join(words)
    # Non-\w (letters/digits/_, Unicode-aware) -> hyphen. \w in Py3 includes CJK by default,
    # so Chinese labels '策略' / '价值函数' will not collapse to empty slugs (2026-05-21 multilingual KG fix).
    s = re.sub(r"[^\w]+", "-", s)
    s = s.strip("-")
    return s


def build_external_id(document_id: int, chapter_id: str, label: str) -> str:
    """Build external_id = '<doc_id>_<chap_id>_<slug>'."""
    return f"{document_id}_{chapter_id}_{slugify(label)}"


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """你是一个学习助理。给你一段学习笔记(Markdown),你的任务是抽出其中的核心概念、方法、例子,以及它们之间的关系。

# Node 三种类型(枚举值严格)

**concept**(定义/定理/抽象实体 —— 你"理解"的东西):
- 例:`policy`, `reward signal`, `value function`, `Markov property`, `state`, `action`, `agent`, `environment`

**method**(算法/技术/可执行过程 —— 你"运行"的东西):
- 例:`Q-learning`, `policy gradient`, `Monte Carlo method`, `temporal-difference learning`
- 也包括动名词形式的活动:`planning`, `value estimation`, `trial-and-error learning`

**example**(具体实例/命名场景 —— 单独的应用案例):
- 例:`Tic-Tac-Toe`, `Atari game`, `CartPole`, `Samuel's Checkers Player`

## 边界启发式(关键!)

某些词同时像 concept 和 method,用以下规则消歧:

| 测试 | 倾向 |
|---|---|
| 你能写"running X" 或 "applying X" 自然 | method |
| 你会写"X is defined as ..." | concept |
| X 是教材里一个具体命名的实例(独有名字) | example |
| 真的拿不准 → 默认 | concept |

具体例子:
- `planning` → method(可以"做 planning")
- `policy` → concept(是定义)
- `Q-learning` → method(明确是算法)
- `exploration` → concept(虽然像活动,但是抽象概念。**默认 concept**)
- `Tic-Tac-Toe` → example(具体名字)

# Label 规则(关键!)

1. **与教材原文语言一致**:英文教材 → 英文 label,中文教材 → 中文 label;**直接复用原文术语**,不要翻译、不要漂移
   - 英文教材例:`policy` / `reward signal` / `value function` / `Markov property`
   - 中文教材例:`策略` / `奖励信号` / `价值函数` / `马尔可夫性质` / `统计学习` / `监督学习` / `损失函数`
2. **英文 label**:单数形式("policy" 不是 "policies")、不带冠词("value function" 不是 "the value function")、普通名词全小写、人名 capitalize
3. **中文 label**:直接照搬术语,无单数/冠词/大小写概念
4. **2-4 个词(英文)或 2-8 字(中文),简洁**
5. **不要在 label 里夹标点**(无需引号、括号、书名号)

# Edge 四种类型(枚举值严格)

**requires**(有向):source 是 target 的**严格前置概念**——不懂 target 就不能懂 source
- 例:`value function` → requires → `reward signal`(value 的定义依赖 reward)

**related_to**(无向):弱关联,同语境提及,无严格依赖
- 例:`agent` ↔ related_to ↔ `environment`

**contrasts_with**(无向):对立或对比,理解一方有助于理解另一方
- 例:`model-based method` ↔ contrasts_with ↔ `model-free method`
- **慎用**:只在教材明确做对比时用(如 "X versus Y", "X whereas Y")

**example_of**(有向):source 是 target 的具体**命名**实例(只用于 example node 出现时)
- 例:`Tic-Tac-Toe` → example_of → `value function`
- **方向是固定的**:具体 example → 抽象 concept,不能反

# 硬约束

1. **relations 里的 source_label / target_label 必须出现在 concepts 列表里**。不要引用 Note 没明确出现的概念。
2. **量化**:concepts 5-10 个(选最核心,不堆砌);relations 3-8 条(只抽教材**明确**陈述的关系,不要推断)
"""


USER_PROMPT_TEMPLATE = """**【最高优先级规则,在 system 规则之上】**
本 Note 的主语言 = **{main_lang}**。所有 `concepts.label` 与 `concepts.description` **必须使用该语言**,**不得混用**:
- 若 Note 原文是双语术语(如 "统计学习 (statistical learning)" / "策略(policy)"),且主语言 = 中文 → label 取 **中文形式** "统计学习" / "策略",不取括号内英文。
- 若主语言 = 英文 → label 取英文形式。
- 同一抽取里不可有的 label 是中文有的是英文。

这是要抽取的 Note:

```markdown
{note_content}
```
{context_block}
按上面的规则抽取 concepts 和 relations。"""


def _detect_main_language(text: str) -> str:
    """Detect a note's primary language by character ratio. CJK > ASCII letters -> Chinese, else English.
    Used to inject a deterministic language lock into USER_PROMPT (prevents LLM English-anchoring on bilingual terms)."""
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    if cjk == 0 and ascii_letters == 0:
        return "中文"            # Fallback default: Chinese (most of our test samples are Chinese textbooks)
    return "中文" if cjk > ascii_letters else "英文"


CONTEXT_BLOCK_TEMPLATE = """

**已有 concepts 上下文**(若本 Note 涉及这些概念,**复用完全相同的 label** 而不是另起新名,以便跨章节去重):
{existing_labels}
"""


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def extract_kg_from_note(
    note_id: int,
    *,
    model_name: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
    context_chapter_depth: int = 3,
    context_label_limit: int = 30,
) -> KGExtraction:
    """Extract KG from a single Note. Returns KGExtraction (not yet persisted -- W3-5 orchestrator handles DB writes).

    Args:
        note_id: Note.id to process
        model_name: which model to use (default sonnet; switch to haiku to save cost)
        max_tokens: LLM max_tokens
        context_chapter_depth: feed existing concepts from the most recent N chapters as dedup context;
            0 = do not feed (clean extraction, used for stability tests)
        context_label_limit: max number of labels to include in context (prevents prompt token blow-up)

    Returns:
        KGExtraction (concepts + relations, Pydantic-validated)
    """
    db = SessionLocal()
    try:
        note = db.get(Note, note_id)
        if note is None:
            raise ValueError(f"Note {note_id} not found")

        context_block = ""
        if context_chapter_depth > 0:
            # Take existing concept labels from the same document (id desc for newest first)
            existing = (
                db.query(KGNode)
                .filter(KGNode.document_id == note.document_id)
                .order_by(KGNode.id.desc())
                .limit(context_label_limit * 3)  # extra headroom for dedup
                .all()
            )
            seen: set[str] = set()
            lines: list[str] = []
            for n in existing:
                if n.label in seen:
                    continue
                seen.add(n.label)
                lines.append(f"- {n.label} ({n.type})")
                if len(lines) >= context_label_limit:
                    break
            if lines:
                context_block = CONTEXT_BLOCK_TEMPLATE.format(
                    existing_labels="\n".join(lines)
                )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            note_content=note.content_md,
            context_block=context_block,
            main_lang=_detect_main_language(note.content_md),
        )
    finally:
        db.close()

    model = ChatAnthropic(
        model=model_name,
        max_tokens=max_tokens,
        api_key=settings.anthropic_api_key,
    )
    structured = model.with_structured_output(KGExtraction)
    # cache_control enables Anthropic prompt caching:
    # SYSTEM_PROMPT (~2000 tokens) is the main repeated input for KG extraction;
    # 8 Notes extracted in sequence (within 5 min) -> one cache_creation, subsequent cache_read at 0.1x price
    result: KGExtraction = structured.invoke([
        SystemMessage(content=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }]),
        HumanMessage(content=user_prompt),
    ])

    # Second-pass server-side validation: each relation's source/target must appear in concepts
    valid_labels = {c.label for c in result.concepts}
    bad_relations = [
        r for r in result.relations
        if r.source_label not in valid_labels or r.target_label not in valid_labels
    ]
    if bad_relations:
        # Do not raise; filter them out (the LLM occasionally references labels not in concepts).
        # A silent drop at the extraction layer is simpler than retrying; W3-5 logs every drop.
        result.relations = [r for r in result.relations if r not in bad_relations]
        # Leave a marker field so the caller can see what was dropped (dynamic attr; not Pydantic-schema'd)
        result.__dict__["_dropped_relations"] = bad_relations

    return result


# --------------------------------------------------------------------------- #
# O4: note_ref_id integrity gate (defense against SQLite id-reuse misalignment; section 30 O4)
# --------------------------------------------------------------------------- #

# Max cosine between label and Note headings >= this value = that Note substantively covers the concept.
# Threshold informed by Phase 3-2 A1 measured score ladder (>=0.725 all correct / <=0.518 all wrong, grey in between). 0.55 is strict:
# O4 prefers AMBIGUOUS (escalate to human) over a misclassified origin that auto-fixes. **Until calibrated on real violations,
# realign's apply defaults to False** (silent-drift discipline; current audit shows 0 violations so no calibration data yet).
ORIGIN_ORACLE_MIN_SIM = 0.55


class NoteRefIntegrityError(ValueError):
    """KGNode.note_ref_id violates the invariant (chapter misalignment / dangling). Used to early-abort backfill pre-flight."""


@dataclass
class NoteRefViolation:
    node_id: int
    label: str
    node_chapter: str
    ref_note_id: int
    ref_note_chapter: str | None      # None = dangling (ref id has no matching Note)
    sim_ref: float                    # label's oracle hit against ref-Note (dangling -> -1.0)
    sim_chap: float                   # label's hit against the Note that owns node.chapter (no such Note -> -1.0)
    chap_note_id: int | None          # current Note id corresponding to node.chapter (None if absent)
    origin: str                       # 'A_idreuse'|'B_relabel'|'AMBIGUOUS'|'NO_TARGET'


@dataclass
class ReAlignReport:
    fixed: list[int]                  # node ids whose note_ref was actually modified (only A_idreuse)
    needs_human: list[NoteRefViolation]   # B/AMBIGUOUS/NO_TARGET, data left untouched


def _default_oracle(note_content_md: str, label: str, model) -> float:
    """Max cosine between label and Note headings. Reuses the same local embedding as Phase 3-2 A1."""
    from sentence_transformers import util
    heads = [h for h, _ in parse_markdown_sections(note_content_md)]
    if not heads:
        return -1.0
    el = model.encode(label, convert_to_tensor=True)
    eh = model.encode(heads, convert_to_tensor=True)
    return float(util.cos_sim(el, eh)[0].max())


def classify_note_ref_violations(db, document_id, *, oracle=None, model=None):
    """Sole classification entry point. Both validate (detect) and realign (fix) call ONLY this --
    single independent signal source, preventing the two paths from drifting after computing
    their own (section 29 #2 lesson: shared classification is a soundness prerequisite, not OCD).

    document_id=None: mirror backfill's whole-DB semantics; the bypass is sealed inside this function (B2 defense in depth).
    oracle: injectable (note_md, label) -> float; None uses _default_oracle (tests inject deterministic stubs).
    """
    if document_id is None:
        # Delta-3: build the model once at the top of the None-branch; do not let each sub-call rebuild ~80MB MiniLM
        if oracle is None and model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
        doc_ids = [r[0] for r in db.query(KGNode.document_id).distinct()]
        out: list[NoteRefViolation] = []
        for d in doc_ids:
            out.extend(classify_note_ref_violations(db, d, oracle=oracle, model=model))
        return out

    if oracle is None:
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
        oracle = lambda md, lab: _default_oracle(md, lab, model)  # noqa: E731

    notes = {
        n.id: n
        for n in db.query(Note).filter(Note.document_id == document_id)
    }
    # Multiple Notes per chapter: pick max id. **Deliberate choice, NOT "to match backfill"** (in practice there is no
    # canonical-Note concept: backfill follows FK without picking a Note, routes_domain order desc is just for list display). Rationale:
    # A_idreuse nodes already have a broken ref; swapping to any Note of the same chapter is an improvement and does not affect chapter correctness
    # (any same-chapter Note is a legal home); which Note/section of that chapter we land on is determined by max-id,
    # which is an acceptable policy choice (max-id = the most recent study_book run for that chapter). ch1.3 has 8 Notes/chapter observed.
    chap_note = {}
    for n in sorted(notes.values(), key=lambda x: x.id):
        chap_note[n.chapter_id] = n   # Ascending order + overwrite-on-write = final value is max id

    out = []
    for node in db.query(KGNode).filter(KGNode.document_id == document_id):
        if node.note_ref_id is None:
            continue
        ref = notes.get(node.note_ref_id)
        if ref is not None and ref.chapter_id == node.chapter_id:
            continue   # Invariant holds, not a violation

        chap_n = chap_note.get(node.chapter_id)
        sim_ref = oracle(ref.content_md, node.label) if ref is not None else -1.0
        sim_chap = oracle(chap_n.content_md, node.label) if chap_n is not None else -1.0

        if chap_n is None:
            origin = "NO_TARGET"
        else:
            T = ORIGIN_ORACLE_MIN_SIM
            ref_hit, chap_hit = sim_ref >= T, sim_chap >= T
            if not ref_hit and chap_hit:
                origin = "A_idreuse"
            elif ref_hit and not chap_hit:
                origin = "B_relabel"
            else:
                origin = "AMBIGUOUS"
        out.append(NoteRefViolation(
            node.id, node.label, node.chapter_id, node.note_ref_id,
            ref.chapter_id if ref is not None else None,
            sim_ref, sim_chap, chap_n.id if chap_n is not None else None, origin,
        ))
    return out


def validate_note_refs(db, document_id, *, oracle=None, model=None) -> None:
    """Detect-only. Raises if any violation exists; the report includes origin + both-side sim evidence.
    Does not classify-and-fix, does not mutate data -- gate only shouts, never patches the KG (same discipline as Eval 1)."""
    vs = classify_note_ref_violations(db, document_id, oracle=oracle, model=model)
    if not vs:
        return
    lines = [
        f"  node={v.node_id} {v.label!r} chapter={v.node_chapter} "
        f"ref→note{v.ref_note_id}(chapter={v.ref_note_chapter}) "
        f"sim_ref={v.sim_ref:.3f} sim_chap={v.sim_chap:.3f} → {v.origin}"
        for v in vs
    ]
    raise NoteRefIntegrityError(
        f"{len(vs)} 个 note_ref 违反不变量(chapter==ref-Note.chapter):\n"
        + "\n".join(lines)
        + "\n→ 跑 scripts/audit_note_refs.py --fix(仅 A_idreuse 自动修,余交人)")


def realign_note_refs(db, document_id, *, apply=False, oracle=None, model=None) -> ReAlignReport:
    """Only origin==A_idreuse: when apply=True, re-derive note_ref to the Note that owns the chapter,
    and clear note_anchor_slug / cross_note_* (forces backfill to recompute, non-destructive ratchet).
    B_relabel / AMBIGUOUS / NO_TARGET: data left untouched, queued into needs_human.

    [No commit]: this DB mutator does not hold transaction boundaries; commit is the caller's responsibility (audit --fix commits explicitly).
    apply defaults to False: until the threshold is calibrated, only dry-run (the first real violation = calibration trigger).
    """
    vs = classify_note_ref_violations(db, document_id, oracle=oracle, model=model)
    fixed, human = [], []
    for v in vs:
        if v.origin == "A_idreuse":
            if apply:
                node = db.get(KGNode, v.node_id)
                node.note_ref_id = v.chap_note_id
                node.note_anchor_slug = None
                node.cross_note_id = None
                node.cross_note_slug = None
                fixed.append(v.node_id)
        else:
            human.append(v)
    db.flush()   # Make mutations visible to caller; commit is caller's job
    return ReAlignReport(fixed=fixed, needs_human=human)
