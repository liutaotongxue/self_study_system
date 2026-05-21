"""KG 抽取(Phase 2-W3-4)—— 从单个 Note 抽出 concepts + relations。

**架构**:独立 service,不走 run_task pipeline。理由:
  - 单轮 LLM 调用,无多轮决策需求
  - run_task 的 Step/ToolCall trace + rule_evaluate 对 KG 抽取没意义
  - 直接 ChatAnthropic.with_structured_output(KGExtraction) 拿到 Pydantic 对象

**LLM 输出契约**:
  - concepts: list[{label, type ∈ {concept,method,example}, description}]
  - relations: list[{source_label, target_label, type ∈ {requires,related_to,contrasts_with,example_of}, notes?}]
  - **LLM 只给 label**,服务端用 slugify 生成 external_id(避免 LLM 在 slug 级别不一致)

**Context window**:抽某 Note 时,可选喂"已有 concepts label 列表"作为 dedup hint,
limit 到最近 30 个 label,防 prompt token 爆。
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
    """Phase 1B-1 同款防御 + LLM 尾垃圾容忍(实测 ch3.10 rebuild job2:
    合法 array 后多吐 '[\\n]'(=[])致 json.loads "Extra data" → build_kg rc=1)。
    严格失败 → raw_decode 取首个完整 JSON 值;尾守卫:仅当尾随【非空】
    list/dict(真·第二个有内容值)才 re-raise 保 loud 不静默丢抽取数据;
    空 []/{}/非JSON 垃圾 → 忽略。well-formed 路径零改动。"""
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
                    raise   # 尾随非空第二值:保持 loud,不静默丢数据
            return obj
    return v


class ConceptItem(BaseModel):
    """LLM 输出的单个概念。"""
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
    """LLM 输出的单个关系。"""
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
    """LLM 总输出结构。"""
    concepts: Annotated[
        list[ConceptItem], BeforeValidator(_parse_if_json_string)
    ] = Field(description="本 Note 抽出的 5-10 个 concepts/methods/examples")
    relations: Annotated[
        list[RelationItem], BeforeValidator(_parse_if_json_string)
    ] = Field(description="本 Note 抽出的 3-8 条关系")


# --------------------------------------------------------------------------- #
# 服务端工具:slug + external_id
# --------------------------------------------------------------------------- #

_ARTICLE_RE = re.compile(r"\b(?:the|a|an)\b", flags=re.IGNORECASE)


def _singularize_word(word: str) -> str:
    """轻量英文单数化:覆盖最常见复数模式,不追求完美。
    **非 ASCII 词(如中文 '策略' / '价值函数')直接原样返回**——
    其他语言无英文复数概念,套规则反而误伤(2026-05-21 多语言 KG 修)。

    英文规则:
      - 长度 ≤ 3:不动(避免短词误伤,如 'is', 'as')
      - 'ies' 结尾且 len > 4:替 'y'(policies → policy)
      - 'sses' 结尾:去 'es'(processes → process)
      - 'ss' 结尾:不动(process, class —— 已是单数)
      - 's' 结尾:去 's'(values → value, actions → action)
      - 其他:不动

    已知会误伤的:'axis' → 'axi', 'iris' → 'iri'。RL 域里不算高频,Phase 2+ 再调。
    """
    if not word.isascii():
        return word                       # 中文/CJK/其它非 ASCII:跳过单数化
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
    """把 Markdown 文本切成 [(heading, content), ...] 段。

    一个 section = 一个 #/##/###/#### heading 行 + 后续直到下一个 heading 之间的所有内容。
    没有 heading 之前的内容(preamble)忽略——Note 通常以 # 开头,无 preamble。

    例:
        '''
        # Ch1.3 Elements
        intro
        ## 1. Policy
        a policy is ...
        ## 2. Reward
        reward is ...
        '''
        → [('Ch1.3 Elements', 'intro\\n'),
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
    """Markdown heading → URL slug (GitHub-style)。**接受 raw 或已 strip 过 ## 的两种形式**。

    例:
      '## 2. Action-Value Methods'   → 'action-value-methods'
      '2. Action-Value Methods'      → 'action-value-methods'  (parse_markdown_sections 给的形式)
      '### Reward vs. Value'         → 'reward-vs-value'
      'TD Update Rule (Eq. 2.4)'     → 'td-update-rule-eq-2-4'
      '## 1.1 Reinforcement Learning'→ 'reinforcement-learning'

    规则:
      1. lowercase + strip
      2. 去掉 leading markdown ## 前缀(如果还在)
      3. 去掉 leading 数字编号(如 '1.', '2.3.')
      4. 非 [a-z0-9] → hyphen,折叠,去首尾
      5. 若结果为空(纯数字 heading),fallback 用原始字符串 hash
    """
    s = heading.lower().strip()
    s = re.sub(r"^#+\s+", "", s)            # 兼容 '## xxx' 形式
    s = re.sub(r"^[\d.]+\s*", "", s)        # 去 leading '2.' / '1.1.'
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    if not s:
        # 纯数字 heading 等极端 case:fallback 用原 heading 简单 normalize
        s = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-") or "unnamed"
    return s


def slugify_headings(headings: list[str]) -> list[str]:
    """对一组 heading 批量 slugify,collision 加 GitHub 同款 -2 / -3 后缀。

    单个 slugify_heading 没法识别"同一 Note 内 ε-Greedy vs Greedy" 这种 Unicode 差异
    被吃掉后的 slug 撞车。批量入口可以见全局,在第二次出现时改名。

    例:
      ['Greedy Action Selection', 'ε-Greedy Action Selection']
      → ['greedy-action-selection', 'greedy-action-selection-2']
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
    """label → slug:Phase 2-W3-4 升级版,**激进归一化以稳定去重**。

    步骤:
      1. lowercase + trim
      2. 砍冠词 the/a/an (但**不**砍介词 of/in/on/at —— 后者承载语义)
      3. 简单单数化(per-word)
      4. 非 [a-z0-9] → hyphen,折叠,去首尾

    例:
      'Markov Property'             → 'markov-property'
      'Markov_Property'             → 'markov-property'
      'value function'              → 'value-function'
      'value functions'             → 'value-function'   (单数化)
      'model of environment'        → 'model-of-environment'
      'model of the environment'    → 'model-of-environment'  (砍 the,合并!)
      'policies'                    → 'policy'           (ies → y)
      'Q-Learning'                  → 'q-learning'
      'process'                     → 'process'          (ss 结尾保留)
    """
    s = label.strip().lower()
    # 砍冠词 (保留介词 of/in/on/at,语义性的)
    s = _ARTICLE_RE.sub("", s)
    # 单词级单数化(_singularize_word 内已跳过非 ASCII)
    words = [w for w in s.split() if w]
    words = [_singularize_word(w) for w in words]
    s = " ".join(words)
    # 非 \w(letters/digits/_,Unicode-aware)→ hyphen。\w 在 Py3 默认含 CJK,
    # 故中文 label '策略' / '价值函数' 不会被吃成空 slug(2026-05-21 多语言 KG 修)。
    s = re.sub(r"[^\w]+", "-", s)
    s = s.strip("-")
    return s


def build_external_id(document_id: int, chapter_id: str, label: str) -> str:
    """组装 external_id = '<doc_id>_<chap_id>_<slug>'。"""
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
    """字符占比判 note 主语言。CJK > ASCII 字母 → 中文,反之 → 英文。
    用于 USER_PROMPT 注入 deterministic 语言锁(防 LLM 在双语术语处英文 anchoring)。"""
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    if cjk == 0 and ascii_letters == 0:
        return "中文"            # 兜底默认中文(我们多数测试样本是中文教材)
    return "中文" if cjk > ascii_letters else "英文"


CONTEXT_BLOCK_TEMPLATE = """

**已有 concepts 上下文**(若本 Note 涉及这些概念,**复用完全相同的 label** 而不是另起新名,以便跨章节去重):
{existing_labels}
"""


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #

def extract_kg_from_note(
    note_id: int,
    *,
    model_name: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
    context_chapter_depth: int = 3,
    context_label_limit: int = 30,
) -> KGExtraction:
    """对单个 Note 抽取 KG。返回 KGExtraction(还没落 DB —— W3-5 orchestrator 负责落)。

    Args:
        note_id: 要处理的 Note.id
        model_name: 用哪个 model(默认 sonnet,可换 haiku 省钱)
        max_tokens: LLM max_tokens
        context_chapter_depth: 喂最近 N 章的现有 concepts 作为 dedup context;
            0 = 不喂(纯净抽取,用于稳定性测试)
        context_label_limit: context 里最多塞多少个 label(防 prompt token 爆)

    Returns:
        KGExtraction(concepts + relations,Pydantic 校验过)
    """
    db = SessionLocal()
    try:
        note = db.get(Note, note_id)
        if note is None:
            raise ValueError(f"Note {note_id} not found")

        context_block = ""
        if context_chapter_depth > 0:
            # 取同 document 已有的 concept labels(按 id desc 拿最新的)
            existing = (
                db.query(KGNode)
                .filter(KGNode.document_id == note.document_id)
                .order_by(KGNode.id.desc())
                .limit(context_label_limit * 3)  # 留 dedup 余量
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
    # cache_control 启用 Anthropic prompt caching:
    # SYSTEM_PROMPT ~2000 tokens 是 KG 抽取的主要重复 input,
    # 8 个 Note 顺序抽取(5 min 内),cache_creation 一次,后续 cache_read 0.1× 价
    result: KGExtraction = structured.invoke([
        SystemMessage(content=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }]),
        HumanMessage(content=user_prompt),
    ])

    # 服务端二次校验:relation 的 source/target 必须在 concepts 里
    valid_labels = {c.label for c in result.concepts}
    bad_relations = [
        r for r in result.relations
        if r.source_label not in valid_labels or r.target_label not in valid_labels
    ]
    if bad_relations:
        # 不抛错,过滤掉(LLM 偶尔会引用不在 concepts 里的 label)。
        # 在抽取层用静默 drop 比 retry 简单,W3-5 会把所有 drop 记录到 log。
        result.relations = [r for r in result.relations if r not in bad_relations]
        # 留个埋点字段方便 caller 看(动态属性,Pydantic 模型上不强制 schema 化)
        result.__dict__["_dropped_relations"] = bad_relations

    return result


# --------------------------------------------------------------------------- #
# O4: note_ref_id 完整性 gate(SQLite id 复用错配防御;§30 O4)
# --------------------------------------------------------------------------- #

# label vs Note headings 最大 cosine ≥ 此值 = 该 Note 实质讲该 concept。
# 依据 Phase 3-2 A1 实测 score 阶梯(≥0.725 全对 / ≤0.518 全错,中间灰)。0.55 偏严:
# O4 宁可判 AMBIGUOUS 交人,也不误判 origin 去 auto-fix。**未在真实 violation 上标定前,
# realign 的 apply 默认 False**(silent-drift 纪律;当前审计 0 violation 故无标定数据)。
ORIGIN_ORACLE_MIN_SIM = 0.55


class NoteRefIntegrityError(ValueError):
    """KGNode.note_ref_id 违反不变量(chapter 错配 / dangling)。backfill pre-flight 早停用。"""


@dataclass
class NoteRefViolation:
    node_id: int
    label: str
    node_chapter: str
    ref_note_id: int
    ref_note_chapter: str | None      # None = dangling(ref id 无对应 Note)
    sim_ref: float                    # label 在 ref-Note 的 oracle 命中(dangling→ -1.0)
    sim_chap: float                   # label 在 node.chapter 对应 Note 的命中(无该 Note→ -1.0)
    chap_note_id: int | None          # node.chapter 当前对应 Note id(无→None)
    origin: str                       # 'A_idreuse'|'B_relabel'|'AMBIGUOUS'|'NO_TARGET'


@dataclass
class ReAlignReport:
    fixed: list[int]                  # 实际改了 note_ref 的 node id(仅 A_idreuse)
    needs_human: list[NoteRefViolation]   # B/AMBIGUOUS/NO_TARGET,未动数据


def _default_oracle(note_content_md: str, label: str, model) -> float:
    """label 与 Note headings 的最大 cosine。复用 Phase 3-2 A1 同款本地 embedding。"""
    from sentence_transformers import util
    heads = [h for h, _ in parse_markdown_sections(note_content_md)]
    if not heads:
        return -1.0
    el = model.encode(label, convert_to_tensor=True)
    eh = model.encode(heads, convert_to_tensor=True)
    return float(util.cos_sim(el, eh)[0].max())


def classify_note_ref_violations(db, document_id, *, oracle=None, model=None):
    """唯一判定入口。validate(detect)与 realign(fix)都【只】调它 —— 单一独立信号源,
    杜绝两边各算一份后漂移(§29 #2 教训:共享判定是 soundness 前提非洁癖)。

    document_id=None:镜像 backfill 全库语义,在函数内封口静默旁路(B2 防御纵深)。
    oracle: 可注入 (note_md, label) -> float;None 用 _default_oracle(测试注入确定性桩)。
    """
    if document_id is None:
        # Δ3:模型在 None-branch 提前建一次,别让每个子调用重建 ~80MB MiniLM
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
    # 同章多 Note 取 max id。**刻意选择,非"与 backfill 一致"**(实测无 canonical-Note
    # 概念:backfill 跟 FK 不挑 Note,routes_domain order desc 仅 list 展示)。理由:
    # A_idreuse 节点 ref 本就坏,换任一同章 Note 都是改善;不影响 chapter 正确性
    # (任一同章 Note 皆合法 home);落到该章哪个 Note 哪 section 由 max-id 决定,
    # 属可接受的策略选择(max-id = 该章最近一次 study_book run)。ch1.3 实测 8 Note/章。
    chap_note = {}
    for n in sorted(notes.values(), key=lambda x: x.id):
        chap_note[n.chapter_id] = n   # 升序后写覆盖 = 最终留 max id

    out = []
    for node in db.query(KGNode).filter(KGNode.document_id == document_id):
        if node.note_ref_id is None:
            continue
        ref = notes.get(node.note_ref_id)
        if ref is not None and ref.chapter_id == node.chapter_id:
            continue   # 不变量成立,非 violation

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
    """detect-only。有 violation 即 raise,报告带 origin + 双侧 sim 证据。
    不分类去修、不动数据 —— gate 只喊不擅改 KG(Eval 1 同纪律)。"""
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
    """仅 origin==A_idreuse 在 apply=True 时重派生 note_ref→chapter 对应 Note,
    并清 note_anchor_slug/cross_note_*(强制 backfill 重算,非破坏 ratchet)。
    B_relabel/AMBIGUOUS/NO_TARGET:不动数据,进 needs_human。

    【不 commit】:库 mutator 不持事务边界,commit 归调用方(audit --fix 后显式)。
    apply 默认 False:阈值未标定前只 dry-run(首个真实 violation = 标定触发点)。
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
    db.flush()   # 让 caller 可见改动;commit 归 caller
    return ReAlignReport(fixed=fixed, needs_human=human)
