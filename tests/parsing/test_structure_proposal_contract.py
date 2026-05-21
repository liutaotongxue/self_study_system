r"""1c fork①:结构提议【契约回归锁】(CI,非 runtime gate)。

钉死 doc=2(S&B)当前【已验证-好】的 LLM 结构抽取 + 确定性 gate 结果。
未来改 structure_gate / chapter_detect 的 _SECTION_TOKEN_RE / 抽取链 若令
通过-gate 的结构【漏掉启发式能独立检出的节】(§29 类)、或破坏 well-formedness、
或破坏正文-对照,则 CI 在 ship 前红 —— 比 runtime gate 早且便宜,且因不接
ingest/不在生产路径 → 零 §29-误杀-生产风险(contract-lock 纪律)。

【为何是 CI 锁而非 runtime 硬门】fork① 实测:doc=2 上 B-1=90/90、
heur−提议=0(零漏节)、S_body−提议 全是数值噪声 → B-2 硬运行时门当前
抓不到任何可达 bug,却已实证 lexical 去噪有真 false-fail 爆炸面(同行-alpha
误杀 8 真节,因 S&B 标题在号的下一行)。故 B-2 不升 runtime 门,改本契约锁。

【fixture 真实性】tests/parsing/fixtures/doc2_run40_proposal.json =
ToolCall.input.sections 逐字捕获(run 40, doc=2),真数据、非反推成绿。

【已命名残留 / 勿当已解决】
- 锁绑 run-40 快照;重抽取(不同 run)字符串可能略异 —— 本锁钉的是
  "提议 ⊇ 启发式" 的契约不变量,非某 run 精确字符串。
- 提议−heur 的 10 个 ch14/15 节 = LLM 读全本 TOC 比启发式多,预期,不锁。
- 层C offset 恒定 = WARN-only(B1-否决,§29 真实污染失效史);本锁不碰。
- 跨语料(非 S&B / 中文)未验,归 Phase-1 残留,本锁不覆盖。

§29(漏节)实质在此 fire = 形态 A 的 ⊇ 断言;正交于 O3/Eval-1。
"""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from sla.parsing.pdf import extract_pages
from sla.parsing.chapter_detect import detect_sections
from sla.parsing.structure_gate import verify_structure_proposal

PDF = "/Users/liutao/Books/SuttonBartoIPRLBook2ndEd.pdf"
_FIX = Path(__file__).parent / "fixtures" / "doc2_run40_proposal.json"


@pytest.fixture(scope="session")
def real_pages():
    return extract_pages(PDF)


@pytest.fixture(scope="session")
def proposal():
    return json.loads(_FIX.read_text())["sections"]


@pytest.fixture(scope="session")
def heur_ids(real_pages):
    return {s.section_id for s in detect_sections(real_pages)}


# --- 契约锁:doc=2 已验证-好结构,改坏抽取/gate → CI 红 ---

def test_contract_layerA_zero_violation(real_pages, proposal):
    """well-formedness 锁:90 提议全 7 字段/两级/chapter_id==ch+id/页序合法。"""
    rep = verify_structure_proposal(real_pages, proposal)
    assert rep.layer_a_violations == []


def test_contract_b1_full_body_presence(real_pages, proposal):
    """B-1 锁:每个提议节 token 在其声称 pdf 页±1 独立出现(100%)。
    破坏 _SECTION_TOKEN_RE / offset / 抽取 → 命中率掉 → CI 红。"""
    rep = verify_structure_proposal(real_pages, proposal)
    assert rep.b1_total == rep.n_proposed
    assert rep.b1_hit == rep.b1_total, rep.b1_misses[:10]


def test_contract_sec29_superset_of_heuristic(proposal, heur_ids):
    """§29 漏节锁(形态 A,实测当前绿):LLM 提议集 ⊇ 确定性
    detect_sections(真PDF) 集 —— LLM 不得漏任何启发式独立检出的节。"""
    prop_ids = {str(s["section_id"]) for s in proposal}
    missing = sorted(heur_ids - prop_ids)
    assert missing == [], f"§29 回归:LLM 漏掉启发式检出的节 {missing}"


def test_contract_nothing_proposed_absent_from_body(real_pages, proposal):
    """提议−S_body=0:不得提议正文 token 扫不到的节。"""
    rep = verify_structure_proposal(real_pages, proposal)
    assert rep.extra_vs_body == []


# --- 失败类:证明这锁真能咬回归(非 green-always)---

def test_sec29_lock_bites_on_dropped_heuristic_section(proposal, heur_ids):
    victim = sorted(heur_ids)[0]
    shrunk = {str(s["section_id"]) for s in proposal
              if str(s["section_id"]) != victim}
    assert not (heur_ids <= shrunk)        # 漏一个启发式节 → ⊇ 破 → 锁 fire


def test_b1_lock_bites_on_corrupted_pdf_page(real_pages, proposal):
    bad = deepcopy(proposal)
    bad[0]["pdf_page_start"] = 999999       # 把首节 pdf 页改到不可能处
    rep = verify_structure_proposal(real_pages, bad)
    assert rep.b1_hit < rep.b1_total        # 命中率掉 → 锁 fire


def test_layerA_lock_bites_on_non_two_level(real_pages, proposal):
    bad = deepcopy(proposal)
    bad[0]["section_id"] = "1"              # 章级非两级 → Hole A 范围墙
    bad[0]["chapter_id"] = "ch1"
    rep = verify_structure_proposal(real_pages, bad)
    assert rep.layer_a_violations != []
