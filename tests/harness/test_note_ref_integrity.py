"""O4 回归:note_ref id-复用错配 gate。

`db` fixture 来自 tests/conftest.py(临时库隔离,零触 app.db)。
oracle 注入确定性桩 → 完全控制 origin 落点,无需真 embedding 模型。
"""
import pytest

from sla.models.domain import Note
from sla.models.kg import KGNode
from sla.harness.kg import (
    classify_note_ref_violations,
    realign_note_refs,
    validate_note_refs,
    NoteRefIntegrityError,
)


def stub_oracle(mapping):
    """content_md 编码成 'NID:<marker>';按 (marker, label) 返回预设 sim。"""
    def _o(md, label):
        nid = int(md.split(":")[1])
        return mapping.get((nid, label), -1.0)
    return _o


def _mk(db, doc, chap, marker):
    n = Note(document_id=doc, chapter_id=chap, content_md=f"NID:{marker}")
    db.add(n)
    db.flush()
    return n


def test_idreuse_origin_A_autofixed(db):
    doc = 999001
    good = _mk(db, doc, "ch9.9", 1)       # tabular method 真家
    wrong = _mk(db, doc, "ch9.1", 2)      # 复用 id 撞来的异章 Note
    node = KGNode(external_id=f"{doc}_ch9.9_x", type="concept", label="tabular method",
                  document_id=doc, chapter_id="ch9.9", note_ref_id=wrong.id)
    db.add(node)
    db.flush()
    orc = stub_oracle({(1, "tabular method"): 0.9, (2, "tabular method"): 0.1})
    vs = classify_note_ref_violations(db, doc, oracle=orc)
    assert len(vs) == 1 and vs[0].origin == "A_idreuse"
    rep = realign_note_refs(db, doc, apply=True, oracle=orc)
    assert rep.fixed == [node.id] and not rep.needs_human
    db.refresh(node)
    assert node.note_ref_id == good.id and node.note_anchor_slug is None


def test_botched_relabel_origin_B_not_propagated(db):
    doc = 999002
    realhome = _mk(db, doc, "ch3.4", 11)  # node 真内容在这
    _mk(db, doc, "ch3.5", 12)             # 被误 relabel 指向的章
    node = KGNode(external_id=f"{doc}_ch3.5_x", type="concept", label="absorbing state",
                  document_id=doc, chapter_id="ch3.5", note_ref_id=realhome.id)
    db.add(node)
    db.flush()
    orc = stub_oracle({(11, "absorbing state"): 0.9, (12, "absorbing state"): 0.1})
    vs = classify_note_ref_violations(db, doc, oracle=orc)
    assert vs[0].origin == "B_relabel"
    rep = realign_note_refs(db, doc, apply=True, oracle=orc)
    assert rep.fixed == [] and [v.node_id for v in rep.needs_human] == [node.id]
    db.refresh(node)
    assert node.note_ref_id == realhome.id   # 未被动


def test_no_target_defined_branch(db):
    doc = 999003
    wrong = _mk(db, doc, "ch5.1", 21)
    node = KGNode(external_id=f"{doc}_ch5.9_x", type="concept", label="orphan concept",
                  document_id=doc, chapter_id="ch5.9", note_ref_id=wrong.id)  # ch5.9 无 Note
    db.add(node)
    db.flush()
    vs = classify_note_ref_violations(db, doc, oracle=stub_oracle({}))
    assert vs[0].origin == "NO_TARGET" and vs[0].chap_note_id is None
    rep = realign_note_refs(db, doc, apply=True, oracle=stub_oracle({}))
    assert rep.fixed == [] and [v.node_id for v in rep.needs_human] == [node.id]


def test_ambiguous_origin_reports_human_never_autofix(db):
    doc = 999004
    refn = _mk(db, doc, "ch2.1", 31)
    chapn = _mk(db, doc, "ch2.9", 32)
    node = KGNode(external_id=f"{doc}_ch2.9_x", type="concept", label="value",
                  document_id=doc, chapter_id="ch2.9", note_ref_id=refn.id)
    db.add(node)
    db.flush()
    orc = stub_oracle({(31, "value"): 0.9, (32, "value"): 0.9})  # 都命中
    vs = classify_note_ref_violations(db, doc, oracle=orc)
    assert vs[0].origin == "AMBIGUOUS"
    rep = realign_note_refs(db, doc, apply=True, oracle=orc)
    assert rep.fixed == [] and [v.node_id for v in rep.needs_human] == [node.id]


def test_validate_raises_with_evidence(db):
    doc = 999005
    w = _mk(db, doc, "ch1.2", 41)
    _mk(db, doc, "ch1.1", 42)
    node = KGNode(external_id=f"{doc}_ch1.1_x", type="concept", label="reward",
                  document_id=doc, chapter_id="ch1.1", note_ref_id=w.id)
    db.add(node)
    db.flush()
    orc = stub_oracle({(42, "reward"): 0.9, (41, "reward"): 0.1})
    with pytest.raises(NoteRefIntegrityError, match=r"A_idreuse"):
        validate_note_refs(db, doc, oracle=orc)


def test_None_document_id_fans_out_all_docs(db):
    # B2 回归锁:不带 --document-id(全库,生产常用路径)→ classify(db, None) 必须
    # fan-out 所有 doc 并各自抓到 violation。无此测试 = B2 blocker 修复仅靠读代码信。
    # 注入 stub oracle → None-branch 不进 SentenceTransformer(Δ3 路径纯净)。
    g1 = _mk(db, 999101, "ch1.9", 101)   # doc1 node 真章
    w1 = _mk(db, 999101, "ch1.1", 102)   # doc1 复用撞来的异章
    n1 = KGNode(external_id="999101_ch1.9_x", type="concept", label="alpha",
                document_id=999101, chapter_id="ch1.9", note_ref_id=w1.id)
    g2 = _mk(db, 999102, "ch2.9", 201)   # doc2 node 真章
    w2 = _mk(db, 999102, "ch2.1", 202)   # doc2 复用撞来的异章
    n2 = KGNode(external_id="999102_ch2.9_x", type="concept", label="beta",
                document_id=999102, chapter_id="ch2.9", note_ref_id=w2.id)
    db.add_all([n1, n2])
    db.flush()
    orc = stub_oracle({
        (101, "alpha"): 0.9, (102, "alpha"): 0.1,
        (201, "beta"): 0.9, (202, "beta"): 0.1,
    })
    vs = classify_note_ref_violations(db, None, oracle=orc)   # None = 全库 fan-out
    got_docs = {db.get(KGNode, v.node_id).document_id for v in vs}
    assert got_docs == {999101, 999102}                        # 两 doc 都被查到
    assert all(v.origin == "A_idreuse" for v in vs) and len(vs) == 2


def test_clean_db_no_violation(db):
    doc = 999006
    n = _mk(db, doc, "ch1.1", 51)
    node = KGNode(external_id=f"{doc}_ch1.1_x", type="concept", label="agent",
                  document_id=doc, chapter_id="ch1.1", note_ref_id=n.id)
    db.add(node)
    db.flush()
    assert classify_note_ref_violations(db, doc, oracle=stub_oracle({})) == []
    validate_note_refs(db, doc, oracle=stub_oracle({}))   # 不 raise
