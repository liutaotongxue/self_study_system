"""O4: note_ref integrity audit and explicit repair.

  python scripts/audit_note_refs.py --document-id 2          # Report only (with origin classification)
  python scripts/audit_note_refs.py --document-id 2 --fix    # Auto-fix A_idreuse only; list the rest for human review
  python scripts/audit_note_refs.py                          # Audit the whole DB (document_id=None)

Only A_idreuse violations are automatically re-derived. B_relabel,
AMBIGUOUS, and NO_TARGET cases are listed for human adjudication.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sla.db import SessionLocal  # noqa: E402
from sla.harness.kg import (  # noqa: E402
    classify_note_ref_violations,
    realign_note_refs,
)


def main():
    p = argparse.ArgumentParser(prog="audit_note_refs")
    p.add_argument("--document-id", type=int, default=None,
                   help="指定 document;省略 = 全库")
    p.add_argument("--fix", action="store_true",
                   help="auto-fix 仅 A_idreuse;B/AMBIGUOUS/NO_TARGET 只列出")
    a = p.parse_args()

    db = SessionLocal()
    try:
        vs = classify_note_ref_violations(db, a.document_id)
        if not vs:
            print("[ok] note_ref 全部一致,0 violation")
            return
        print(f"=== {len(vs)} violations,origin: {dict(Counter(v.origin for v in vs))} ===")
        for v in vs:
            print(f"  node={v.node_id} {v.label!r} {v.node_chapter} "
                  f"ref→note{v.ref_note_id}({v.ref_note_chapter}) "
                  f"sim_ref={v.sim_ref:.3f} sim_chap={v.sim_chap:.3f} → {v.origin}")

        rep = realign_note_refs(db, a.document_id, apply=a.fix)
        if a.fix and rep.fixed:
            db.commit()   # Δ1:库函数不自持 commit,调用方显式
        print(f"\nfixed(A_idreuse, {'已写库' if a.fix else 'dry-run'}): {rep.fixed}")
        print(f"needs_human(不自动动): {[v.node_id for v in rep.needs_human]}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
