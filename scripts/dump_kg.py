"""Dump the KG of a document as native JSON to a file or stdout.

Usage:
  python scripts/dump_kg.py --document-id 2 --output /tmp/sutton_kg.json
  python scripts/dump_kg.py --document-id 2                              # Defaults to stdout
  python scripts/dump_kg.py --document-id 2 --indent 4
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sla.harness.kg_export import to_native_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(prog="dump_kg")
    parser.add_argument("--document-id", type=int, required=True)
    parser.add_argument(
        "--output", default="-",
        help="输出路径('-' = stdout,默认 '-')",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON 缩进(默认 2)")
    args = parser.parse_args()

    payload = to_native_json(args.document_id)

    if args.output == "-":
        print(json.dumps(payload, indent=args.indent, ensure_ascii=False))
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=args.indent, ensure_ascii=False)
        stats = payload["stats"]
        print(f"wrote {args.output}")
        print(f"  document_id: {payload['document_id']}  title: {payload['document_title']!r}")
        print(f"  chapters:    {len(payload['chapters'])}  {payload['chapters']}")
        print(f"  nodes:       {stats['total_nodes']}  by type: {stats['nodes_by_type']}")
        print(f"  edges:       {stats['total_edges']}  by type: {stats['edges_by_type']}")


if __name__ == "__main__":
    main()
