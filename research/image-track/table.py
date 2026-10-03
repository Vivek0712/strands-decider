"""Markdown results table over run directories (run.py outputs, optionally rescaled).

    python research/image-track/table.py name=dir [name=dir ...]
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import row  # noqa: E402

COLS = [("NB acc", "nb_acc"), ("NB G", "nb_g"), ("NB ECE", "nb_ece"), ("POPE acc", "pope_acc"),
        ("POPE Brier", "pope_brier"), ("POPE ECE", "pope_ece"), ("IJB exact", "ijb_exact"),
        ("IJB all", "ijb_all"), ("IJB ECE", "ijb_ece"), ("blind conf NB", "blind_nb_conf"),
        ("blind ECE NB", "blind_nb_ece"), ("blind conf POPE", "blind_pope_conf"),
        ("blind ECE POPE", "blind_pope_ece")]
FAMS = ["ArxivQA", "CLEVR-HOPE", "Geometry3K", "FinQA", "ScreenSpot", "Multimodal-Mind2Web"]


def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))


def main() -> None:
    rows = []
    for arg in sys.argv[1:]:
        name, d = arg.split("=", 1)
        r = row(d)
        r["run"] = name
        rows.append(r)
    print("| run | " + " | ".join(c for c, _ in COLS) + " |")
    print("|---" * (len(COLS) + 1) + "|")
    for r in rows:
        print(f"| {r['run']} | " + " | ".join(fmt(r.get(k)) for _, k in COLS) + " |")
    print()
    print("| run | " + " | ".join(FAMS) + " |")
    print("|---" * (len(FAMS) + 1) + "|")
    for r in rows:
        fam = r.get("ijb_fam", {})
        print(f"| {r['run']} | " + " | ".join(fmt(fam.get(f)) for f in FAMS) + " |")


if __name__ == "__main__":
    main()
