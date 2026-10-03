"""Re-score a JevBench v1 results.jsonl with the main v1.5 rules (an approximation).

v1.5's scorer is not public. This applies its documented rules to the v1 public tasks:
a yes/no answer with 0.20 < P(yes) < 0.80 is an abstention and counts wrong; accuracy is
chance-corrected per item; choice, yes/no and score count 1/3 each. It omits the tier
weights and the sealed half, so treat it as a private yardstick, not a board score.

    python evaluation/jevbench/v15_proxy.py results.jsonl [results2.jsonl ...]
"""
from __future__ import annotations

import json
import sys


def kind(r: dict) -> str:
    if r.get("ordinal_ev") is not None:
        return "score"
    return "noul" if set((r.get("probs") or {})) == {"yes", "no"} else "choice"


def score(path: str) -> dict:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    by: dict[str, list[float]] = {"choice": [], "noul": [], "score": []}
    raw: dict[str, list[bool]] = {"choice": [], "noul": [], "score": []}
    abst = 0
    for r in rows:
        k = kind(r)
        n = max(2, len(r.get("probs") or {}))
        right = bool(r.get("correct"))
        if k == "noul" and 0.2 < (r.get("probs") or {}).get("yes", 0.5) < 0.8:
            abst += 1
            right = False
        chance = 1.0 / n
        by[k].append(((1.0 if right else 0.0) - chance) / (1 - chance))
        raw[k].append(bool(r.get("correct")))
    per = {k: round(100 * sum(v) / len(v), 1) for k, v in by.items() if v}
    return {"tasks": len(rows), "v1_accuracy": round(sum(sum(v) for v in raw.values()) / len(rows), 3),
            "noul_abstentions": abst, "competence_by_type": per,
            "intelligence_proxy": round(sum(per.values()) / len(per), 1)}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, json.dumps(score(p)))
