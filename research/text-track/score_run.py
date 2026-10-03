"""Text-track scorer: one row of metrics per JevBench v1 run directory (or results.jsonl).

Adds to evaluation/jevbench/v15_proxy.py: the share of yes/no answers in the v1.5
abstention band (0.2 < P(yes) < 0.8), band answers per family, the "yes" share, and
Brier / ECE (from JevBench's own summary.json when present, else recomputed the same
way: Brier = mean over tasks of sum_k (p_k - 1[k = gold])^2 is not recoverable without
gold for wrong answers, so only summary.json values are reported as Brier).

    python research/text-track/score_run.py RUN_DIR_OR_RESULTS [...]  [--json out.json]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "evaluation", "jevbench"))
from v15_proxy import kind, score  # noqa: E402


def ece_from_rows(rows: list[dict], bins: int = 10) -> float:
    """Top-probability ECE, 10 equal bins (JevBench's binning)."""
    b: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for r in rows:
        p = r.get("probs") or {}
        if not p:
            continue
        conf = max(p.values())
        b[min(bins - 1, int(conf * bins))].append((conf, bool(r.get("correct"))))
    n = sum(len(x) for x in b)
    return sum(abs(sum(c for c, _ in x) / len(x) - sum(k for _, k in x) / len(x)) * len(x)
               for x in b if x) / max(n, 1)


def run(path: str) -> dict:
    d = path if os.path.isdir(path) else os.path.dirname(path)
    res = os.path.join(d, "results.jsonl") if os.path.isdir(path) else path
    rows = [json.loads(line) for line in open(res) if line.strip()]
    out = score(res)
    yn = [r for r in rows if kind(r) == "noul"]
    band = [r for r in yn if 0.2 < (r.get("probs") or {}).get("yes", 0.5) < 0.8]
    out["noul_n"] = len(yn)
    out["noul_correct"] = sum(bool(r["correct"]) for r in yn)
    out["band_share"] = round(len(band) / max(1, len(yn)), 3)
    out["band_correct"] = sum(bool(r["correct"]) for r in band)
    out["band_by_family"] = dict(Counter(r["family"] for r in band).most_common())
    out["yes_share"] = round(sum(r.get("predicted") == "yes" for r in yn) / max(1, len(yn)), 3)
    out["choice_correct"] = sum(bool(r["correct"]) for r in rows if kind(r) == "choice")
    out["score_correct"] = sum(bool(r["correct"]) for r in rows if kind(r) == "score")
    out["ece_top"] = round(ece_from_rows(rows), 4)
    sp = os.path.join(d, "summary.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        out["n_correct"] = s.get("n_correct")
        out["brier"] = round(s.get("brier_mean", float("nan")), 4)
        e = s.get("ece")
        out["ece"] = round(e["ece"] if isinstance(e, dict) else e, 4)
        out["ordinal_mae"] = round(s.get("ordinal_mae", float("nan")), 4)
        pc = s.get("paraphrase_consistency") or {}
        out["paraphrase_consistency"] = round(pc.get("agreement", float("nan")), 3)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    jout = None
    if "--json" in args:
        i = args.index("--json")
        jout = args[i + 1]
        args = args[:i] + args[i + 2:]
    allr = {}
    for p in args:
        allr[p] = run(p)
        print(p, json.dumps(allr[p]))
    if jout:
        json.dump(allr, open(jout, "w"), indent=2)
