"""Stage C gate: does unseen-v2 order systems as the board's sealed half does?

Reads a small CSV of official scores (`system,official`; official_v1.5.5.csv holds the
board's sealed-half Intelligence for the systems in systems.json) and one unseen-v2 run per
system (rank_check.py's OUT/<system>/, or `--run NAME=DIR`). Reports each system's unseen-v2
Intelligence, the Spearman correlation with the official order, and a 95% bootstrap interval
over items (every system rescored on the same resampled items). The gate, fixed in advance
(research-next/PLAN-FINAL.md): proceed only if rho >= 0.7. It checks ordering, not numbers.

    python evaluation/unseen_v2/spearman.py --official evaluation/unseen_v2/official_v1.5.5.csv \\
        --dir reports/rank [--run ours=reports/unseen_v2/v20-soup] [--split all] [--json out.json]

Systems in the CSV without a run, and runs without an official score, are listed and left out
of rho (a run without one, such as our own checkpoint, is still scored and shown).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unseen import score as v1
from unseen_v2 import score

GATE = 0.7


def ranks(x: np.ndarray) -> np.ndarray:
    """Ranks along the last axis, ties given their average rank."""
    order = np.argsort(x, axis=-1, kind="stable")
    r = np.empty_like(x, dtype=float)
    np.put_along_axis(r, order, np.arange(x.shape[-1], dtype=float) + 1, axis=-1)
    flat = x.reshape(-1, x.shape[-1])
    rf = r.reshape(-1, x.shape[-1])
    for row, rr in zip(flat, rf, strict=True):
        vals, inv, counts = np.unique(row, return_inverse=True, return_counts=True)
        if (counts > 1).any():
            sums = np.zeros(len(vals))
            np.add.at(sums, inv, rr)
            rr[:] = (sums / counts)[inv]
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman's rho of `a` (..., n) against `b` (n,): Pearson's r of their ranks."""
    ra, rb = ranks(a), ranks(b[None, :])[0]
    ra = ra - ra.mean(-1, keepdims=True)
    rb = rb - rb.mean()
    den = np.sqrt((ra**2).sum(-1) * (rb**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray((ra * rb).sum(-1) / den)


def official(path: str) -> dict[str, float]:
    with open(path, encoding="utf-8", newline="") as fh:
        return {r["system"].strip(): float(r["official"]) for r in csv.DictReader(fh)}


def compare(runs: dict[str, list[dict[str, Any]]], board: dict[str, float], resamples: int = 2000,
            seed: int = 0) -> dict[str, Any]:
    names = sorted(runs)
    mats = np.stack(score.aligned([runs[n] for n in names]))  # systems x items x columns
    ours = {n: float(score.measures(m.sum(0))["intelligence"]) for n, m in zip(names, mats, strict=True)}
    both = [i for i, n in enumerate(names) if n in board]
    out: dict[str, Any] = {"unseen_v2_intelligence": {n: round(v, 2) for n, v in ours.items()},
                           "official": {n: board[n] for n in names if n in board},
                           "missing_runs": sorted(set(board) - set(runs)), "n": len(both)}
    if len(both) < 3:
        out["rho"] = None
        return out
    ref = np.array([board[names[i]] for i in both])
    point = float(spearman(np.array([ours[names[i]] for i in both]), ref))
    rng = np.random.default_rng(seed)
    n = mats.shape[1]
    w = rng.multinomial(n, np.full(n, 1.0 / n), size=resamples).astype(np.float64)
    sums = np.einsum("bi,sic->bsc", w, mats[both])
    boot = spearman(score.measures(sums)["intelligence"], ref)
    lo, hi = np.quantile(boot[~np.isnan(boot)], [0.025, 0.975])
    out.update(rho=round(point, 3), ci95=[round(float(lo), 3), round(float(hi), 3)], gate=GATE,
               passes=point >= GATE)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Spearman of unseen-v2 Intelligence against official scores.")
    ap.add_argument("--official", required=True, help="CSV with columns system,official")
    ap.add_argument("--dir", help="rank_check.py's OUT: one subdirectory per system")
    ap.add_argument("--run", action="append", default=[], help="NAME=DIR: another run to score")
    ap.add_argument("--split", choices=("dev", "test", "all"), default="all")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    paths = {}
    if a.dir:
        for name in sorted(os.listdir(a.dir)):
            if os.path.exists(os.path.join(a.dir, name, "predictions.jsonl")):
                paths[name] = os.path.join(a.dir, name)
    for spec in a.run:
        name, _, path = spec.partition("=")
        paths[name] = path
    runs = {n: score.select(v1.load(p), a.split) for n, p in paths.items()}
    report = compare(runs, official(a.official), a.resamples)
    print(json.dumps(report, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
