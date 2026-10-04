"""Grade answers on unseen-v2, and compare runs item by item with bootstrap intervals.

The grading rules are v1's (evaluation/unseen/score.py, which takes them from
evaluation/jevbench/v15_proxy.py): yes/no answers inside (0.2, 0.8) count wrong, yes/no and
choice credit is chance-corrected, score is graded by the expected level against chance,
and Intelligence is the mean of the three competences. A local measure, not a JevBench score.

v2 adds what the Stage D decision rules read (research/next/STAGE_A.md):

  - continuous proper scores beside the thresholded competences: NLL of the gold option,
    yes/no Brier, and the score type's ranked probability score,
    RPS = (1/(K-1)) * sum_k (F_k - 1[gold <= k])^2 over the K-1 cumulative levels;
  - the yes/no band mass (the share of yes/no answers inside the band) and the
    top-probability ECE;
  - `--split dev|test|all`;
  - paired comparisons: every measure is a function of per-item sums, so a bootstrap
    replicate is a reweighting of the items, shared by every run (paired by item). With
    `--resample-runs` each replicate also redraws the runs of each arm with replacement
    (a block bootstrap over items x seeds), which is what a seed-to-seed spread needs.

    python evaluation/unseen_v2/score.py RUN [RUN ...] [--vs BASE ...] [--split dev] [--json out.json]

A run is evaluation/unseen/run.py's output directory (or its predictions.jsonl). Needs numpy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jevbench.v15_proxy import BAND, credit, level_errors
from unseen import score as v1

Row = dict[str, Any]
KINDS = ("noul", "choice", "score")
BINS = 10
EPS = 1e-12
# Feature columns, one row per item (see `features`).
COLS = ["is_noul", "is_choice", "is_score", "credit_noul", "credit_choice", "err", "chance", "nll", "brier",
        "band", "rps"]
C = {c: i for i, c in enumerate(COLS)}


def rps(probs: list[float], gold: int) -> float:
    """Ranked probability score, normalised by K-1 (0 best, 1 worst); levels in order."""
    k = len(probs)
    total = sum(probs)
    cdf, out = 0.0, 0.0
    for level in range(k - 1):
        cdf += probs[level] / total
        out += (cdf - (1.0 if gold <= level else 0.0)) ** 2
    return out / (k - 1)


def features(rows: list[Row]) -> np.ndarray:
    """Per-item columns (COLS, then BINS confidence sums and BINS right counts for the ECE)
    whose sums give every measure."""
    out = np.zeros((len(rows), len(COLS) + 2 * BINS))
    for i, r in enumerate(rows):
        p, g, kind = r["probs"], r["gold"], r["kind"]
        f = out[i]
        f[C["is_" + kind]] = 1.0
        if kind == "score":
            f[C["err"]], f[C["chance"]] = level_errors(p, g)
            f[C["rps"]] = rps(p, g)
        else:
            f[C["credit_" + kind]] = credit(v1.right(r), len(p))
        if kind == "noul":
            f[C["brier"]] = (p[1] - g) ** 2
            f[C["band"]] = float(BAND[0] < p[1] < BAND[1])
        f[C["nll"]] = -math.log(max(p[g] / sum(p), EPS))
        conf = max(p)
        b = min(BINS - 1, int(conf * BINS))
        f[len(COLS) + b] = conf
        f[len(COLS) + BINS + b] = float(max(range(len(p)), key=lambda j: p[j]) == g)
    return out


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def measures(sums: np.ndarray) -> dict[str, np.ndarray]:
    """Every reported measure from item sums `sums[..., column]` (any leading shape)."""
    s = {c: sums[..., i] for c, i in C.items()}
    n = s["is_noul"] + s["is_choice"] + s["is_score"]
    comp = {
        "noul": 100 * _ratio(s["credit_noul"], s["is_noul"]),
        "choice": 100 * _ratio(s["credit_choice"], s["is_choice"]),
        "score": 100 * (1 - _ratio(s["err"], s["chance"])),
    }
    present = [v for k, v in comp.items() if np.all(s["is_" + k] > 0)]
    conf, right = sums[..., len(COLS):len(COLS) + BINS], sums[..., len(COLS) + BINS:]
    return {
        "intelligence": sum(present) / len(present) if present else np.zeros_like(n),
        **{f"competence_{k}": v for k, v in comp.items()},
        "nll": _ratio(s["nll"], n),
        "brier_yes_no": _ratio(s["brier"], s["is_noul"]),
        "band_mass": _ratio(s["band"], s["is_noul"]),
        "rps_score": _ratio(s["rps"], s["is_score"]),
        "ece_top": _ratio(np.abs(conf - right).sum(axis=-1), n),
    }


def summary(rows: list[Row]) -> dict[str, Any]:
    """v1's summary, the v2 measures, and the measures per family."""
    out = v1.summary(rows)
    m = measures(features(rows).sum(axis=0))
    out.update({k: (None if np.isnan(v) else round(float(v), 4)) for k, v in m.items()})
    return out


def select(rows: list[Row], split: str) -> list[Row]:
    if split == "all":
        return rows
    if any("split" not in r for r in rows):
        raise ValueError("these answers carry no split: answer unseen-v2 rows with evaluation/unseen/run.py")
    return [r for r in rows if r["split"] == split]


def aligned(runs: list[list[Row]]) -> list[np.ndarray]:
    """Each run's item matrix, rows in one shared id order; every run must answer the same items."""
    ids = sorted(r["id"] for r in runs[0])
    out = []
    for rows in runs:
        by = {r["id"]: r for r in rows}
        if sorted(by) != ids:
            raise ValueError("the runs do not answer the same items")
        out.append(features([by[i] for i in ids]))
    return out


def paired_bootstrap(arm: list[list[Row]], base: list[list[Row]], resamples: int = 10_000, seed: int = 0,
                     resample_runs: bool = False, chunk: int = 250) -> dict[str, dict[str, Any]]:
    """Mean over `arm` runs minus mean over `base` runs of every measure, with a 95%
    interval: each replicate reweights the items (multinomial counts, the same for every
    run), and with `resample_runs` also redraws each side's runs with replacement."""
    mats = aligned(arm + base)
    x = np.stack(mats)  # runs x items x columns
    na, n = len(arm), x.shape[1]

    def diff(sums: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> dict[str, np.ndarray]:
        m = measures(sums)  # leading shape: (replicates, runs)
        return {k: np.take_along_axis(v, ia, -1).mean(-1) - np.take_along_axis(v, ib, -1).mean(-1)
                for k, v in m.items()}

    every_a = np.arange(na)[None, :]
    every_b = np.arange(na, x.shape[0])[None, :]
    point = {k: float(v[0]) for k, v in diff(x.sum(axis=1)[None], every_a, every_b).items()}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[np.ndarray]] = {k: [] for k in point}
    done = 0
    while done < resamples:
        b = min(chunk, resamples - done)
        w = rng.multinomial(n, np.full(n, 1.0 / n), size=b).astype(np.float64)  # replicates x items
        sums = np.einsum("bi,ric->brc", w, x)
        if resample_runs:
            ia = rng.integers(0, na, size=(b, na))
            ib = rng.integers(na, x.shape[0], size=(b, x.shape[0] - na))
        else:
            ia, ib = np.repeat(every_a, b, 0), np.repeat(every_b, b, 0)
        for k, v in diff(sums, ia, ib).items():
            draws[k].append(v)
        done += b
    out = {}
    for k, v in point.items():
        d = np.concatenate(draws[k])
        d = d[~np.isnan(d)]
        if math.isnan(v) or not d.size:
            continue
        lo, hi = np.quantile(d, [0.025, 0.975])
        out[k] = {"diff": round(v, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                  "excludes_0": bool(lo > 0 or hi < 0)}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Grade answers on unseen-v2 (a local measure).")
    ap.add_argument("runs", nargs="+", help="run.py output directories or predictions.jsonl files")
    ap.add_argument("--vs", nargs="*", default=[], help="baseline runs: compare the runs above with these")
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--resample-runs", action="store_true", help="also redraw the runs of each side")
    ap.add_argument("--json", help="also write everything to this file")
    a = ap.parse_args(argv)
    loaded = {p: select(v1.load(p), a.split) for p in a.runs + a.vs}
    report: dict[str, Any] = {"split": a.split, "runs": {}}
    for p, rows in loaded.items():
        report["runs"][p] = summary(rows)
        print(p, json.dumps(report["runs"][p]))
    if a.vs:
        report["vs"] = paired_bootstrap([loaded[p] for p in a.runs], [loaded[p] for p in a.vs],
                                        resamples=a.resamples, resample_runs=a.resample_runs)
        print("runs vs --vs:", json.dumps(report["vs"]))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
