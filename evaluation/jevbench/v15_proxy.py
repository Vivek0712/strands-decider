#!/usr/bin/env python3
"""A LOCAL PROXY for JevBench v1.5 scoring, applied to JevBench v1 public results.

Not the official score. v1.5's scorer and its sealed half are not public; this applies
the published v1.5 rules to the 231 public v1 tasks of a `results.jsonl`:

  - a yes/no answer with 0.2 < P(yes) < 0.8 is an abstention and counts as wrong;
  - each task scores chance-corrected: (correct - 1/n) / (1 - 1/n), n options (at least 2);
  - competence per question type (choice, yes/no, score) is the mean over its tasks, and
    the proxy Intelligence is the mean of the three.

It leaves out v1.5's tier weights and sealed tasks, so the number ranks runs against
each other on this machine and nothing else; never compare it with a board score. The
question type is read from the result itself (an `ordinal_ev` means score; probabilities
over exactly yes and no mean yes/no), as v1 results carry no type.

Per run it also reports how many yes/no answers fall in the abstention band (and by
family), the share of yes answers, the top-probability ECE, and JevBench's own Brier, ECE
and ordinal MAE from the run's `summary.json` when it is there. With `--vs`, the runs
given first are taken as seeds of one arm and compared with the `--vs` runs (v19, say)
task by task: the mean difference over seeds and a paired bootstrap 95% interval over
the 231 tasks (10,000 resamples).

    python evaluation/jevbench/v15_proxy.py RUN [RUN ...] [--vs BASE ...] [--json out.json]

RUN is a JevBench run directory (results.jsonl, summary.json) or a results.jsonl.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Any

BAND = (0.2, 0.8)  # v1.5: a yes/no answer strictly inside is an abstention
KINDS = ("choice", "noul", "score")


def kind(r: dict[str, Any]) -> str:
    """"score", "noul" (yes/no) or "choice", from the result row alone."""
    if r.get("ordinal_ev") is not None:
        return "score"
    return "noul" if set(r.get("probs") or {}) == {"yes", "no"} else "choice"


def in_band(r: dict[str, Any]) -> bool:
    return kind(r) == "noul" and BAND[0] < (r.get("probs") or {}).get("yes", 0.5) < BAND[1]


def task_score(r: dict[str, Any]) -> float:
    """The chance-corrected credit for one task under the v1.5 rules."""
    right = bool(r.get("correct")) and not in_band(r)
    chance = 1.0 / max(2, len(r.get("probs") or {}))
    return ((1.0 if right else 0.0) - chance) / (1 - chance)


def proxy(rows: list[dict[str, Any]]) -> tuple[dict[str, float], float]:
    """Competence per question type (in %), and the proxy Intelligence, their mean."""
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(kind(r), []).append(task_score(r))
    per = {k: 100 * sum(v) / len(v) for k, v in by.items()}
    return per, sum(per.values()) / len(per)


def ece_top(rows: list[dict[str, Any]], bins: int = 10) -> float:
    """Top-probability ECE over equal-width bins (JevBench's binning)."""
    cells: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for r in rows:
        if r.get("probs"):
            conf = max(r["probs"].values())
            cells[min(bins - 1, int(conf * bins))].append((conf, bool(r.get("correct"))))
    n = sum(map(len, cells))
    return sum(abs(sum(c for c, _ in x) - sum(k for _, k in x)) for x in cells if x) / max(n, 1)


def load(path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """A run's result rows, and its summary.json (empty when absent)."""
    run = path if os.path.isdir(path) else os.path.dirname(path)
    res = os.path.join(path, "results.jsonl") if os.path.isdir(path) else path
    with open(res, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    summary: dict[str, Any] = {}
    if os.path.exists(os.path.join(run, "summary.json")):
        with open(os.path.join(run, "summary.json"), encoding="utf-8") as fh:
            summary = json.load(fh)
    return rows, summary


def metrics(rows: list[dict[str, Any]], summary: dict[str, Any] | None = None) -> dict[str, Any]:
    """Every number this script reports for one run."""
    per, intelligence = proxy(rows)
    yn = [r for r in rows if kind(r) == "noul"]
    band = [r for r in yn if in_band(r)]
    out: dict[str, Any] = {
        "tasks": len(rows), "n_correct": sum(bool(r.get("correct")) for r in rows),
        "competence_by_type": {k: round(v, 1) for k, v in per.items()},
        "intelligence_proxy": round(intelligence, 1),
        "yes_no": len(yn), "yes_no_correct": sum(bool(r.get("correct")) for r in yn),
        "yes_no_in_band": len(band), "band_correct": sum(bool(r.get("correct")) for r in band),
        "band_by_family": dict(Counter(r.get("family") for r in band).most_common()),
        "yes_share": round(sum(r.get("predicted") == "yes" for r in yn) / max(1, len(yn)), 3),
        "ece_top": round(ece_top(rows), 4),
    }
    if summary:
        ece = summary.get("ece")
        out.update(brier=summary.get("brier_mean"), ece=ece["ece"] if isinstance(ece, dict) else ece,
                   ordinal_mae=summary.get("ordinal_mae"))
    return out


def paired_bootstrap(arm: list[list[dict[str, Any]]], base: list[list[dict[str, Any]]],
                     resamples: int = 10_000, seed: int = 0) -> dict[str, dict[str, float]]:
    """Mean over seeds of `arm` minus mean over `base`, for the proxy Intelligence, the
    yes/no answers in the band and the tasks right, with a paired bootstrap 95% interval:
    tasks are resampled with replacement and every run is scored on the same sample."""
    runs = [{r["task_id"]: r for r in rows} for rows in arm + base]
    ids = sorted(runs[0])
    if any(sorted(run) != ids for run in runs):
        raise ValueError("the runs do not cover the same tasks")

    def stats(sample: list[str]) -> list[float]:
        vals = []
        for run in runs:
            rows = [run[t] for t in sample]
            vals.append([proxy(rows)[1], sum(map(in_band, rows)), sum(bool(r.get("correct")) for r in rows)])
        a, b = vals[: len(arm)], vals[len(arm):]
        return [sum(v[j] for v in a) / len(a) - sum(v[j] for v in b) / len(b) for j in range(3)]

    rng = random.Random(seed)
    point = stats(ids)
    samples = [stats(rng.choices(ids, k=len(ids))) for _ in range(resamples)]
    draws = [sorted(column) for column in zip(*samples, strict=True)]
    lo, hi = int(0.025 * resamples), int(0.975 * resamples) - 1
    names = ("intelligence_proxy", "yes_no_in_band", "n_correct")
    return {n: {"diff": round(point[j], 2), "ci95": [round(draws[j][lo], 2), round(draws[j][hi], 2)]}
            for j, n in enumerate(names)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="A local proxy for JevBench v1.5 scoring (not the official score).")
    ap.add_argument("runs", nargs="+", help="JevBench run directories or results.jsonl files")
    ap.add_argument("--vs", nargs="*", default=[], help="baseline runs: compare the runs above with these")
    ap.add_argument("--json", help="also write everything to this file")
    a = ap.parse_args(argv)
    report: dict[str, Any] = {"runs": {}}
    loaded = {p: load(p) for p in a.runs + a.vs}
    for p, (rows, summary) in loaded.items():
        report["runs"][p] = metrics(rows, summary)
        print(p, json.dumps(report["runs"][p]))
    if a.vs:
        report["vs"] = paired_bootstrap([loaded[p][0] for p in a.runs], [loaded[p][0] for p in a.vs])
        print("runs vs --vs:", json.dumps(report["vs"]))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
