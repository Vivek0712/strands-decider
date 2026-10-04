"""Arm A3: a pooled, recalibrated teacher, mixed with gold on every row.

v20's teacher target kept the 27B's distribution only where it agreed with gold, over
replayed distributions elsewhere. Conditioning a target on agreement with the realised
label is improper: the student's optimum is pulled off the truth (verify_math.md: with a
calibrated teacher at q = 0.60 it learns 0.46, and rows at q = 0.8-0.85 are pulled into the
0.2-0.8 band). The fix, applied by train.py when `pool_teacher_files` is set:

    loss(row) = alpha * KL(pool || student) + (1 - alpha) * CE(gold, student)

on EVERY row a teacher labelled, with a constant alpha (`pool_alpha`): no agreement filter
and no down-weighting of the rows where the teachers disagree with gold. Rows no teacher
labelled keep plain CE. `pool` is the mean of the teachers' probabilities (a linear pool,
never a product of experts, which double-counts shared evidence), each teacher first
recalibrated by a per-kind temperature fitted by NLL on a held-out split of its own rows.
With one teacher file (the 27B alone) the pool is that teacher, recalibrated.

A teacher file is data/teacher.py's format, {"i": row index in the concatenation of the
config's train_files, "probs": [...canonical order]}, labelled on every row of the kinds
wanted, agreeing or not (`python -m strands_decider.data.teacher_yn label --kinds noul
choice`). `fit` writes its calibration beside it, `<file>.calibration.json`, which train.py
reads and refuses to run without:

    python -m strands_decider.data.teacher_pool fit --teacher data/teacher_pool_qwen35-27b.jsonl

`fit` also reports the teacher's accuracy, NLL and ECE per kind on the held-out rows,
before and after recalibration: the quality pilot a second teacher must pass first.
The held-out rows are drawn by `--seed` and recorded; the temperatures are fitted on them
alone, and the training target uses them on every row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections.abc import Sequence
from typing import Any

from . import shards
from .format import Example, read_jsonl

KINDS = ("noul", "choice", "score")
# log-temperature search range: T in [0.1, 10]
LOG_T = (math.log(0.1), math.log(10.0))


def calibration_path(teacher_file: str) -> str:
    return teacher_file + ".calibration.json"


def recalibrate(probs: Sequence[float], temperature: float) -> list[float]:
    """softmax(log p / T): T > 1 flattens, T < 1 sharpens. Zero-probability options stay 0."""
    logs = [math.log(p) / temperature if p > 0 else -math.inf for p in probs]
    top = max(logs)
    exp = [math.exp(x - top) if x > -math.inf else 0.0 for x in logs]
    total = sum(exp)
    return [x / total for x in exp]


def nll(rows: Sequence[tuple[Sequence[float], int]], temperature: float = 1.0) -> float:
    """Mean negative log probability of the gold option after recalibration."""
    return -sum(math.log(max(recalibrate(p, temperature)[g], 1e-12)) for p, g in rows) / max(len(rows), 1)


def ece(rows: Sequence[tuple[Sequence[float], int]], temperature: float = 1.0, bins: int = 10) -> float:
    """Top-probability ECE over equal-width bins."""
    cells: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for p, g in rows:
        q = recalibrate(p, temperature)
        conf = max(q)
        cells[min(bins - 1, int(conf * bins))].append((conf, q.index(conf) == g))
    return sum(abs(sum(c for c, _ in x) - sum(k for _, k in x)) for x in cells if x) / max(len(rows), 1)


def fit_temperature(rows: Sequence[tuple[Sequence[float], int]], steps: int = 60) -> float:
    """The temperature minimising held-out NLL: a grid over log T, then golden-section
    search around the best grid point (NLL is unimodal in log T in practice)."""
    if not rows:
        return 1.0
    grid = [LOG_T[0] + (LOG_T[1] - LOG_T[0]) * k / 40 for k in range(41)]
    best = min(grid, key=lambda x: nll(rows, math.exp(x)))
    lo, hi = best - (LOG_T[1] - LOG_T[0]) / 40, best + (LOG_T[1] - LOG_T[0]) / 40
    g = (math.sqrt(5) - 1) / 2
    for _ in range(steps):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if nll(rows, math.exp(a)) <= nll(rows, math.exp(b)):
            hi = b
        else:
            lo = a
    return math.exp((lo + hi) / 2)


def heldout_indices(indices: Sequence[int], fraction: float, seed: int) -> set[int]:
    """A `fraction` of the labelled rows, drawn by `seed`."""
    pool = sorted(indices)
    return set(random.Random(seed).sample(pool, round(fraction * len(pool))))


def _corpus(files: Sequence[str]) -> list[Example]:
    out: list[Example] = []
    for f in files:
        out.extend(read_jsonl(f))
    return out


def fit(teacher_file: str, train_files: Sequence[str], fraction: float = 0.1, seed: int = 0,
        examples: Sequence[Example] | None = None) -> dict[str, Any]:
    """Per-kind temperatures for `teacher_file`, fitted on a held-out `fraction` of its rows,
    and the teacher's quality there before and after."""
    examples = list(examples) if examples is not None else _corpus(train_files)
    raw = shards.read(teacher_file)
    for i, p in raw.items():
        if i >= len(examples) or len(p) != examples[i].n_options:
            raise ValueError(f"teacher row {i} does not match train_files (wrong order or files?)")
    held = heldout_indices(list(raw), fraction, seed)
    report: dict[str, Any] = {}
    temps: dict[str, float] = {}
    for kind in KINDS:
        rows = [(raw[i], examples[i].label) for i in sorted(held) if examples[i].kind == kind]
        if not rows:
            continue
        t = fit_temperature(rows)
        temps[kind] = round(t, 4)
        acc = sum(max(range(len(p)), key=p.__getitem__) == g for p, g in rows) / len(rows)
        report[kind] = {"heldout_rows": len(rows), "accuracy": round(acc, 4),
                        "nll": [round(nll(rows), 4), round(nll(rows, t), 4)],
                        "ece": [round(ece(rows), 4), round(ece(rows, t), 4)]}
    with open(teacher_file, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return {"teacher_file": os.path.basename(teacher_file), "teacher_sha256": digest,
            "train_files": list(train_files), "heldout_fraction": fraction, "seed": seed,
            "labelled_rows": len(raw), "temperature_by_kind": temps,
            "heldout_report_before_after": report}


def load_pool(files: Sequence[str], examples: Sequence[Example], train_files: Sequence[str]) -> dict[int, list[float]]:
    """{row index: pooled probabilities}: every teacher's rows recalibrated by its own
    temperatures, then averaged over the teachers that labelled the row. Each teacher's
    calibration must exist and must have been fitted on a prefix of `train_files` (A1's
    files are appended after the teacher's, so its indices still hold)."""
    sums: dict[int, list[float]] = {}
    counts: dict[int, int] = {}
    for f in files:
        cal_path = calibration_path(f)
        if not os.path.exists(cal_path):
            raise FileNotFoundError(f"{cal_path}: run `python -m strands_decider.data.teacher_pool fit "
                                    f"--teacher {f}` first")
        with open(cal_path, encoding="utf-8") as fh:
            cal = json.load(fh)
        labelled_on = list(cal["train_files"])
        if list(train_files[: len(labelled_on)]) != labelled_on:
            raise ValueError(f"{f} was labelled on {labelled_on}, which does not begin this config's "
                             f"train_files {list(train_files)}")
        temps = cal["temperature_by_kind"]
        for i, p in shards.read(f).items():
            ex = examples[i]
            if len(p) != ex.n_options:
                raise ValueError(f"{f} row {i} has {len(p)} probs for {ex.n_options} options")
            q = recalibrate(p, float(temps.get(ex.kind, 1.0)))
            if i in sums:
                sums[i] = [a + b for a, b in zip(sums[i], q, strict=True)]
            else:
                sums[i] = q
            counts[i] = counts.get(i, 0) + 1
    return {i: [x / counts[i] for x in s] for i, s in sums.items()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Recalibrate a teacher file for the pooled-teacher loss (A3).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit per-kind temperatures on a held-out split; write <file>.calibration.json")
    f.add_argument("--teacher", required=True, help="a teacher file labelled on every row, agreeing or not")
    f.add_argument("--train-files", nargs="+", default=None,
                   help="the files the teacher indexes, in order (default: teacher_yn's TRAIN_FILES)")
    f.add_argument("--heldout-fraction", type=float, default=0.1)
    f.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from .teacher_yn import TRAIN_FILES

    out = fit(a.teacher, a.train_files or TRAIN_FILES, a.heldout_fraction, a.seed)
    with open(calibration_path(a.teacher), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
