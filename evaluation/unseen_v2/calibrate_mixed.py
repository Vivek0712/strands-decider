"""Stage E calibration: per-kind temperatures fitted by NLL on a 50/50 mix of in-family
held-out rows and unseen-v2 dev rows, written into a copy of the checkpoint.

The board pools calibration over its open and sealed halves, and the sealed half is
mostly families we never trained on. Fitting on in-family held-out rows alone left us
over-confident on unseen yes/no (research-next/PLAN.md §2.3.4), so each kind's fit sees as
many unseen-v2 dev rows as in-family rows. unseen-v2 test rows are never read here.

    python evaluation/unseen_v2/calibrate_mixed.py CKPT COPY --unseen data/unseen_v2.jsonl \\
        --in-family data/holdout_v5_norule.jsonl data/multistep_v14_eval.jsonl ... [--per-kind 1000]

For each kind, min(per-kind, available) rows are drawn from each side (in-family rows
evenly over the given files, `evaluate.partition_examples`' calibration half), so the
two halves weigh the same. Writes COPY (the checkpoint with the new temperatures) and
COPY/calibration_mixed.json (the rows used, the old and new temperatures, the NLL and ECE
of each half before and after).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KINDS = ("noul", "choice", "score")


def draw(rows: list[Any], n: int, seed: int) -> list[Any]:
    return random.Random(seed).sample(rows, min(n, len(rows)))


def mix(in_family: dict[str, list[Any]], unseen: list[Any], per_kind: int, seed: int = 0) -> tuple[list[Any], list[Any]]:
    """(in-family rows, unseen rows), equal counts per kind: each kind takes
    min(per_kind, in-family available, unseen available) from each side, the in-family share
    spread evenly over the files."""
    ours: list[Any] = []
    theirs: list[Any] = []
    for kind in KINDS:
        u = [e for e in unseen if e.kind == kind]
        files = {f: [e for e in rows if e.kind == kind] for f, rows in in_family.items()}
        files = {f: rows for f, rows in files.items() if rows}
        n = min(per_kind, len(u), sum(map(len, files.values())))
        if not n:
            continue
        theirs += draw(u, n, seed)
        share, left = n // max(len(files), 1), n
        for j, (_, rows) in enumerate(sorted(files.items())):
            take = left if j == len(files) - 1 else min(share, len(rows))
            got = draw(rows, take, seed + j)
            ours += got
            left -= len(got)
    return ours, theirs


def main(argv: list[str] | None = None) -> None:
    from strands_decider.data.format import Example, read_jsonl
    from strands_decider.evaluate import (
        collect_logits,
        fit_temperature_by_kind,
        partition_examples,
        predictions_from_logits,
        summarise,
    )
    from strands_decider.modeling import StrandsDeciderModel

    ap = argparse.ArgumentParser(description="Fit temperatures on in-family + unseen-v2 dev, 50/50.")
    ap.add_argument("checkpoint")
    ap.add_argument("copy")
    ap.add_argument("--unseen", required=True, help="unseen-v2 rows (build.py); only split=dev is read")
    ap.add_argument("--in-family", nargs="+", required=True, help="in-family held-out files")
    ap.add_argument("--per-kind", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    with open(a.unseen, encoding="utf-8") as fh:
        unseen = [Example.from_dict(r) for r in map(json.loads, fh) if r.get("split") == "dev"]
    in_family = {f: partition_examples(list(read_jsonl(f)), "calib") for f in a.in_family}
    ours, theirs = mix(in_family, unseen, a.per_kind, a.seed)
    model = StrandsDeciderModel.load(a.checkpoint)
    old = dict(model.config.temperature_by_kind)
    eps = model.config.ordinal_smoothing

    def logits(rows: list[Example]) -> Any:
        return collect_logits(model, rows, device=a.device, max_length=model.config.max_length)

    lo, lt = logits(ours), logits(theirs)
    import torch

    both = (torch.cat([lo[0], lt[0]]), torch.cat([lo[1], lt[1]]), torch.cat([lo[2], lt[2]]), lo[3] + lt[3])
    new = {**old, **fit_temperature_by_kind(*both, objective="nll", ordinal_smoothing=eps)}
    report: dict[str, Any] = {"old": old, "new": new, "rows": {
        "in_family": {k: sum(e.kind == k for e in ours) for k in KINDS},
        "unseen_dev": {k: sum(e.kind == k for e in theirs) for k in KINDS}}}
    for name, (lg, y, s, ex) in (("in_family", lo), ("unseen_dev", lt)):
        for tag, t in (("old", old), ("new", new)):
            o = summarise(predictions_from_logits(lg, y, s, ex, t, eps), by_task=False)["overall"]
            report.setdefault(name, {})[tag] = {"nll": o["nll"], "ece": o["ece"], "accuracy": o["accuracy"]}
    shutil.rmtree(a.copy, ignore_errors=True)
    shutil.copytree(a.checkpoint, a.copy)
    from strands_decider.modeling import config_path

    path = config_path(a.copy)
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["temperature_by_kind"] = new
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    with open(os.path.join(a.copy, "calibration_mixed.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
