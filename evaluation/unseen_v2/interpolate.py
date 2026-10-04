"""Stage E: is there a loss barrier between two seeds? Soup them only if there is none.

Mode connectivity is not guaranteed at a high learning rate (Model Soups reports barriers),
so before averaging two seeds this walks the line between them: for each t in `--points`
the weighted soup (1 - t) * A + t * B (strands_decider.soup with coefs; the merged LoRA
update is interpolated exactly, not its factors), scored by NLL and unseen-v2 measures on
the same rows, every point at temperature 1 (the soup resets calibration, so the endpoints
are scored the same way). The barrier is

    max over interior t of  NLL(t) - ((1 - t) * NLL(0) + t * NLL(1))

and the rule, fixed before the run: soup only if the barrier is <= --tolerance (default 0).

    python evaluation/unseen_v2/interpolate.py checkpoints/next-e-4b-s0 checkpoints/next-e-4b-s1 \\
        --rows data/unseen_v2.jsonl --split dev --limit 3000 --json reports/interp-4b.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unseen_v2 import score


def barrier(curve: list[dict[str, Any]]) -> float:
    """The largest rise of NLL above the straight line between the endpoints."""
    a, b = curve[0], curve[-1]
    return max((p["nll"] - ((1 - p["t"]) * a["nll"] + p["t"] * b["nll"]) for p in curve[1:-1]), default=0.0)


def rows_of(path: str, split: str, limit: int, seed: int = 0) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    rows = [dict(r, id=i) for i, r in enumerate(rows) if split == "all" or r.get("split") == split]
    return sorted(random.Random(seed).sample(rows, min(limit, len(rows))), key=lambda r: r["id"])


def point(ckpt: str, rows: list[dict[str, Any]], device: str) -> dict[str, Any]:
    from strands_decider.data.format import Example
    from strands_decider.evaluate import collect_logits, predictions_from_logits
    from strands_decider.modeling import StrandsDeciderModel

    model = StrandsDeciderModel.load(ckpt)
    exs = [Example.from_dict(r) for r in rows]
    lg, y, s, ex = collect_logits(model, exs, device=device, max_length=model.config.max_length)
    preds = predictions_from_logits(lg, y, s, ex, 1.0)
    answered = [{"id": r["id"], "task": r["task"], "kind": r["kind"], "gold": r["label"], "probs": p.probs}
                for r, p in zip(rows, preds, strict=True)]
    m = score.measures(score.features(answered).sum(axis=0))
    return {k: round(float(v), 4) for k, v in m.items() if v == v}


def main(argv: list[str] | None = None) -> None:
    from strands_decider.soup import soup

    ap = argparse.ArgumentParser(description="The loss along the line between two seeds, before souping.")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rows", required=True, help="unseen-v2 rows (or any typed rows with a label)")
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--points", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--tolerance", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    ts = sorted(set(a.points) | {0.0, 1.0})
    rows = rows_of(a.rows, a.split, a.limit)
    curve = []
    scratch = tempfile.mkdtemp(prefix="interp-")
    try:
        for t in ts:
            out = os.path.join(scratch, f"t{t:.3f}")
            soup([a.a, a.b], out, coefs=[1 - t, t])
            curve.append({"t": t, **point(out, rows, a.device)})
            shutil.rmtree(out)
            print(json.dumps(curve[-1]), flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    report = {"a": a.a, "b": a.b, "rows": len(rows), "split": a.split, "curve": curve,
              "barrier_nll": round(barrier(curve), 5), "tolerance": a.tolerance}
    report["soup"] = report["barrier_nll"] <= a.tolerance
    print(json.dumps({k: v for k, v in report.items() if k != "curve"}))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
