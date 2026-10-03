"""Fit and apply per-kind image temperatures from evaluation/vision/run.py outputs.

run.py stores each item's probabilities after the temperature it ran with (T0, v19's
`temperature_by_kind` unless --temps). log p = logit / T0 + c, so the logits are
recovered up to a per-row constant and any other temperature can be applied exactly,
with no GPU re-forward.

    # fit on held-out items (never an evaluation set)
    python research/image-track/temps.py fit --rows calib/strands-v19.jsonl --out temps.json
    # rescore an evaluation run's with-image rows under those temperatures
    python research/image-track/temps.py apply --run evalrun --temps temps.json --out evalrun-T

`apply` leaves the image-removed (blind) rows as they are: the server answers a request
without images on the text path, with the text temperatures.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys

V19_T0 = {"noul": 0.9107136998460428, "choice": 0.734189596436441, "score": 1.32780942142348}


def read(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def logits(probs: list[float], t0: float) -> list[float]:
    return [t0 * math.log(max(p, 1e-30)) for p in probs]


def softmax_t(z: list[float], t: float) -> list[float]:
    m = max(z)
    e = [math.exp((x - m) / t) for x in z]
    s = sum(e)
    return [x / s for x in e]


def nll(rows: list[dict], t0: dict[str, float], t: float) -> float:
    tot = 0.0
    for r in rows:
        p = softmax_t(logits(r["probs"], t0[r["kind"]]), t)
        tot -= math.log(max(p[r["gold"]], 1e-30))
    return tot / len(rows)


def fit(rows: list[dict], t0: dict[str, float]) -> dict[str, float]:
    out = {}
    grid = [math.exp(math.log(0.3) + i * (math.log(6.0) - math.log(0.3)) / 600) for i in range(601)]
    for kind in sorted({r["kind"] for r in rows}):
        sel = [r for r in rows if r["kind"] == kind]
        best = min(grid, key=lambda t: nll(sel, t0, t))
        out[kind] = round(best, 4)
        print(f"[temps] {kind}: n={len(sel)} T={best:.4f} nll {nll(sel, t0, t0[kind]):.4f} -> "
              f"{nll(sel, t0, best):.4f}", file=sys.stderr)
    return out


def rescale(rows: list[dict], t0: dict[str, float], t1: dict[str, float]) -> list[dict]:
    out = []
    for r in rows:
        k = r["kind"]
        out.append({**r, "probs": softmax_t(logits(r["probs"], t0[k]), t1.get(k, t0[k]))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--rows", nargs="+", required=True, help="with-image result jsonl(s), held out")
    f.add_argument("--t0", default=json.dumps(V19_T0), help="temperatures the rows were scored at")
    f.add_argument("--out", required=True)
    a_ = sub.add_parser("apply")
    a_.add_argument("--run", required=True, help="run.py output directory")
    a_.add_argument("--temps", required=True, help="json file from `fit`")
    a_.add_argument("--t0", default=json.dumps(V19_T0))
    a_.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = json.loads(a.t0)
    if a.cmd == "fit":
        rows = [r for p in a.rows for r in read(p)]
        temps = fit(rows, t0)
        with open(a.out, "w") as fh:
            json.dump(temps, fh, indent=2)
        print(json.dumps(temps))
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    t1 = json.load(open(a.temps))
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for name in sorted(os.listdir(a.run)):
        if not name.endswith(".jsonl"):
            continue
        rows = read(os.path.join(a.run, name))
        tag = name[: -len(".jsonl")]
        if not tag.endswith("-blind"):
            rows = rescale(rows, t0, t1)
            with open(os.path.join(a.out, name), "w") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in rows)
        else:
            shutil.copy(os.path.join(a.run, name), os.path.join(a.out, name))
        res[tag] = rows
    from score import score_runs

    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump({"args": {"rescaled_from": a.run, "temps": t1, "t0": t0}, "scores": score_runs(res)},
                  fh, indent=2)
    print(f"[temps] wrote {a.out}")


if __name__ == "__main__":
    main()
