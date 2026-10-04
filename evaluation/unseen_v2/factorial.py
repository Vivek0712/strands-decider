"""Stage D: main effects of the 2^3 factorial on unseen-v2, and the pre-registered decision.

The eight cells (configs/next/stage-d/) switch three arms on or off:

  A1  new training task families (strands_decider.data.families)
  A2  RPS + NLL for the score type (TrainConfig.score_rps_weight, ordinal_smoothing 0)
  A3  the fixed two-teacher loss (TrainConfig.pool_teacher_files / pool_alpha)

A cell is named by its switches, A1 A2 A3, as three digits: `000` is the control, `101` is
A1 and A3. All cells continue one checkpoint with one seed, so they see the same data
order and differ only in their arms (common random numbers). An arm's main effect is the
mean over the four cells with it on minus the mean over the four with it off, paired by
item with a bootstrap 95% interval (score.paired_bootstrap).

The decision rule, fixed before any cell runs (research-next/PLAN-FINAL.md, Stage D):
adopt arm X only if its pre-registered main metric improves with a 95% interval excluding 0,
and no other headline metric drops by more than 1 point (point estimate).

  A1: unseen-v2 Intelligence            A2: score-type competence (expected level)
  A3: yes/no competence (unseen-v2)

Headline metrics: unseen-v2 Intelligence and the three competences, plus public v1.5-style
Intelligence when `--public` gives it per cell (a JSON {"000": 37.1, ...} from
evaluation/jevbench/v15_proxy.py; a point estimate, no interval). Interactions (sign
contrasts of two arms) are reported, not decided on.

    python evaluation/unseen_v2/factorial.py --cell 000=reports/unseen_v2/d000 ... --cell 111=... \\
        [--split dev] [--public public.json] [--json out.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unseen import score as v1
from unseen_v2 import score

ARMS = ("A1", "A2", "A3")
PRIMARY = {"A1": "intelligence", "A2": "competence_score", "A3": "competence_noul"}
HEADLINE = ("intelligence", "competence_noul", "competence_choice", "competence_score")
MAX_DROP = 1.0


def cells(specs: list[str]) -> dict[str, str]:
    out = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or len(name) != 3 or set(name) - {"0", "1"}:
            raise SystemExit(f"--cell takes NNN=PATH with N in 0/1 (A1 A2 A3), got {spec!r}")
        out[name] = path
    missing = {"".join(b) for b in itertools.product("01", repeat=3)} - set(out)
    if missing:
        raise SystemExit(f"cells missing: {sorted(missing)}")
    return out


def contrast(runs: dict[str, list[dict[str, Any]]], sign: Any, **kw: Any) -> dict[str, dict[str, Any]]:
    """Mean over the cells where `sign(name)` is true minus the mean over the others."""
    plus = [rows for name, rows in sorted(runs.items()) if sign(name)]
    minus = [rows for name, rows in sorted(runs.items()) if not sign(name)]
    return score.paired_bootstrap(plus, minus, **kw)


def decide(effects: dict[str, dict[str, dict[str, Any]]], public: dict[str, float] | None) -> dict[str, Any]:
    out = {}
    for j, arm in enumerate(ARMS):
        eff = effects[arm]
        main = eff[PRIMARY[arm]]
        drops = {m: eff[m]["diff"] for m in HEADLINE if m != PRIMARY[arm] and m in eff and eff[m]["diff"] < -MAX_DROP}
        if public:
            on = [v for k, v in public.items() if k[j] == "1"]
            off = [v for k, v in public.items() if k[j] == "0"]
            d = sum(on) / len(on) - sum(off) / len(off)
            if d < -MAX_DROP:
                drops["public_intelligence"] = round(d, 2)
        improves = main["diff"] > 0 and main["excludes_0"]
        out[arm] = {"primary": PRIMARY[arm], "effect": main, "drops_over_1": drops,
                    "adopt": bool(improves and not drops)}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Main effects of the Stage D factorial on unseen-v2.")
    ap.add_argument("--cell", action="append", required=True, help="NNN=RUN, the switches of A1 A2 A3")
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    ap.add_argument("--public", help="JSON of public v1.5-style Intelligence per cell")
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    paths = cells(a.cell)
    runs = {name: score.select(v1.load(p), a.split) for name, p in paths.items()}
    public = None
    if a.public:
        with open(a.public, encoding="utf-8") as fh:
            public = {k: float(v) for k, v in json.load(fh).items()}
    kw = {"resamples": a.resamples}
    effects = {arm: contrast(runs, lambda n, j=j: n[j] == "1", **kw) for j, arm in enumerate(ARMS)}
    interactions = {f"{ARMS[i]}x{ARMS[j]}": {m: v for m, v in contrast(
        runs, lambda n, i=i, j=j: n[i] == n[j], **kw).items() if m in HEADLINE}
        for i, j in itertools.combinations(range(3), 2)}
    report = {"split": a.split, "cells": paths, "main_effects": effects, "interactions": interactions,
              "decision": decide(effects, public)}
    for arm, d in report["decision"].items():
        print(f"{arm}: {d['primary']} {d['effect']['diff']:+.2f} {d['effect']['ci95']} "
              f"drops {d['drops_over_1'] or 'none'} -> {'ADOPT' if d['adopt'] else 'reject'}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
