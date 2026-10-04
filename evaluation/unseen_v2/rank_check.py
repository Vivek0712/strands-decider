"""Stage C: run public board systems on unseen-v2, evaluation only, then check rank agreement.

Two steps. `run` answers unseen-v2 with every system in systems.json that has an adapter
(or those named by --only), through evaluation/unseen/run.py's adapter interface, into
OUT/<system>/ (predictions.jsonl, run_meta.json, and system.json: the systems.json entry
plus the revision the download was pinned to). Each system's download is passed in, never
stored here:

    python evaluation/unseen_v2/rank_check.py run --rows data/unseen_v2.jsonl --out reports/rank \\
        --repo decider-2b=/path/to/download --revision decider-2b=<commit> [--only decider-2b ...]

A system behind the System One API is run with `--url NAME=http://HOST:PORT` instead.
`spearman` then compares our ordering with the board's (spearman.py):

    python evaluation/unseen_v2/spearman.py --official evaluation/unseen_v2/official_v1.5.5.csv \\
        --dir reports/rank [--run ours=reports/unseen_v2/v20-soup]

Nothing here trains on, or keeps, a system's answers beyond its probabilities per item;
OUT gets a DO_NOT_TRAIN.md marker that says so.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from unseen import run as runner  # noqa: E402

MARKER = ("# Evaluation only\n\nThese are other systems' answers on unseen-v2 (Stage C). They measure rank "
          "agreement with the board and nothing else. Never use them as training rows, teacher targets or "
          "calibration data.\n")


def _pairs(values: list[str] | None, flag: str) -> dict[str, str]:
    out = {}
    for v in values or []:
        k, sep, x = v.partition("=")
        if not sep:
            raise SystemExit(f"{flag} takes NAME=VALUE, got {v!r}")
        out[k] = x
    return out


def systems(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        out: list[dict[str, Any]] = json.load(fh)["systems"]
    return out


def run(a: argparse.Namespace) -> None:
    repos, revisions, urls = _pairs(a.repo, "--repo"), _pairs(a.revision, "--revision"), _pairs(a.url, "--url")
    todo = [s for s in systems(a.systems) if not a.only or s["name"] in a.only]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "DO_NOT_TRAIN.md"), "w", encoding="utf-8") as fh:
        fh.write(MARKER)
    skipped = []
    for s in todo:
        name = s["name"]
        out = os.path.join(a.out, name)
        if os.path.exists(os.path.join(out, "predictions.jsonl")):
            print(f"{name}: already answered, skipped")
            continue
        if name in urls:
            args = ["--adapter", urls[name]]
        elif s.get("adapter") and name in repos:
            os.environ["SD_RANK_REPO"] = os.path.abspath(repos[name])
            args = ["--adapter", s["adapter"], "--path", HERE]
        else:
            skipped.append(name)
            continue
        if name not in revisions:
            raise SystemExit(f"{name}: pass --revision {name}=<the commit its download is pinned to>")
        runner.main(["--rows", a.rows, "--out", out, *args])
        with open(os.path.join(out, "system.json"), "w", encoding="utf-8") as fh:
            json.dump({**s, "revision": revisions[name]}, fh, indent=2)
    if skipped:
        print("not run (no adapter, --repo or --url):", ", ".join(skipped))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run public board systems on unseen-v2 (evaluation only).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--rows", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--systems", default=os.path.join(HERE, "systems.json"))
    r.add_argument("--repo", action="append", help="NAME=DIR: the system's download")
    r.add_argument("--revision", action="append", help="NAME=COMMIT: the revision that download is pinned to")
    r.add_argument("--url", action="append", help="NAME=http://HOST:PORT: a System One server")
    r.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    run(a)


if __name__ == "__main__":
    main()
