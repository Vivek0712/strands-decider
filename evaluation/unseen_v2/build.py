"""unseen-v2: 9,000 questions from families no Strands Decider trains on, a third per type.

v1 (evaluation/unseen/) had 1,050 rows, too few to see a 3-point effect. v2 keeps v1's five
families, adds two public datasets and four generator families whose gold is computed by a
program (generators.py), and balances the three question types, the weights the board's
headline gives them:

  yes/no   StrategyQA 400, RuleTaker depth 5 400, CommonsenseQA 2.0 400,
           chess / arithmetic / calendar / seating 450 each
  choice   CommonsenseQA 400, ARC-Challenge 400, HellaSwag 400,
           chess / arithmetic / calendar / seating 450 each
  score    STS-B 600, chess / arithmetic / calendar / seating 600 each

Not one of these families is in a training mixture: data/sources.md lists every training
source, and tests/test_unseen_v2.py checks the families against it and against the A1
training generators (strands_decider.data.families). BoolQ and MNLI are training sources,
so BoolQ-hard and e-SNLI are left out, as is anything taken from JevBench or Image JevBench.

Every row has a `split`: within each family, 40% `dev` (model selection reads it) and 60%
`test` (read once per stage), drawn by the seed. One `random.Random(seed)` draws everything
in a fixed order, the public downloads are pinned to a revision and their files hashed, so
the same inputs give the same rows; the manifest written beside the output records the
revisions, file hashes, library versions and the output's own sha256.

    pip install "strands-decider[unseen]"     # datasets and python-chess
    python evaluation/unseen_v2/build.py --out data/unseen_v2.jsonl

Rows are `Example`s (data/format.py) with a `task` of `unseen/<family>`, plus `split` and,
for generated rows, `template`. evaluation/unseen/run.py answers them; score.py here grades.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unseen import build as v1
from unseen_v2 import generators

Row = dict[str, Any]

# Public files read directly at a pinned revision: (repo, revision, file, licence).
FILES = {
    "csqa2": ("tasksource/commonsense_qa_2.0", "23ff83b18ed76882f3af1a403a7c464b72efcd86",
              "teach_your_ai_dev.json", "CC BY 4.0"),
    "hellaswag": ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76",
                  "data/validation-00000-of-00001.parquet", "MIT"),
}
# Families that must never be in unseen-v2: training sources, near-family NLI, and the
# benchmark itself (tests/test_unseen_v2.py).
EXCLUDED = ("boolq", "mnli", "snli", "esnli", "e-snli", "jevbench", "jev")
COUNTS: dict[str, dict[str, int]] = {
    "noul": {"strategyqa": 400, "ruletaker_d5": 400, "csqa2": 400,
             **{g: 450 for g in generators.FAMILIES}},
    "choice": {"commonsenseqa": 400, "arc_challenge": 400, "hellaswag": 400,
               **{g: 450 for g in generators.FAMILIES}},
    "score": {"stsb": 600, **{g: 600 for g in generators.FAMILIES}},
}
DEV_FRACTION = 0.4


def fetch(name: str) -> tuple[str, str]:
    """A pinned public file: its local path and sha256."""
    from huggingface_hub import hf_hub_download

    repo, revision, filename, _ = FILES[name]
    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    with open(path, "rb") as fh:
        return path, hashlib.sha256(fh.read()).hexdigest()


def csqa2(lines: list[str], rng: random.Random, n: int) -> list[Row]:
    """CommonsenseQA 2.0 dev: a yes/no question or a claim, gold yes or no."""
    rows = [json.loads(x) for x in lines if x.strip()]
    rng.shuffle(rows)
    return [{"kind": "noul", "state": f"Question or claim: {r['question']}",
             "instructions": "Is the answer yes (for a claim: is the claim true)?",
             "options": v1.YES_NO, "label": int(r["answer"] == "yes"), "task": "unseen/csqa2"}
            for r in rows if r.get("answer") in ("yes", "no")][:n]


def hellaswag(rows: list[Row], rng: random.Random, n: int) -> list[Row]:
    """HellaSwag validation: the context and four endings, gold the right ending. An
    ending repeated within an item keeps the item out (options are named by their text)."""
    rng.shuffle(rows)
    out = []
    for r in rows:
        endings = [e.strip() for e in r["endings"]]
        if len(set(endings)) != len(endings) or str(r.get("label", "")) == "":
            continue
        out.append({"kind": "choice", "state": f"Activity: {r['activity_label']}\nContext: {r['ctx']}",
                    "instructions": "Which ending is the most plausible continuation of the context?",
                    "options": [[e, e] for e in endings], "label": int(r["label"]), "task": "unseen/hellaswag"})
        if len(out) == n:
            break
    return out


def split(rows: list[Row], rng: random.Random) -> list[Row]:
    """Each family's rows, 40% dev and 60% test, drawn without replacement."""
    out: list[Row] = []
    for task in dict.fromkeys(r["task"] for r in rows):
        mine = [r for r in rows if r["task"] == task]
        dev = set(rng.sample(range(len(mine)), round(DEV_FRACTION * len(mine))))
        out += [{**r, "split": "dev" if i in dev else "test"} for i, r in enumerate(mine)]
    return out


def build(ruletaker_file: str, seed: int = 0, counts: dict[str, dict[str, int]] | None = None,
          public: dict[str, Any] | None = None) -> list[Row]:
    """The whole set, in the order the generator draws it. `public` maps each public family
    to its loaded rows (tests pass synthetic ones); by default they are downloaded."""
    counts = counts or COUNTS
    rng = random.Random(seed)
    if public is None:
        with open(fetch("csqa2")[0], encoding="utf-8") as fh:
            lines = fh.readlines()
        import pyarrow.parquet as pq

        public = {"strategyqa": v1._load("ChilleD/StrategyQA", "train"),
                  "commonsenseqa": v1._load("tau/commonsense_qa", "validation"),
                  "arc_challenge": v1._load("allenai/ai2_arc", "test", "ARC-Challenge"),
                  "stsb": v1._load("sentence-transformers/stsb", "test"),
                  "csqa2": lines, "hellaswag": pq.read_table(fetch("hellaswag")[0]).to_pylist()}
    c = {k: dict(v) for k, v in counts.items()}
    rows = v1.strategyqa(public["strategyqa"], rng, c["noul"].pop("strategyqa"))
    with open(ruletaker_file, encoding="utf-8") as fh:
        rows += v1.ruletaker(fh.readlines(), rng, c["noul"].pop("ruletaker_d5"))
    rows += csqa2(public["csqa2"], rng, c["noul"].pop("csqa2"))
    rows += v1.multiple_choice(public["commonsenseqa"], rng, "Which answer is most plausible?",
                               "unseen/commonsenseqa", c["choice"].pop("commonsenseqa"))
    rows += v1.multiple_choice(public["arc_challenge"], rng, "Which answer is correct?", "unseen/arc_challenge",
                               c["choice"].pop("arc_challenge"))
    rows += hellaswag(public["hellaswag"], rng, c["choice"].pop("hellaswag"))
    rows += v1.stsb(public["stsb"], rng, c["score"].pop("stsb"))
    for kind in ("noul", "choice", "score"):
        for family, n in c[kind].items():
            rows += generators.draw(family, kind, n, rng)
    bad = [r["task"] for r in rows if any(x in r["task"].lower() for x in EXCLUDED)]
    if bad:
        raise ValueError(f"excluded families in unseen-v2: {sorted(set(bad))}")
    return split(rows, rng)


def main(argv: list[str] | None = None) -> None:
    import chess
    import datasets

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ruletaker", default="data/train_v5.holdout.jsonl",
                    help="strands-decider data build's held-out file")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rows = build(a.ruletaker, a.seed)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(a.ruletaker, "rb") as fh:
        held_out = hashlib.sha256(fh.read()).hexdigest()
    manifest = {
        "rows": len(rows), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "seed": a.seed,
        "by_kind": dict(Counter(r["kind"] for r in rows)),
        "by_task_kind": dict(Counter(f"{r['task']}/{r['kind']}" for r in rows)),
        "by_split_kind": dict(Counter(f"{r['split']}/{r['kind']}" for r in rows)),
        "revisions": v1.REVISIONS,
        "files": {k: {"repo": v[0], "revision": v[1], "file": v[2], "licence": v[3], "sha256": fetch(k)[1]}
                  for k, v in FILES.items()},
        "datasets": datasets.__version__, "python_chess": chess.__version__,
        "ruletaker_file": a.ruletaker, "ruletaker_sha256": held_out,
    }
    with open(a.out + ".manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
