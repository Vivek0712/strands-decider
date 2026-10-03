"""T4: label the yes/no-heavy rows of the v19 corpus with a stronger open teacher.

Reads the teacher exactly as data/teacher.py does (SemIf prompt, thinking off, softmax over
the option letters' next-token logits), but only on selected rows of the concatenated
train_files, and writes {"i": global row index, "probs": [...canonical order]} -- the
format TrainConfig.teacher_file expects. A second pass (`--build`) keeps a teacher row only
where the teacher's argmax equals the gold label, and merges with a base teacher file
(v14 replay on the multistep rows), the new labels taking precedence on rows both cover.

    python research/text-track/teacher_yn.py label --model Qwen/Qwen3.5-27B --out data/t27b_raw.jsonl
    python research/text-track/teacher_yn.py build --raw data/t27b_raw.jsonl \
        --base data/replay_v14_multistep.jsonl --out data/teacher_t4.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from strands_decider.data import teacher as T  # noqa: E402
from strands_decider.data.format import read_jsonl  # noqa: E402

TRAIN_FILES = ["data/train_v5.jsonl", "data/multistep_v14.jsonl", "data/generated_v16.jsonl",
               "data/generated_v18.jsonl", "data/adequacy_hs2.jsonl", "data/adequacy_gen.jsonl"]
# Sources whose every row is selected; for the rest only yes/no ("noul") rows.
ALL_ROWS = {"data/adequacy_hs2.jsonl", "data/adequacy_gen.jsonl"}


def corpus(files: list[str]):
    i = 0
    for f in files:
        for ex in read_jsonl(f):
            yield i, f, ex
            i += 1


def select(files: list[str], sources: set[str]) -> list[tuple[int, str, object]]:
    out = []
    for i, f, ex in corpus(files):
        if os.path.basename(f) not in sources and f not in sources:
            continue
        if f in ALL_ROWS or ex.kind == "noul":
            out.append((i, f, ex))
    return out


def cmd_label(a: argparse.Namespace) -> None:
    rows = select(TRAIN_FILES, set(a.sources))
    by = defaultdict(int)
    for _, f, _ in rows:
        by[f] += 1
    print(f"selected {len(rows):,} rows: {dict(by)}", flush=True)
    done: set[int] = set()
    if os.path.exists(a.out):
        done = {json.loads(line)["i"] for line in open(a.out)}
        print(f"resuming: {len(done):,} already labelled", flush=True)
    todo = [r for r in rows if r[0] not in done]
    if a.limit:
        todo = todo[: a.limit]
    model, tok = T.load(a.model, a.revision)
    exs = [ex for _, _, ex in todo]
    gidx = [i for i, _, _ in todo]
    t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as fh:
        def sink(j: int, p: list[float]) -> None:
            fh.write(json.dumps({"i": gidx[j], "probs": [round(x, 6) for x in p]}) + "\n")
            fh.flush()
        T.label(model, tok, exs, max_batch_tokens=a.max_batch_tokens, max_batch=a.max_batch,
                max_tokens=a.max_tokens, sink=sink, log_every=1000)
    print(f"labelled in {(time.time() - t0) / 60:.1f} min", flush=True)


def cmd_build(a: argparse.Namespace) -> None:
    gold = {}
    src_of = {}
    task_of = {}
    for i, f, ex in corpus(TRAIN_FILES):
        gold[i] = ex.label
        src_of[i] = f
        task_of[i] = ex.task
    raw = {}
    for line in open(a.raw):
        d = json.loads(line)
        raw[d["i"]] = d["probs"]
    stats = defaultdict(lambda: [0, 0, 0.0])  # n, agree, sum of top prob when agreeing
    keep = {}
    for i, p in raw.items():
        am = max(range(len(p)), key=p.__getitem__)
        s = stats[src_of[i]]
        s[0] += 1
        if am == gold[i]:
            s[1] += 1
            s[2] += p[am]
            keep[i] = p
    for f, (n, ag, conf) in sorted(stats.items()):
        print(f"{f:<32} labelled {n:>6,}  agree {ag / max(n, 1):.3f}  "
              f"mean P(gold) where kept {conf / max(ag, 1):.3f}")
    base = {}
    if a.base:
        for line in open(a.base):
            d = json.loads(line)
            base[d["i"]] = d["probs"]
    merged = dict(base)
    replaced = sum(1 for i in keep if i in base)
    merged.update(keep)
    with open(a.out, "w", encoding="utf-8") as fh:
        for i in sorted(merged):
            fh.write(json.dumps({"i": i, "probs": merged[i]}) + "\n")
    print(f"{a.out}: {len(merged):,} rows ({len(base):,} base, {len(keep):,} kept new, "
          f"{replaced:,} base rows replaced)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("label")
    lb.add_argument("--model", default="Qwen/Qwen3.5-27B")
    lb.add_argument("--revision", default="fc05daec18b0a78c049392ed2e771dde82bdf654")
    lb.add_argument("--out", required=True)
    lb.add_argument("--sources", nargs="*", default=[os.path.basename(f) for f in TRAIN_FILES])
    lb.add_argument("--max-batch-tokens", type=int, default=32000)
    lb.add_argument("--max-batch", type=int, default=128)
    lb.add_argument("--max-tokens", type=int, default=4096)
    lb.add_argument("--limit", type=int, default=0)
    bd = sub.add_parser("build")
    bd.add_argument("--raw", required=True)
    bd.add_argument("--base", default="")
    bd.add_argument("--out", required=True)
    a = ap.parse_args()
    {"label": cmd_label, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
