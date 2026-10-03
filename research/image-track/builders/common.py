"""Shared row format for the image training set.

A row is the in-tree `Example` (kind, state, instructions, options, label, task,
weight, instruction_variants) plus:
  images      image paths (relative to the data root), placed inside <state>
  ablation    True for an image-removed copy: weight 0, trained only by KL to the
              frozen torso's reading of the same text-only prompt
  source      dataset the row came from (its licence is in SOURCES)
  source_id   what the row was derived from (image id / task id), so a row, its pair
              and its ablation copy stay on one side of the val split
  pair_id     ties the two halves of a minimal pair (same question, other image/answer)
"""

from __future__ import annotations

import json
import random
from typing import Any

SOURCES = {
    "vqav2": "VQAv2 annotations CC BY 4.0 (visualqa.org); images COCO train2014 (COCO terms: "
             "annotations CC BY 4.0, images under their Flickr licences)",
    "coco_count": "COCO train2014 instance annotations CC BY 4.0; images COCO train2014",
    "mind2web": "Multimodal-Mind2Web TRAIN split (HF osunlp/Multimodal-Mind2Web card: openrail; "
                "Mind2Web data CC BY 4.0); markers drawn by us",
    "charts": "own synthetic matplotlib renders (no third-party data)",
    "docs": "own synthetic document renders (no third-party data)",
    "tabfact": "TabFact train split (wenhuchen/Table-Fact-Checking, MIT; tables from Wikipedia "
               "CC BY-SA); rendered by us",
    "text_replay": "v19 training rows committed in this repo (data/synthetic, Apache-2.0)",
}


# What the server renders for a noul asked without criteria (prompting.NOUL_DEFAULT_CRITERIA):
# image nouls are trained exactly as they are served.
NOUL_DEFAULT = [["false", "the statement does not hold for this state"],
                ["true", "the statement holds for this state"]]


def row(kind: str, instructions: str, options: list[list[str]], label: int, *, task: str,
        source: str, images: list[str], source_id: str, state: str = "",
        variants: list[str] | None = None, pair_id: str | None = None) -> dict[str, Any]:
    assert 0 <= label < len(options), (label, options)
    assert len({o[0] for o in options}) == len(options), options
    return {"kind": kind, "state": state, "instructions": instructions, "options": options,
            "label": label, "task": task, "weight": 1.0,
            "instruction_variants": variants or [], "images": images, "ablation": False,
            "source": source, "source_id": source_id, "pair_id": pair_id}


def yesno(question: str, yes: bool, rng: random.Random, **kw: Any) -> dict[str, Any]:
    """A yes/no question, rendered as a noul (default criteria, as served) or as a
    yes/no choice (the form Image JevBench and NaturalBench's MC use)."""
    if rng.random() < 0.6:
        return row("noul", question, NOUL_DEFAULT, int(yes), **kw)
    opts = [["yes", "Yes"], ["no", "No"]]
    return row("choice", question, opts, 0 if yes else 1, **kw)


def write(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[build] {path}: {len(rows)} rows")
