"""Score run.py result rows without torch, and print the compact table the LOG uses.

    python research/image-track/score.py <run_dir> [<run_dir> ...]

`score_runs` mirrors run.py's `score` (kept torch-free so it runs anywhere).
IJB "exact" is the 60 faithfully rebuilt preview items (ArxivQA, CLEVR-HOPE, Geometry3K).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../evaluation/vision"))
from metrics import image_dependence, naturalbench_paired, summarise  # noqa: E402


def score_runs(all_res: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tag, rs in all_res.items():
        by: dict[str, list[dict[str, Any]]] = {}
        for r in rs:
            by.setdefault(r["bench"], []).append(r)
        out[tag] = {b: summarise(v) for b, v in by.items()}
        if by.get("naturalbench"):
            out[tag]["naturalbench"]["paired"] = naturalbench_paired(by["naturalbench"])
        ijb = by.get("ijb_preview")
        if ijb:
            block = out[tag]["ijb_preview"]
            block["exact_only"] = summarise([r for r in ijb if r.get("exact")])["all"]
            per: dict[str, list[dict[str, Any]]] = {}
            for r in ijb:
                per.setdefault(r["dataset"], []).append(r)
            block["by_dataset"] = {d: summarise(v)["all"] for d, v in sorted(per.items())}
    for tag in list(all_res):
        if not tag.endswith("-blind") and f"{tag}-blind" in all_res:
            out[tag]["image_dependence"] = image_dependence(all_res[tag], all_res[f"{tag}-blind"])
    return out


def load_run(d: str) -> dict[str, list[dict[str, Any]]]:
    res = {}
    for name in sorted(os.listdir(d)):
        if name.endswith(".jsonl"):
            with open(os.path.join(d, name)) as fh:
                res[name[:-6]] = [json.loads(x) for x in fh if x.strip()]
    return res


def row(d: str) -> dict[str, Any]:
    sc = score_runs(load_run(d))
    main = next(t for t in sc if not t.endswith("-blind"))
    m, b = sc[main], sc.get(main + "-blind", {})
    out: dict[str, Any] = {"run": os.path.basename(d.rstrip("/"))}
    if "naturalbench" in m:
        nb = m["naturalbench"]
        out.update(nb_acc=nb["all"]["accuracy"], nb_g=nb["paired"]["g_acc"], nb_ece=nb["all"]["ece"])
    if "pope_adversarial" in m:
        p = m["pope_adversarial"]["all"]
        out.update(pope_acc=p["accuracy"], pope_brier=p["brier"], pope_ece=p["ece"])
    if "ijb_preview" in m:
        ij = m["ijb_preview"]
        out.update(ijb_exact=f'{round(ij["exact_only"]["accuracy"] * ij["exact_only"]["n"])}/{ij["exact_only"]["n"]}',
                   ijb_all=ij["all"]["accuracy"], ijb_ece=ij["all"]["ece"], ijb_exact_ece=ij["exact_only"]["ece"],
                   ijb_fam={k: v["accuracy"] for k, v in ij["by_dataset"].items()})
    for bench, key in (("naturalbench", "nb"), ("pope_adversarial", "pope")):
        if bench in b:
            out[f"blind_{key}_conf"] = b[bench]["all"]["mean_confidence"]
            out[f"blind_{key}_ece"] = b[bench]["all"]["ece"]
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(row(d)))
