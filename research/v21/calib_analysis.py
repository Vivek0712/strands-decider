# ruff: noqa  (exploratory research script: run as-is, not library code; see research/v21/RESULTS.md)
"""Calibration steps 1-2 for text checkpoints (v19, bake-off MiniCPM5, v21 seeds and soup).

Step 1 (diagnose): per question type (noul / choice / score), on held-out rows none of the
checkpoints trained on (v19's committed eval splits + a sample of holdout_v5_norule; no JevBench
items): accuracy, mean confidence, over/under-confidence, ECE (10 bins), reliability bins --
with each checkpoint's current temperatures. Also the same on its JevBench v1 public results.

Step 2 (refit): per-type temperatures fitted by NLL on the FIT half of the held-out rows, judged
on the CHECK half, then applied to the JevBench results exactly (a softmax temperature change is
p_new ∝ p_cur^(T_cur/T_new), argmax unchanged) and re-scored with the v1.5 proxy.

    python calib_analysis.py --out /root/runs/calib CKPT_NAME=CKPT_DIR:JEVBENCH_RUN_DIR ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys

import torch

sys.path.insert(0, "/root/sd/evaluation/jevbench")
from strands_decider.data.format import Example  # noqa: E402
from strands_decider.infer import _option_token_index  # noqa: E402
from strands_decider.modeling import StrandsDeciderModel, masked_log_softmax  # noqa: E402
from strands_decider.prompting import build_prompt  # noqa: E402
import v15_proxy  # noqa: E402

D = "/root/sd/data"
FILES = {"adequacy_hs2_eval.jsonl": 0, "adequacy_gen_eval.jsonl": 0, "generated_v16_eval.jsonl": 0,
         "generated_v18_eval.jsonl": 0, "multistep_v14_eval.jsonl": 900, "holdout_v5_norule.jsonl": 1800}
KINDS = ("noul", "choice", "score")
GRID = [round(math.exp(x / 40), 4) for x in range(-60, 81)]  # T from ~0.22 to ~7.4


def rows() -> list[tuple[str, Example]]:  # noqa: D103
    out = []
    for f, cap in FILES.items():
        lines = open(os.path.join(D, f), encoding="utf-8").readlines()
        if cap and len(lines) > cap:
            lines = random.Random(0).sample(lines, cap)
        out += [(f"{f}:{i}", Example.from_dict(json.loads(l))) for i, l in enumerate(lines)]
    return out


def half(rid: str) -> str:
    return "fit" if int(hashlib.sha256(rid.encode()).hexdigest(), 16) % 2 == 0 else "check"


@torch.no_grad()
def raw_logprobs(ck: str, data: list[tuple[str, Example]]) -> tuple[list[dict], dict]:
    """Per row: log-probs at temperature 1 (raw), kind, gold, split."""
    m = StrandsDeciderModel.load(ck).to("cuda").eval()
    res = []
    for rid, ex in data:
        prompt, rq = build_prompt(ex.state, ex.to_question())
        enc = m.tokenizer(prompt, return_offsets_mapping=True)
        if len(enc["input_ids"]) > m.config.max_length or ex.kind not in KINDS:
            continue
        opt = _option_token_index(enc["offset_mapping"], rq.option_spans, len(prompt) - len(rq.text))
        ids = torch.tensor([enc["input_ids"]], device="cuda")
        n = torch.tensor([rq.n_slots], device="cuda")
        out = m(ids, torch.ones_like(ids), n, opt_idx=torch.tensor([opt], device="cuda"), temperature=1.0)
        lp = masked_log_softmax(out["logits"].float(), n)[0][: rq.n_slots]
        res.append({"id": rid, "kind": ex.kind, "gold": int(ex.label), "split": half(rid), "lp": [float(x) for x in lp]})
    temps = {k: float((m.config.temperature_by_kind or {}).get(k, m.config.temperature)) for k in KINDS}
    del m
    torch.cuda.empty_cache()
    return res, temps


def probs_at(lp: list[float], T: float) -> list[float]:
    z = [x / T for x in lp]
    mx = max(z)
    e = [math.exp(x - mx) for x in z]
    s = sum(e)
    return [x / s for x in e]


def stats(rs: list[dict], T: float, bins: int = 10) -> dict:
    if not rs:
        return {"n": 0}
    conf, acc, nll, cells = [], [], 0.0, [[] for _ in range(bins)]
    for r in rs:
        p = probs_at(r["lp"], T)
        c = max(p)
        a = float(p.index(c) == r["gold"])
        conf.append(c)
        acc.append(a)
        nll -= math.log(max(p[r["gold"]], 1e-12))
        cells[min(bins - 1, int(c * bins))].append((c, a))
    ece = sum(len(b) / len(rs) * abs(sum(x for x, _ in b) / len(b) - sum(y for _, y in b) / len(b)) for b in cells if b)
    rel = [{"bin": i / bins, "n": len(b), "conf": round(sum(x for x, _ in b) / len(b), 3), "acc": round(sum(y for _, y in b) / len(b), 3)} for i, b in enumerate(cells) if b]
    return {"n": len(rs), "acc": round(sum(acc) / len(rs), 4), "mean_conf": round(sum(conf) / len(rs), 4),
            "over_conf": round((sum(conf) - sum(acc)) / len(rs), 4), "ece": round(ece, 4),
            "nll": round(nll / len(rs), 4), "reliability": rel}


def fit_T(rs: list[dict]) -> float:
    def nll(T: float) -> float:
        return sum(-math.log(max(probs_at(r["lp"], T)[r["gold"]], 1e-12)) for r in rs)
    return min(GRID, key=nll) if rs else 1.0


def jev_rescaled(run: str, t_cur: dict, t_new: dict, out_dir: str) -> dict:
    """JevBench results with per-kind temperatures changed exactly, then the v1.5 proxy."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(run, "results.jsonl"), encoding="utf-8") as fh, \
         open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8") as oh:
        for line in fh:
            r = json.loads(line)
            p = r.get("probs") or {}
            if p:
                k = v15_proxy.kind(r)
                a = t_cur[k] / t_new[k]
                w = {o: max(v, 1e-12) ** a for o, v in p.items()}
                s = sum(w.values())
                r["probs"] = {o: round(v / s, 6) for o, v in w.items()}  # argmax (hence correct) unchanged
            oh.write(json.dumps(r) + "\n")
    for f in ("run_meta.json", "summary.json"):
        if os.path.exists(os.path.join(run, f)):
            os.system(f"cp {os.path.join(run, f)} {out_dir}/")
    rs, _ = v15_proxy.load(out_dir)
    return v15_proxy.metrics(rs)  # ece_top recomputed from the new probabilities


def jev_kind_stats(run: str) -> dict:
    out = {}
    rs = [json.loads(l) for l in open(os.path.join(run, "results.jsonl"), encoding="utf-8")]
    for k in KINDS:
        sub = [r for r in rs if r.get("probs") and v15_proxy.kind(r) == k]
        if sub:
            c = [max(r["probs"].values()) for r in sub]
            a = [float(bool(r.get("correct"))) for r in sub]
            out[k] = {"n": len(sub), "acc": round(sum(a) / len(sub), 4), "mean_conf": round(sum(c) / len(sub), 4),
                      "over_conf": round((sum(c) - sum(a)) / len(sub), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("models", nargs="+", help="NAME=CKPT_DIR:JEVBENCH_RUN_DIR")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = rows()
    report = {"heldout_rows": len(data), "files": FILES, "models": {}}
    for spec in a.models:
        name, rest = spec.split("=", 1)
        ck, run = rest.split(":", 1)
        print(f"[calib] {name}: forwarding {len(data)} held-out rows", flush=True)
        res, t_cur = raw_logprobs(ck, data)
        json.dump(res, open(os.path.join(a.out, f"{name}.heldout_logprobs.json"), "w"))
        by = {k: {s: [r for r in res if r["kind"] == k and r["split"] == s] for s in ("fit", "check")} for k in KINDS}
        t_new = {k: fit_T(by[k]["fit"]) if by[k]["fit"] else t_cur[k] for k in KINDS}
        m = {"t_current": t_cur, "t_refit": t_new,
             "step1_heldout_current": {k: stats(by[k]["check"] + by[k]["fit"], t_cur[k]) for k in KINDS},
             "step1_jevbench_current": jev_kind_stats(run),
             "step2_heldout_check": {k: {"current": stats(by[k]["check"], t_cur[k]), "refit": stats(by[k]["check"], t_new[k])} for k in KINDS}}
        m["step2_jevbench_current"] = v15_proxy.metrics(v15_proxy.load(run)[0])
        try:
            m["step2_jevbench_refit"] = jev_rescaled(run, t_cur, t_new, os.path.join(a.out, f"{name}-jev-refit"))
        except Exception as e:  # noqa: BLE001
            m["step2_jevbench_refit"] = f"error {e}"
        report["models"][name] = m
        json.dump(report, open(os.path.join(a.out, "calib_report.json"), "w"), indent=1)
        print(f"[calib] {name}: T {t_cur} -> {t_new}", flush=True)
    print("[calib] done", flush=True)


if __name__ == "__main__":
    main()
