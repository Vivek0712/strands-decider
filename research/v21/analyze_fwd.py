# ruff: noqa  (exploratory research script: run as-is, not library code; see research/v21/RESULTS.md)
"""Step 1 + 2 analysis from forward_all.py outputs.
Fix A per model: yes/no temperature kept; choice and score refit by NLL on the held-out FIT half.
Unseen-family ('sealed-like') v1.5-style scoring per type: noul CC with the 0.2-0.8 band counted wrong;
choice chance-corrected; score by expected level (nMAE vs chance). Intelligence = mean of the 3 types.
Also each model's public JevBench v1.5-style Intelligence (current and Fix A), for the public->unseen drop."""
import json, math, os, sys
sys.path.insert(0, "/Users/vivekrajaps/vision-decider/sd-v21/evaluation/jevbench")
import v15_proxy as V
FWD = sys.argv[1]
GRID = [round(math.exp(x / 40), 4) for x in range(-60, 81)]
def probs(lp, T):
    z = [x / T for x in lp]; m = max(z); e = [math.exp(x - m) for x in z]; s = sum(e); return [x / s for x in e]
def nll(rows, T): return sum(-math.log(max(probs(r["lp"], T)[r["gold"]], 1e-12)) for r in rows)
def fit(rows): return min(GRID, key=lambda T: nll(rows, T)) if rows else 1.0
def cc_type(rows, kind, T):
    if not rows: return None
    if kind == "noul":
        right = sum(1 for r in rows if (lambda py: not (0.2 < py < 0.8) and ((py >= 0.8) == (r["gold"] == 1)))(probs(r["lp"], T)[1]))
        return 100 * (right / len(rows) - 0.5) / 0.5
    if kind == "choice":
        acc = sum(1 for r in rows if (lambda p: p.index(max(p)) == r["gold"])(probs(r["lp"], T))) / len(rows)
        cbar = sum(1 / len(r["lp"]) for r in rows) / len(rows); return 100 * (acc - cbar) / (1 - cbar)
    num = den = 0.0
    for r in rows:
        p = probs(r["lp"], T); K = len(p); ev = sum(i * v for i, v in enumerate(p))
        num += abs(ev - r["gold"]) / (K - 1); den += sum(abs(l - r["gold"]) for l in range(K)) / K / (K - 1)
    return 100 * (1 - num / den)
def ece(rows, temps, bins=10):
    cells = [[] for _ in range(bins)]
    for r in rows:
        p = probs(r["lp"], temps[r["kind"]]); c = max(p); cells[min(bins - 1, int(c * bins))].append((c, float(p.index(c) == r["gold"])))
    n = len(rows); return sum(len(b) / n * abs(sum(x for x, _ in b) / len(b) - sum(y for _, y in b) / len(b)) for b in cells if b)
def yn_over(rows, T):
    yn = [r for r in rows if r["kind"] == "noul"]
    if not yn: return None
    c = [max(probs(r["lp"], T)) for r in yn]; a = [float(probs(r["lp"], T).index(max(probs(r["lp"], T))) == r["gold"]) for r in yn]
    return (sum(c) - sum(a)) / len(yn)
def jev_I(run_dir, t_cur, t_new):
    if not run_dir or not os.path.exists(run_dir + "/results.jsonl"): return None
    rows = [json.loads(l) for l in open(run_dir + "/results.jsonl")]; tf = run_dir + "/all.jsonl" if os.path.exists(run_dir + "/all.jsonl") else "/Users/vivekrajaps/vision-decider/checkpoints/h200-v21-results/runs/v21-minicpm5-2b-s0/all.jsonl"; tasks = {t["id"]: t for t in map(json.loads, open(tf))}  # same 231 tasks in every run
    out = []
    for r in rows:
        if r.get("probs"):
            k = V.kind(r); a = t_cur[k] / t_new[k]; w = {o: max(v, 1e-12) ** a for o, v in r["probs"].items()}; s = sum(w.values()); r = dict(r, probs={o: v / s for o, v in w.items()})
        out.append(r)
    per, _ = V.proxy(out); num = den = 0
    for r in out:
        if V.kind(r) != "score" or not r.get("probs"): continue
        g = int(tasks[r["task_id"]]["expected"]); K = len(tasks[r["task_id"]]["labels"]); ev = sum(float(k) * v for k, v in r["probs"].items())
        num += abs(ev - g) / (K - 1); den += sum(abs(l - g) for l in range(K)) / K / (K - 1)
    return (per["noul"] + per["choice"] + 100 * (1 - num / den)) / 3
A = "/Users/vivekrajaps/vision-decider/checkpoints/h200-v21-results/runs"; B = "/Users/vivekrajaps/vision-decider/checkpoints/h200nvl-machineB-partial"
JEV = {"v19": f"{A}/v19", "v21-s0": f"{A}/v21-minicpm5-2b-s0", "v21-s1": f"{A}/v21-minicpm5-2b-s1", "v21-s2": f"{A}/v21-minicpm5-2b-s2", "v21-soup": f"{A}/v21-minicpm5-2b-soup",
       "v20long-soup": f"{A}/v21-qwen35-2b-v19cont-soup", "eyes-s0": f"{B}/minicpm5-img-seed0/jevbench", "eyes-s1": f"{B}/minicpm5-img-seed1/jevbench", "eyes-s2": f"{B}/minicpm5-img-seed2/jevbench"}
os.system(f"cp {A}/v19/all.jsonl /dev/null 2>&1")
print(f"{'model':13} {'FixA T (yn/ch/sc)':24} | {'public I cur->FixA':19} | {'UNSEEN I cur->FixA':19} | unseen yn/ch/sc (FixA)   | unseen ECE cur->FixA | unseen yes/no overconf")
res = {}
for f in sorted(os.listdir(FWD)):
    if not f.endswith(".json") or f.startswith("step12"): continue
    n = f[:-5]; d = json.load(open(f"{FWD}/{f}")); tc = d["temps"]; H = d["heldout"]; U = d["unseen"]
    tn = {"noul": tc["noul"], "choice": fit([r for r in H if r["kind"] == "choice" and r["split"] == "fit"]), "score": fit([r for r in H if r["kind"] == "score" and r["split"] == "fit"])}
    def I(rows, temps):
        t = {k: cc_type([r for r in rows if r["kind"] == k], k, temps[k]) for k in ("noul", "choice", "score")}
        return sum(v for v in t.values() if v is not None) / sum(v is not None for v in t.values()), t
    Ic, _ = I(U, tc); If, tf = I(U, tn)
    pc = jev_I(JEV.get(n), tc, tc); pf = jev_I(JEV.get(n), tc, tn)
    res[n] = dict(t_fixA=tn, unseen_I_cur=Ic, unseen_I_fixA=If, unseen_types=tf, public_I_cur=pc, public_I_fixA=pf, unseen_ece_cur=ece(U, tc), unseen_ece_fixA=ece(U, tn), yn_over=yn_over(U, tc["noul"]))
    fmt = lambda x: "  -  " if x is None else f"{x:5.1f}"
    print(f"{n:13} {str({k: round(v, 2) for k, v in tn.items()}):24} | {fmt(pc)} -> {fmt(pf)}      | {Ic:5.1f} -> {If:5.1f}      | {tf['noul']:5.1f}/{tf['choice']:5.1f}/{tf['score']:5.1f}  | {res[n]['unseen_ece_cur']:.3f} -> {res[n]['unseen_ece_fixA']:.3f}     | {res[n]['yn_over']:+.3f}")
json.dump(res, open(f"{FWD}/step12_summary.json", "w"), indent=1)
