# ruff: noqa  (exploratory research script: run as-is, not library code; see research/v21/RESULTS.md)
"""Run a competitor 2B decider, in process with its own published code and calibration, on the
unseen-family set; write rows in forward_all.py's format (lp = log of the returned probabilities,
canonical option order; temps all 1.0 since each system's own calibration is already applied).
    python comp_run.py decider REPO_DIR OUT.json     (Mapika decider-2b: decider.infer.Decider.system_one)
    python comp_run.py flymy   REPO_DIR OUT.json     (FlyMy decision-2b-preview: model.load().decide)"""
import json, math, sys
sysname, repo, out = sys.argv[1:4]
sys.path.insert(0, repo)
rows = [json.loads(l) for l in open("/root/unseen.jsonl")]

def question(r):
    names = [o[0] for o in r["options"]]; descs = [o[1] for o in r["options"]]
    if r["kind"] == "noul":
        return {"type": "noul", "instructions": r["instructions"], "criteria": {"false": descs[0], "true": descs[1]}}, ["false", "true"]
    if r["kind"] == "choice":
        return {"type": "choice", "instructions": r["instructions"], "criteria": {n: d for n, d in zip(names, descs)}}, names
    return {"type": "score", "instructions": r["instructions"], "criteria": descs}, [str(i) for i in range(len(descs))]

def probs_from(ans, kind, keys):
    if kind == "noul":
        if "noul" in ans and isinstance(ans["noul"], (int, float)): py = float(ans["noul"]); return [1 - py, py]
        p = ans.get("probabilities") or ans.get("probs") or {}
        py = p.get("true", p.get("yes")); return [1 - float(py), float(py)]
    p = ans.get("probabilities") or ans.get("probs") or ans.get("distribution")
    if p is None and kind == "score":
        p = ans.get("legend_probabilities") or ans.get("level_fit")
    if isinstance(p, list): return [float(x) for x in p]
    v = [float(p.get(k, p.get(str(k), 0.0))) for k in keys]; s = sum(v) or 1.0; return [x / s for x in v]

if sysname == "decider":
    from decider.infer import Decider
    d = Decider(repo)
    def ask(r):
        q, keys = question(r); res = d.system_one(r["state"], {"q": q}); a = res["answers"]["q"]
        return probs_from(a, r["kind"], keys), a
else:
    import model
    d = model.load()
    def ask(r):
        q, keys = question(r); a = d.decide(r["state"], q)
        return probs_from(a, r["kind"], keys), a

res, first = [], {}
for i, r in enumerate(rows):
    p, raw = ask(r)
    if r["kind"] not in first: first[r["kind"]] = raw
    res.append({"id": f"unseen:{i}", "task": r["task"], "kind": r["kind"], "gold": int(r["label"]), "split": "unseen",
                "lp": [math.log(max(x, 1e-12)) for x in p]})
    if i % 200 == 0: print(sysname, i, flush=True)
print("first raw answers:", json.dumps(first)[:900])
json.dump({"temps": {"noul": 1.0, "choice": 1.0, "score": 1.0}, "heldout": [], "unseen": res}, open(out, "w"))
print("COMP_DONE", sysname, len(res))
