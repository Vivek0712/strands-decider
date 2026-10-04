# ruff: noqa  (exploratory research script: run as-is, not library code; see research/v21/RESULTS.md)
"""Balanced yes/no temperature: maximise (CC_noul + C_noul)/2 on held-out yes/no rows, where
CC_noul = 100*(acc_with_band - 0.5)/0.5 (0.2<P(yes)<0.8 counts wrong, as v1.5) and
C_noul = 100*(1 - 2*ECE(P(yes))) (an approximation of the board's noul calibration). Score/choice
keep the NLL refit. Then the JevBench results are rescaled exactly and re-scored with the v1.5 proxy."""
import json, math, os, sys
sys.path.insert(0, "/root/calib"); sys.path.insert(0, "/root/sd/evaluation/jevbench")
import v15_proxy
from calib_analysis import probs_at, jev_rescaled, GRID
R = json.load(open("/root/runs/calib/calib_report.json"))
RUNS = {"v21-s0": "v21-minicpm5-2b-s0", "v21-s1": "v21-minicpm5-2b-s1", "v21-s2": "v21-minicpm5-2b-s2",
        "v21-soup": "v21-minicpm5-2b-soup", "bakeoff-minicpm5": "bakeoff-minicpm5-2b", "v19": "v19"}
def yes_index(r):  # noul rows: option order is (yes, no) in canonical rendering? find via gold label convention
    return 0
def noul_obj(rows, T):
    bins = [[] for _ in range(10)]; right = 0
    for r in rows:
        p = probs_at(r["lp"], T); py = p[1]; gold_yes = r["gold"] == 1  # noul options are (false, true)
        if not (0.2 < py < 0.8) and ((py >= 0.8) == gold_yes): right += 1
        bins[min(9, int(py * 10))].append((py, 1.0 if gold_yes else 0.0))
    n = len(rows); ece = sum(len(b) / n * abs(sum(x for x, _ in b) / len(b) - sum(y for _, y in b) / len(b)) for b in bins if b)
    cc = 100 * (right / n - 0.5) / 0.5; c = 100 * (1 - 2 * ece)
    return (cc + c) / 2, cc, c
out = {}
for name, run in RUNS.items():
    m = R["models"][name]; lp = json.load(open(f"/root/runs/calib/{name}.heldout_logprobs.json"))
    fit = [r for r in lp if r["kind"] == "noul" and r["split"] == "fit"]; chk = [r for r in lp if r["kind"] == "noul" and r["split"] == "check"]
    tb = max(GRID, key=lambda T: noul_obj(fit, T)[0])
    t_new = dict(m["t_refit"]); t_new["noul"] = tb
    cur = noul_obj(chk, m["t_current"]["noul"]); bal = noul_obj(chk, tb); nll = noul_obj(chk, m["t_refit"]["noul"])
    j = jev_rescaled(f"/root/runs/{run}", m["t_current"], t_new, f"/root/runs/calib/{name}-jev-balanced")
    a = m["step2_jevbench_current"]
    out[name] = {"t_yesno_current": round(m["t_current"]["noul"], 3), "t_yesno_balanced": tb, "t_yesno_nll": m["t_refit"]["noul"],
                 "heldout_check_obj(cur/bal/nll)": [round(cur[0], 1), round(bal[0], 1), round(nll[0], 1)],
                 "jev_I(cur->bal)": [a["intelligence_proxy"], j["intelligence_proxy"]], "jev_band(cur->bal)": [a["yes_no_in_band"], j["yes_no_in_band"]],
                 "jev_ece_top(cur->bal)": [a["ece_top"], j["ece_top"]], "t_all_balanced": t_new}
    print(name, json.dumps(out[name]))
json.dump(out, open("/root/runs/calib/balanced.json", "w"), indent=1)
