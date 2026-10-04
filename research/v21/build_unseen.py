# ruff: noqa  (exploratory research script: run as-is, not library code; see research/v21/RESULTS.md)
"""Private 'sealed-like' text set from public datasets NOT in any training mix (data/sources.md):
StrategyQA (noul), RuleTaker depth-5/NatLang (noul, held out of training), CommonsenseQA and
ARC-Challenge (choice), STS-B (score, 6 levels). ~250 rows per family, seed 0. Rows use the repo's
Example format so the server renders them exactly as training/serving do."""
import json, random, sys
from datasets import load_dataset
rng = random.Random(0); out = []
YN = [["false", "The answer is no."], ["true", "The answer is yes."]]
def add(r): out.append(r)
# StrategyQA (implicit multi-step yes/no); question only, no facts given
for name in ["ChilleD/StrategyQA", "wics/strategy-qa"]:
    try:
        ds = load_dataset(name, split="train"); break
    except Exception as e: print("strategyqa", name, "failed:", str(e)[:80]); ds = None
if ds is not None:
    rows = list(ds); rng.shuffle(rows)
    for r in rows[:250]:
        q = r.get("question"); a = r.get("answer")
        if q is None or a is None: continue
        add({"kind": "noul", "state": q, "instructions": "Is the true answer to this question yes?", "options": YN, "label": int(bool(a)), "task": "unseen/strategyqa"})
# RuleTaker held-out depths (never trained on)
n = 0
for l in open("data/train_v5.holdout.jsonl"):
    r = json.loads(l)
    if "ruletaker" in r.get("task", "") and ("d5" in r["task"] or "natlang" in r["task"]) and rng.random() < 0.5 and n < 250:
        r["task"] = "unseen/" + r["task"]; add(r); n += 1
# CommonsenseQA (5-way) and ARC-Challenge
cq = list(load_dataset("tau/commonsense_qa", split="validation")); rng.shuffle(cq)
for r in cq[:150]:
    labs = r["choices"]["label"]; txt = r["choices"]["text"]
    add({"kind": "choice", "state": r["question"], "instructions": "Which answer is most plausible?", "options": [[t, t] for t in txt], "label": labs.index(r["answerKey"]), "task": "unseen/commonsenseqa"})
arc = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")); rng.shuffle(arc)
for r in arc[:150]:
    labs = r["choices"]["label"]; txt = r["choices"]["text"]
    if r["answerKey"] not in labs: continue
    add({"kind": "choice", "state": r["question"], "instructions": "Which answer is correct?", "options": [[t, t] for t in txt], "label": labs.index(r["answerKey"]), "task": "unseen/arc_challenge"})
# STS-B similarity -> 6 ordinal levels (0..5)
LV = ["completely different meaning", "mostly different, share a topic", "not equivalent, share some details", "roughly equivalent, some important details differ", "mostly equivalent, minor details differ", "completely equivalent in meaning"]
for name, cfg, split in [("sentence-transformers/stsb", None, "test"), ("mteb/stsbenchmark-sts", None, "test")]:
    try:
        st = list(load_dataset(name, cfg, split=split)) if cfg else list(load_dataset(name, split=split)); break
    except Exception as e: print("stsb", name, "failed:", str(e)[:80]); st = []
rng.shuffle(st)
for r in st[:250]:
    s = r.get("score"); s = s * 5 if s is not None and s <= 1.0 else s
    if s is None: continue
    add({"kind": "score", "state": f"Sentence A: {r['sentence1']}\nSentence B: {r['sentence2']}", "instructions": "How similar in meaning are sentence A and sentence B?", "options": [[str(i), d] for i, d in enumerate(LV)], "label": int(round(s)), "task": "unseen/stsb"})
from collections import Counter
print(Counter(r["task"] for r in out))
with open(sys.argv[1], "w") as fh:
    for r in out: fh.write(json.dumps(r) + "\n")
