# ruff: noqa  (exploratory research script: run as-is, not library code; see research/v21/RESULTS.md)
"""Raw (temperature 1) option log-probs for each model on (a) the held-out calibration rows of
calib_analysis.py and (b) the unseen-family set; plus each model's current per-kind temperatures.
Text checkpoints load with StrandsDeciderModel, vision/graft ones with load_vision_model (no image).
    python forward_all.py OUTDIR NAME=text:CKPT NAME=vision:CKPT ..."""
import json, os, sys
import torch
sys.path.insert(0, "/root/calib")
from calib_analysis import rows as heldout_rows, half, KINDS
from strands_decider.data.format import Example
from strands_decider.infer import _option_token_index
from strands_decider.modeling import StrandsDeciderModel, masked_log_softmax
from strands_decider.prompting import build_prompt

def load(kind, ck):
    if kind == "vision":
        from strands_decider.vision import load_vision_model
        return load_vision_model(ck)
    return StrandsDeciderModel.load(ck)

@torch.no_grad()
def forward(m, data, vision):
    res = []
    for rid, ex, split in data:
        prompt, rq = build_prompt(ex.state, ex.to_question())
        enc = m.tokenizer(prompt, return_offsets_mapping=True)
        if len(enc["input_ids"]) > m.config.max_length or ex.kind not in KINDS: continue
        opt = _option_token_index(enc["offset_mapping"], rq.option_spans, len(prompt) - len(rq.text))
        ids = torch.tensor([enc["input_ids"]], device="cuda"); n = torch.tensor([rq.n_slots], device="cuda")
        if vision and hasattr(m, "torso"):
            try:
                from strands_decider.vision import qwen_base
                qwen_base(m.torso).rope_deltas = None
            except Exception:
                pass
        out = m(ids, torch.ones_like(ids), n, opt_idx=torch.tensor([opt], device="cuda"), temperature=1.0)
        lp = masked_log_softmax(out["logits"].float(), n)[0][: rq.n_slots]
        res.append({"id": rid, "task": ex.task, "kind": ex.kind, "gold": int(ex.label), "split": split, "lp": [float(x) for x in lp]})
    return res

out_dir = sys.argv[1]; os.makedirs(out_dir, exist_ok=True)
held = [(rid, ex, half(rid)) for rid, ex in heldout_rows()]
unseen = [(f"unseen:{i}", Example.from_dict(json.loads(l)), "unseen") for i, l in enumerate(open("/root/unseen.jsonl"))]
for spec in sys.argv[2:]:
    name, rest = spec.split("=", 1); kind, ck = rest.split(":", 1)
    if os.path.exists(f"{out_dir}/{name}.json"): print("skip", name); continue
    m = load(kind, ck).to("cuda").eval()
    temps = {k: float((m.config.temperature_by_kind or {}).get(k, m.config.temperature)) for k in KINDS}
    r = {"temps": temps, "heldout": forward(m, held, kind == "vision"), "unseen": forward(m, unseen, kind == "vision")}
    json.dump(r, open(f"{out_dir}/{name}.json", "w")); print(name, "temps", temps, "heldout", len(r["heldout"]), "unseen", len(r["unseen"]), flush=True)
    del m; torch.cuda.empty_cache()
print("FORWARD_DONE", flush=True)
