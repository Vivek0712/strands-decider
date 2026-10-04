"""The two other torso families, Llama (MiniCPM5) and Gemma 4, on the tiny random-weight
bases of tests/tiny_bases.py (nothing is downloaded): the torso loads without a vision
tower, LoRA reaches the modules it should, the frozen readout is the LM's own
option-number distribution, training moves the loss, a checkpoint round-trips, and the
shared-prefix path answers as encoding each full prompt does.
"""

from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace

import pytest

transformers = pytest.importorskip("transformers")
if tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2]) < (5, 18):
    pytest.skip("the tiny Gemma 4 needs transformers >= 5.18", allow_module_level=True)

import torch  # noqa: E402
from tiny_bases import ATTENTION_MLP, save_gemma4, save_llama  # noqa: E402

from strands_decider.data.collate import CollatorConfig, SystemOneCollator  # noqa: E402
from strands_decider.data.format import Example, write_jsonl  # noqa: E402
from strands_decider.infer import EngineConfig, SystemOneEngine  # noqa: E402
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel  # noqa: E402
from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion  # noqa: E402
from strands_decider.train import TrainConfig, train  # noqa: E402

TORSOS = {"llama": "LlamaModel", "gemma4": "Gemma4TextModel"}
# Gemma 4's last num_kv_shared_layers layers reuse earlier layers' keys and values and have
# no k_proj / v_proj of their own; the tiny Gemma shares layers 2 and 3.
LORA = {"llama": {(i, m) for i in range(2) for m in ATTENTION_MLP},
        "gemma4": {(i, m) for i in range(4) for m in ATTENTION_MLP
                   if i < 2 or m not in ("k_proj", "v_proj")}}


@pytest.fixture(scope="module", params=["llama", "gemma4"])
def base(request, tmp_path_factory):
    save = {"llama": save_llama, "gemma4": save_gemma4}[request.param]
    return request.param, save(str(tmp_path_factory.mktemp(request.param)))


def _model(base_dir: str) -> StrandsDeciderModel:
    cfg = StrandsDeciderConfig(base_model=base_dir, head_type="pointer", pointer_dim=16,
                               torch_dtype="float32", max_length=512, lora_targets=ATTENTION_MLP)
    return StrandsDeciderModel.from_pretrained_base(cfg).eval()


def _rows(n: int) -> list[Example]:
    out = []
    for k in range(n):
        if k % 2:
            out.append(Example(kind="noul", state=f"Order {k} shipped late.", instructions="Was it late?",
                               options=[["false", "on time"], ["true", "late"]], label=1, task="late"))
        else:
            out.append(Example(kind="choice", state=f"Ticket {k}: refund please.", instructions="Which team?",
                               options=[["billing", ""], ["shipping", ""], ["sales", ""]], label=0,
                               task="route"))
    return out


def _batch(model: StrandsDeciderModel) -> dict[str, torch.Tensor]:
    coll = SystemOneCollator(model.tokenizer, CollatorConfig(max_length=512, head_type="pointer"),
                             train=False)
    return coll(_rows(4))


def _lora_modules(model: StrandsDeciderModel) -> set[tuple[int, str]]:
    found = set()
    for name, _ in model.torso.named_modules():
        m = re.search(r"layers\.(\d+)\..*\.(\w+)\.lora_A$", name)
        if m:
            found.add((int(m.group(1)), m.group(2)))
    return found


def test_the_text_decoder_loads_alone(base):
    family, d = base
    model = _model(d)
    assert type(model.torso.base_model.model).__name__ == TORSOS[family]
    assert not any("vision" in n or "audio" in n for n, _ in model.torso.named_parameters())
    assert model.hidden_size(model.torso) == 64
    ids = model.tokenizer("<state>")["input_ids"]
    assert ids[0] == model.tokenizer.bos_token_id  # both families' tokenisers prepend BOS
    assert not StrandsDeciderModel.is_hybrid(model.torso)


def test_gemma4_decoder_weights_are_the_checkpoints(tmp_path):
    d = save_gemma4(str(tmp_path))
    full = transformers.Gemma4ForConditionalGeneration.from_pretrained(d).model.language_model
    cfg = StrandsDeciderConfig(base_model=d, torch_dtype="float32", use_lora=False)
    torso = StrandsDeciderModel.from_pretrained_base(cfg).torso
    want = full.state_dict()
    got = torso.state_dict()
    assert got.keys() == want.keys()
    for k in want:
        torch.testing.assert_close(got[k], want[k], rtol=0, atol=0, msg=k)


def test_lora_reaches_attention_and_mlp_of_every_layer(base):
    family, d = base
    assert _lora_modules(_model(d)) == LORA[family]


@torch.no_grad()
def test_forward_shapes(base):
    _, d = base
    model = _model(d)
    b = _batch(model)
    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], n_slots=b["n_slots"],
                opt_idx=b["opt_idx"], labels=b["labels"])
    assert out["log_probs"].shape == (4, 3)
    assert torch.isfinite(out["loss"])
    p = out["log_probs"].exp()
    torch.testing.assert_close(p.sum(-1), torch.ones(4))
    assert torch.equal(p[1::2, 2], torch.zeros(2))  # the yes/no rows' missing third option


@torch.no_grad()
def test_frozen_readout_is_the_lms_own_option_number_distribution(base):
    """The untied Llama head is read from the checkpoint, Gemma's logits are soft-capped."""
    family, d = base
    model = _model(d)
    b = _batch(model)
    lp, eligible = model.frozen_slot_log_probs(b["input_ids"], b["attention_mask"], b["n_slots"])
    assert bool(eligible.all())
    lm = (transformers.LlamaForCausalLM if family == "llama"
          else transformers.Gemma4ForConditionalGeneration).from_pretrained(d).eval()
    logits = lm(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
    last = logits[torch.arange(4), b["attention_mask"].sum(1) - 1]
    ids = list(model.slot_token_ids().values())
    for r in range(4):
        n = int(b["n_slots"][r])
        torch.testing.assert_close(lp[r, :n], last[r, ids[:n]].log_softmax(-1), atol=1e-5, rtol=1e-5)


def test_option_numbers_past_nine_are_not_read():
    """A vocabulary with single tokens for 10-24 (MiniCPM5's) still reads slots 0-8 only."""
    stub = SimpleNamespace(config=SimpleNamespace(num_slots=24),
                           tokenizer=SimpleNamespace(encode=lambda s, add_special_tokens: [int(s)]))
    assert StrandsDeciderModel.slot_token_ids(stub) == {k: k + 1 for k in range(9)}


def test_three_steps_from_the_raw_base_reduce_the_loss(base, tmp_path):
    _, d = base
    write_jsonl(str(tmp_path / "train.jsonl"), _rows(8))
    write_jsonl(str(tmp_path / "val.jsonl"), _rows(2))
    cfg = TrainConfig(base_model=d, train_files=[str(tmp_path / "train.jsonl")],
                      val_files=[str(tmp_path / "val.jsonl")], head_type="pointer", pointer_dim=16,
                      lora_targets=ATTENTION_MLP, max_length=512, kl_frozen_weight=0.3,
                      kl_frozen_skip_kinds=["noul"], micro_batch_size=8, grad_accum=1, epochs=3, max_steps=3,
                      lr=1e-2, head_lr=1e-2, log_every=1, eval_every=0, shuffle_options=False,
                      output_dir=str(tmp_path / "out"))
    out = train(cfg)
    with open(os.path.join(out, "history.json"), encoding="utf-8") as fh:
        losses = [h["loss"] for h in json.load(fh) if "loss" in h]
    assert len(losses) == 3 and losses[-1] < losses[0]
    loaded = StrandsDeciderModel.load(out)
    assert _lora_modules(loaded) == _lora_modules(_model(d))


@torch.no_grad()
def test_checkpoint_round_trip(base, tmp_path):
    _, d = base
    model = _model(d)
    for n, p in model.torso.named_parameters():
        if "lora_B" in n:
            p.normal_(0, 0.05)
    model.save_pretrained(str(tmp_path / "ck"))
    loaded = StrandsDeciderModel.load(str(tmp_path / "ck"))
    loaded.torso.to(torch.float32)
    b = _batch(model)
    kw = {k: b[k] for k in ("input_ids", "attention_mask", "n_slots", "opt_idx")}
    torch.testing.assert_close(loaded(**kw)["log_probs"], model(**kw)["log_probs"])


def test_shared_prefix_answers_as_full_prompts_do(base):
    _, d = base
    model = _model(d)
    with torch.no_grad():
        for n, p in model.torso.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    # Longer than the tiny Gemma's 8-token sliding window, so its cache has evicted tokens.
    state = {"ticket": "My card was charged twice for order 123. Please refund one charge.",
             "customer": "since 2019"}
    questions = {
        "route": ChoiceQuestion(instructions="Route it.",
                                criteria={"billing": "money", "shipping": "", "other": "else"}),
        "refund": NoulQuestion(instructions="The customer wants a refund."),
        "urgency": ScoreQuestion(instructions="How urgent?", criteria=["low", "mid", "high"]),
    }
    shared_engine = SystemOneEngine(model, EngineConfig(device="cpu", use_prefix_cache=True))
    shared = shared_engine.ask(state, questions)
    assert shared_engine.cfg.use_prefix_cache  # the cache forked; no fallback
    full = SystemOneEngine(model, EngineConfig(device="cpu", use_prefix_cache=False)).ask(state, questions)
    assert shared.answers["route"].probabilities == pytest.approx(full.answers["route"].probabilities, abs=1e-3)
    assert shared.answers["refund"].noul == pytest.approx(full.answers["refund"].noul, abs=1e-3)
    assert shared.answers["urgency"].probabilities == pytest.approx(full.answers["urgency"].probabilities,
                                                                    abs=1e-3)


def test_a_checkpoint_naming_the_revision_base_model_revision_loads(tmp_path):
    """Checkpoints saved while the field was called `base_model_revision` keep their pin."""
    path = tmp_path / "strands_decider_config.json"
    path.write_text(json.dumps({"base_model": "org/base", "base_model_revision": "abc123"}))
    assert StrandsDeciderConfig.from_json(str(path)).base_revision == "abc123"
    path.write_text(json.dumps({"base_model": "org/base", "base_revision": "def456"}))
    assert StrandsDeciderConfig.from_json(str(path)).base_revision == "def456"
