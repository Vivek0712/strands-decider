"""Image input on a Gemma 4 torso (its training: test_vision_train_gemma.py), on the tiny
random-weight Gemma 4 of tests/tiny_bases.py (`save_gemma4(images=True)`: a 4-layer decoder with sliding, full and
KV-shared layers, a 1-layer vision tower, a 1-layer audio tower, a real GemmaTokenizer with
Gemma 4's image tokens, and a Gemma4ImageProcessorPil). Nothing is downloaded. The tests pin
what the feature promises:

* a TEXT Gemma checkpoint loads onto Gemma4ForConditionalGeneration's torso (vision tower and
  projection kept and frozen, audio tower never built), its adapter remapped onto the same
  decoder, LoRA on exactly the text checkpoint's modules;
* text requests to the vision engine get the text engine's answers;
* images enter as `<|image>` + one `<|image|>` per soft token the processor reports +
  `<image|>`, and the shared-prefix path over one or two images (a prefix far longer than the
  sliding window, forked across three questions) equals a plain full forward;
* the frozen KL reference with an image is the LM's own soft-capped reading;
* a checkpoint saves and loads back to the same answers;
* evaluation (run.py's strands and untrained systems, text_check.py) and `serve --vision`
  work for a Gemma checkpoint.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys

import pytest

transformers = pytest.importorskip("transformers")
PIL = pytest.importorskip("PIL")
if tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2]) < (5, 18):
    pytest.skip("image input needs transformers >= 5.18", allow_module_level=True)

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from tiny_bases import (  # noqa: E402
    ATTENTION_MLP,
    GEMMA_IMAGE_TOKENS,
    save_gemma4,
    save_gemma4_checkpoint,
)

from strands_decider.infer import EngineConfig, SystemOneEngine, _option_token_index  # noqa: E402
from strands_decider.modeling import (  # noqa: E402
    StrandsDeciderConfig,
    StrandsDeciderModel,
    masked_log_softmax,
)
from strands_decider.prompting import render_question  # noqa: E402
from strands_decider.schema import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)
from strands_decider.vision import (  # noqa: E402
    GEMMA,
    GemmaImages,
    VisionDeciderModel,
    VisionEngine,
    VisionEngineConfig,
    decode_image,
    fit_image,
    load_image_processor,
)

BOI, IMAGE, EOI = GEMMA_IMAGE_TOKENS
# The tiny Gemma's last two layers share earlier layers' keys and values: no k_proj / v_proj.
LORA = {(i, m) for i in range(4) for m in ATTENTION_MLP if i < 2 or m not in ("k_proj", "v_proj")}
STATE = "Help! My payouts have been failing for 3 days."
QUESTIONS = {
    "signed": NoulQuestion(instructions="Is the form signed?"),
    "button": ChoiceQuestion(instructions="Which control next?",
                             criteria={"submit": "the blue submit button", "cancel": "the grey cancel link"}),
    "sharp": ScoreQuestion(instructions="How sharp?", criteria=["blurred", "soft", "sharp"]),
}


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory):
    return save_gemma4(str(tmp_path_factory.mktemp("tiny-gemma4")), images=True)


@pytest.fixture(scope="module")
def ckpt(base_dir, tmp_path_factory):
    return save_gemma4_checkpoint(base_dir, str(tmp_path_factory.mktemp("gemma-text-ckpt")))


@pytest.fixture(scope="module")
def engines(ckpt):
    text = SystemOneEngine(StrandsDeciderModel.load(ckpt), EngineConfig(device="cpu"))
    vision = VisionEngine(VisionDeciderModel.load(ckpt), VisionEngineConfig(device="cpu"))
    return text, vision


def _b64(w: int, h: int, seed: int) -> str:
    g = torch.Generator().manual_seed(seed)
    buf = io.BytesIO()
    Image.fromarray((torch.rand(h, w, 3, generator=g) * 255).to(torch.uint8).numpy()).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _lora_modules(torso) -> set[tuple[int, str]]:
    found = set()
    for name, _ in torso.named_modules():
        m = re.search(r"language_model\.layers\.(\d+)\..*\.(\w+)\.lora_A$", name)
        if m:
            found.add((int(m.group(1)), m.group(2)))
    return found


# ---- the torso ------------------------------------------------------------------------


def test_multimodal_torso_without_audio_and_frozen_vision(ckpt):
    for trainable in (False, True):
        model = VisionDeciderModel.load(ckpt, trainable=trainable)
        base = model.torso.base_model.model
        assert model.family == GEMMA
        assert type(base).__name__ == "Gemma4Model"
        assert base.audio_tower is None and base.embed_audio is None
        assert not any("audio" in n for n, _ in model.torso.named_parameters())
        frozen = list(base.vision_tower.parameters()) + list(base.embed_vision.parameters())
        assert frozen and not any(p.requires_grad for p in frozen)
        assert {p.requires_grad for n, p in model.torso.named_parameters() if "lora_" in n} == {trainable}
        assert not any(p.requires_grad for n, p in model.torso.named_parameters() if "lora_" not in n)
        assert model.hidden_size(model.torso) == 64
        assert not model.is_hybrid(model.torso)  # sliding-window layers are attention


def test_torso_weights_are_the_checkpoints(base_dir):
    """Decoder, vision tower and projection exactly as Gemma4ForConditionalGeneration loads them."""
    cfg = StrandsDeciderConfig(base_model=base_dir, torch_dtype="float32", use_lora=False)
    torso = VisionDeciderModel._load_torso(cfg, None, None)
    # frozen by the loader itself, not only by PEFT (a checkpoint without LoRA trains the rest)
    assert not any(p.requires_grad for n, p in torso.named_parameters() if n.startswith(("vision_tower.", "embed_vision.")))
    assert all(p.requires_grad for p in torso.language_model.parameters())
    want = transformers.Gemma4ForConditionalGeneration.from_pretrained(base_dir).model.state_dict()
    audio = {k for k in want if k.startswith(("audio_tower.", "embed_audio."))}
    assert audio  # the base ships an audio tower, which the image path leaves unread
    want = {k: v for k, v in want.items() if k not in audio}
    got = torso.state_dict()
    assert got.keys() == want.keys()
    assert any(k.startswith("vision_tower.") for k in got) and any(k.startswith("embed_vision.") for k in got)
    for k in want:
        torch.testing.assert_close(got[k], want[k], rtol=0, atol=0, msg=k)


def test_lora_reaches_exactly_the_text_checkpoints_modules(ckpt, engines):
    text, vision = engines
    lora = [n for n, _ in vision.model.torso.named_modules() if n.endswith("lora_A")]
    assert lora and all(".language_model." in n for n in lora)
    assert _lora_modules(vision.model.torso) == LORA
    assert {(int(re.search(r"layers\.(\d+)\.", n).group(1)), re.search(r"\.(\w+)\.lora_A$", n).group(1))
            for n, _ in text.model.torso.named_modules() if n.endswith("lora_A")} == LORA


def test_text_adapter_is_remapped_onto_the_multimodal_decoder(ckpt, engines):
    _, vision = engines
    saved = load_file(os.path.join(ckpt, "lora", "adapter_model.safetensors"))
    assert saved and not any(".language_model." in k for k in saved)  # a text checkpoint
    params = dict(vision.model.torso.named_parameters())
    for k, v in saved.items():
        name = k.replace("base_model.model.", "base_model.model.language_model.", 1)
        name = name.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
        torch.testing.assert_close(params[name], v, rtol=0, atol=0, msg=k)


def test_processor_is_pinned_to_pil(engines, base_dir):
    _, vision = engines
    assert type(vision.image_processor).__name__ == "Gemma4ImageProcessorPil"
    assert vision.image_processor.max_soft_tokens == 70  # read from the base, not a default
    cfg = StrandsDeciderConfig(base_model=base_dir)
    assert type(load_image_processor(cfg)).__name__ == "Gemma4ImageProcessorPil"


# ---- the prompt -------------------------------------------------------------------------


def test_image_state_layout():
    prompt = GemmaImages(None)
    assert prompt.state("two pages", [1, 1]) == (
        "<state>\n<|image><|image|><image|>\n<|image><|image|><image|>\ntwo pages\n</state>\n")
    assert prompt.state("", [1]) == "<state>\n<|image><|image|><image|>\n</state>\n"
    assert prompt.state("text", []) == "<state>\ntext\n</state>\n"
    two = prompt.state("", [3, 2])
    assert two == f"<state>\n{BOI}{IMAGE * 3}{EOI}\n{BOI}{IMAGE * 2}{EOI}\n</state>\n"


@pytest.mark.parametrize("sizes", [[(200, 100)], [(64, 48), (50, 300)]])
def test_image_token_counts_are_what_the_vision_tower_emits(engines, sizes):
    """One `<|image|>` per soft token, each image between its own `<|image>` and `<image|>`;
    the counts are the processor's, and the vision tower fills exactly that many."""
    _, vision = engines
    tok, cfg = vision.tok, vision.model.torso.config
    images = [Image.new("RGB", s, (30 * i, 90, 160)) for i, s in enumerate(sizes)]
    counts, mm = vision.prompter.process(images)
    assert set(mm) == {"pixel_values", "image_position_ids"}
    # 3 x 3 pooled 16 px patches of the resized image: the real patches over 9
    real = (mm["image_position_ids"] != -1).all(-1).sum(-1)
    assert counts == (real // 9).tolist() and all(0 < c <= 70 for c in counts)
    ids = tok(vision.prompter.state("x", counts))["input_ids"]
    assert ids.count(cfg.image_token_id) == sum(counts)
    assert ids.count(cfg.boi_token_id) == ids.count(cfg.eoi_token_id) == len(images)
    for k, n in enumerate(counts):  # each run of placeholders sits between its own markers
        start = [i for i, t in enumerate(ids) if t == cfg.boi_token_id][k]
        assert ids[start + 1 : start + 1 + n] == [cfg.image_token_id] * n and ids[start + 1 + n] == cfg.eoi_token_id
    with torch.no_grad():
        out = vision.model.torso(input_ids=torch.tensor([ids]), **mm)
    assert out.image_hidden_states.shape[0] == sum(counts)


# ---- answers ----------------------------------------------------------------------------


def test_text_requests_unchanged(engines):
    text, vision = engines
    req = SystemOneRequest(state=STATE, questions=QUESTIONS)
    assert text.evaluate(req).model_dump()["answers"] == vision.evaluate(req).model_dump()["answers"]


def _reference(vision: VisionEngine, state: str, images: list[str], q) -> list[float]:
    """Plain full forward of state + one question, positions left to transformers."""
    rq = render_question(q)
    counts, mm = vision.prompter.process([fit_image(decode_image(b), 448) for b in images])
    text = vision.prompter.state(state, counts) + rq.text
    enc = vision.tok(text, return_offsets_mapping=True)
    opt = _option_token_index(enc["offset_mapping"], rq.option_spans, len(text) - len(rq.text))
    ids = torch.tensor([enc["input_ids"]])
    with torch.no_grad():
        out = vision.model(ids, torch.ones_like(ids), torch.tensor([rq.n_slots]), opt_idx=torch.tensor([opt]),
                           temperature=vision._temperatures([rq.kind]), **mm)
    return masked_log_softmax(out["logits"], torch.tensor([rq.n_slots])).exp()[0, : rq.n_slots].tolist()


@pytest.mark.parametrize("sizes", [[(600, 800)], [(448, 448), (1000, 200)]])
def test_image_answers_match_full_forward(engines, sizes):
    """The cached prefix (images + state, far past the 8-token sliding window) forked across
    three questions answers as each full prompt does."""
    _, vision = engines
    images = [_b64(w, h, i) for i, (w, h) in enumerate(sizes)]
    rendered = [render_question(q) for q in QUESTIONS.values()]
    pil = [fit_image(decode_image(b), 448) for b in images]
    probs, ntok = vision._image_probs("Checkout page after the user tapped pay.", pil, rendered)
    assert ntok > 60 * len(images)
    for i, q in enumerate(QUESTIONS.values()):
        ref = _reference(vision, "Checkout page after the user tapped pay.", images, q)
        got = probs[i, : len(ref)].tolist()
        assert max(abs(x - y) for x, y in zip(got, ref, strict=True)) < 1e-5


def test_image_request_answers_through_the_engine(engines):
    _, vision = engines
    req = SystemOneRequest(state=STATE, questions=QUESTIONS, images=["data:image/png;base64," + _b64(320, 200, 2)])
    out = vision.evaluate(req)
    assert 0.0 <= out.answers["signed"].noul <= 1.0
    assert sum(out.answers["button"].probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    blind = vision.evaluate(SystemOneRequest(state=STATE, questions=QUESTIONS))
    assert out.answers["signed"].noul != pytest.approx(blind.answers["signed"].noul, abs=1e-6)  # the image counts


@torch.no_grad()
def test_frozen_readout_with_an_image_is_the_lms_own(engines, base_dir):
    """The KL reference over an image prompt: Gemma4ForConditionalGeneration's soft-capped
    logits for the option numbers, adapters off."""
    _, vision = engines
    model = vision.model
    q = render_question(QUESTIONS["button"])
    counts, mm = vision.prompter.process([Image.new("RGB", (120, 90), (200, 30, 30))])
    ids = torch.tensor([vision.tok(vision.prompter.state("", counts) + q.text)["input_ids"]])
    lp, eligible = model.frozen_slot_log_probs(ids, torch.ones_like(ids), torch.tensor([q.n_slots]), **mm)
    assert bool(eligible.all())
    lm = transformers.Gemma4ForConditionalGeneration.from_pretrained(base_dir).eval()
    logits = lm(input_ids=ids, **mm).logits[0, -1]
    want = logits[list(model.slot_token_ids().values())[: q.n_slots]].log_softmax(-1)
    torch.testing.assert_close(lp[0, : q.n_slots], want, atol=1e-5, rtol=1e-5)


# ---- checkpoints -------------------------------------------------------------------------


@torch.no_grad()
def test_save_load_round_trip(ckpt, tmp_path):
    model = VisionDeciderModel.load(ckpt, trainable=True)
    for n, p in model.torso.named_parameters():
        if "lora_" in n:
            p.add_(torch.randn_like(p) * 0.05)
    for p in model.head.parameters():
        p.add_(torch.randn_like(p) * 0.05)
    model.save_pretrained(str(tmp_path / "ck"))
    loaded = VisionDeciderModel.load(str(tmp_path / "ck"))
    a = VisionEngine(model.eval(), VisionEngineConfig(device="cpu"))
    b = VisionEngine(loaded, VisionEngineConfig(device="cpu"))
    req = SystemOneRequest(state=STATE, questions=QUESTIONS, images=[_b64(300, 200, 9)])
    assert a.evaluate(req).model_dump()["answers"] == b.evaluate(req).model_dump()["answers"]
    text = SystemOneRequest(state=STATE, questions=QUESTIONS)
    assert a.evaluate(text).model_dump()["answers"] == b.evaluate(text).model_dump()["answers"]


# ---- evaluation and serving -------------------------------------------------------------


def test_evaluation_systems_run_on_gemma(ckpt, base_dir):
    from vision.run import QwenUntrained, Strands

    item = {"kind": "noul", "question": "Is the form signed?", "options": [("no", ""), ("yes", "")],
            "gold": 1, "image": base64.b64decode(_b64(320, 320, 6))}
    for system in (Strands(ckpt), QwenUntrained(base_dir)):
        assert isinstance(system.prompter, GemmaImages)
        for blind in (False, True):
            p = system.probs(item, blind)
            assert len(p) == 2 and sum(p) == pytest.approx(1.0, abs=1e-5)
        assert system.probs(item, False) != pytest.approx(system.probs(item, True), abs=1e-6)


def test_text_check_runs_on_gemma(ckpt, tmp_path, monkeypatch):
    from vision import text_check

    row = {"kind": "noul", "state": "Order 7 shipped late.", "instructions": "Was it late?",
           "options": [["false", "on time"], ["true", "late"]], "label": 1, "task": "late"}
    for f in text_check.FILES:
        (tmp_path / f).write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(sys, "argv", ["text_check", "--checkpoint", ckpt, "--data-dir", str(tmp_path),
                                      "--out", str(tmp_path / "out.json"), "--device", "cpu"])
    text_check.main()
    rows = json.loads((tmp_path / "out.json").read_text())["rows"]
    text = SystemOneEngine(StrandsDeciderModel.load(ckpt), EngineConfig(device="cpu")).ask(
        row["state"], {"q": NoulQuestion(instructions=row["instructions"],
                                         criteria={"false": "on time", "true": "late"})})
    # the text engine reports noul to 4 decimals
    assert len(rows) == 3 and rows[0]["probs"][1] == pytest.approx(text.answers["q"].noul, abs=1e-4)


def test_serve_vision(ckpt):
    from fastapi.testclient import TestClient

    from strands_decider import server

    questions = {"signed": {"type": "noul", "instructions": "Is the form signed?"},
                 "button": {"type": "choice", "instructions": "Which control next?",
                            "criteria": {"submit": "blue", "cancel": "grey"}}}
    vision_app = TestClient(server.create_app(ckpt, device="cpu", vision=True))
    r = vision_app.post("/v1/systemone", json={"state": "Checkout page.", "images": [_b64(320, 320, 3)],
                                               "questions": questions})
    assert r.status_code == 200, r.text
    assert vision_app.get("/health").json()["vision"] is True
    text_app = TestClient(server.create_app(ckpt, device="cpu"))
    body = {"state": STATE, "questions": questions}
    got, want = vision_app.post("/v1/systemone", json=body).json(), text_app.post("/v1/systemone", json=body).json()
    assert got["answers"] == want["answers"]
