# Images

Strands Decider (v19) answers questions about images when it is loaded with `--vision`
(so does a checkpoint on Gemma 4 E2B; see [Gemma 4 E2B checkpoints](#gemma-4-e2b-checkpoints)).
No weights change: the published checkpoint is used as is, and the images are read by the
vision tower that already ships inside Qwen3.5-2B-Base. Text-only requests to a vision
server get the same answers as from a text server.

## Using it

```bash
pip install "strands-decider[vision]"     # Pillow; needs transformers >= 5.18
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --vision --port 8000
```

A request adds `images`: a list of base64 images, PNG, JPEG, WebP or GIF (a
`data:image/...;base64,` prefix is accepted). The state may be empty.

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Checkout page after the user tapped Pay.",
  "images": ["'"$(base64 < screen.png | tr -d '\n')"'"],
  "questions": {
    "paid":   {"type": "noul",   "instructions": "Did the payment succeed?"},
    "next":   {"type": "choice", "instructions": "Which control should the agent use next?",
               "criteria": {"retry": "a retry button is shown", "back": "return to the cart",
                            "none": "nothing to do"}},
    "severe": {"type": "score",  "instructions": "How severe is the error shown?",
               "criteria": ["no error", "warning", "blocking error"]}
  }
}'
```

From the command line, `ask --image FILE` (repeatable) loads the vision tower for that call.
A server started without `--vision` answers a request with `images` with HTTP 422 rather
than ignore them. `/health` reports `"vision": true|false`. `--strict-window` and
`--max-batch` apply to image requests as they do to text ones.

Limits, fixed per server:

- Each image is scaled down (never up) so its longer side is at most 448 px: 196 tokens
  for a square image (patch 16, merge 2). At most 4 images per request.
- An image over 4096 x 4096 pixels is refused from its header, before it is decoded.
  Other formats are refused before they are parsed. EXIF rotation is applied, and
  transparent areas are read as white.
- Images count against the same 4,096-token window as the state. Without
  `--strict-window`, the questions keep their reserve, the images are kept whole, and the
  state text loses its end, as text requests do. With it, an over-long prompt gets 422.
- The vision tower adds about 331M parameters (0.66 GB in bf16).

## How it works

`StrandsDeciderModel._load_torso` loads only the text decoder of Qwen3.5. `--vision` loads
`Qwen3_5ForConditionalGeneration(...).model` instead: the ViT, the patch merger and the
same 24-layer decoder. The checkpoint's LoRA adapter was trained on the bare decoder
(`layers.N...`); inside the multimodal model the same decoder sits under
`language_model.layers.N...`, so the adapter's keys are remapped onto it
(`VisionDeciderModel.load`). The vision tower is frozen; nothing in it is adapted.

Images go inside `<state>` as Qwen vision placeholders, before the state text:

```
<state>
<|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>
Checkout page after the user tapped Pay.
</state>
<question type="noul"> ... <answer>
```

Each `<|image_pad|>` stands for one merged patch. Everything after the state is text, so
the option positions, the pointer readout, the temperatures and every confidence formula
are the text model's. The shared-prefix cache works as for text: the image and state are
encoded once and forked across the questions.

Two details matter for exact answers:

- **Positions.** Qwen3.5 uses M-RoPE: an image advances the rotary positions by its grid
  size, not its token count, so text after an image sits at `token index + rope_delta`.
  The image path passes these positions explicitly for the cached prefix and the question
  suffixes. Left to transformers, a suffix-only forward builds positions from the full
  mask, and reads a `rope_deltas` that an earlier plain image forward left on the module.
- **Pixels.** The image processor is pinned to the PIL backend
  (`Qwen2VLImageProcessorPil`). `Qwen2VLImageProcessor` switches to torchvision when it
  is installed, and that changes pixel values enough to flip a few answers.

Checks (`tests/test_vision.py`, offline, on a tiny random-weight Qwen3.5 with a real
`Qwen3_5Tokenizer`): text requests to a vision engine get the text engine's answers; over
one and two images the shared-prefix path matches a plain full forward to under 1e-5; no
position state leaks into the next request; the window and decoding rules above. Each
test fails when the fix it pins is removed.

## Gemma 4 E2B checkpoints

A Strands Decider trained on `google/gemma-4-E2B` (the bake-off's
[configs/experiments/bakeoff/gemma4-e2b.yaml](../configs/experiments/bakeoff/gemma4-e2b.yaml), or
one image-trained from it) loads with `--vision` the same way; the family is read from the
checkpoint's `base_model`:

```bash
strands-decider serve checkpoints/bakeoff-gemma4-e2b --vision --port 8000
```

- **Torso.** `Gemma4ForConditionalGeneration(...).model` at the checkpoint's pinned
  `base_revision`, built without its audio tower (`audio_config` unset, so the
  checkpoint's audio weights are never read): the vision encoder (16 layers, d 768), its
  projection into the decoder (`embed_vision`), both frozen, and the same 35-layer decoder.
  The text checkpoint's adapter is remapped from `layers.N...` to `language_model.layers.N...`
  exactly as for Qwen, onto the same modules (the last 20 layers share keys and values and
  have no `k_proj` / `v_proj`). Text-only requests get the text server's answers.
- **Images in the prompt.** `<|image>`, then one `<|image|>` per soft token, then `<image|>`,
  inside `<state>` before the text, as Qwen's placeholders are. The count is the processor's
  `num_soft_tokens_per_image`: the image is resized (aspect kept, up or down) to at most
  `max_soft_tokens` x 9 patches of 16 px, and every 3 x 3 patches pool to one token. E2B ships
  `max_soft_tokens: 280`, so an image takes 256-280 tokens whatever its size (about 645,000
  pixels); the server's 448 px cap or pixel budget is applied first, and only changes how much
  detail survives into that resize, not the token count.
- **Positions.** Gemma 4 uses 1-D RoPE (one position per token, image tokens included), so
  the default positions are already right for a question suffix after the cached image
  prefix, and nothing is left on the module between requests. E2B attends causally to image
  tokens (`use_bidirectional_attention` is unset); the image path still passes
  `mm_token_type_ids`, which a Gemma 4 with bidirectional image attention would need.
- **Pixels.** Pinned to `Gemma4ImageProcessorPil`, read from the base's `processor_config.json`.
- **Readout.** The frozen KL reference reads the LM's option-number logits soft-capped by
  `final_logit_softcapping` (30), as the text torso does; on the multimodal model that value
  sits on the text config.

Checks (`tests/test_vision_gemma.py`, offline, on a tiny random-weight
`Gemma4ForConditionalGeneration` with sliding, full and KV-shared layers, a vision and an
audio tower and a real `GemmaTokenizer`): the torso's weights are the checkpoint's and no
audio weight is loaded; LoRA reaches exactly the text checkpoint's modules and the adapter is
the text checkpoint's; text requests match the text engine; placeholder counts are what the
vision tower emits; over one and two images (prefixes far past the sliding window, forked
across three questions) the shared-prefix path matches a full forward to under 1e-5; the KL
reference with an image is the LM's own reading; three steps of image training lower the
loss; a checkpoint round-trips; `run.py`, `text_check.py` and `serve --vision` run on it. No
real Gemma 4 weights were run: nothing here measures how well it answers over images.

## How well it does

Measured with the published v19, no image training, against the untrained base and an
image-trained 2B decider, at 448 px (Mapika at its 768 px). Lower is better for ECE and
Brier.

| System | NaturalBench acc | NaturalBench G-Acc | NaturalBench ECE | POPE-adv acc | POPE-adv Brier | POPE-adv ECE | Image JevBench preview, exact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-2B-Base, untrained option-number readout | 0.712 | 0.177 | 0.048 | 0.865 | 0.227 | 0.116 | 28/60 |
| Strands Decider (v19), `--vision` | 0.784 | 0.327 | 0.014 | 0.877 | 0.202 | 0.072 | 37/60 |
| Mapika/decider-2b-vision (image-trained) | 0.794 | 0.360 | 0.080 | 0.860 | 0.204 | 0.061 | 38/60 |

v19 against Mapika, paired bootstrap 95% CIs (v19 minus Mapika): NaturalBench ECE
-0.066 (-0.083 to -0.032), the one clear difference; NaturalBench accuracy -0.010 (-0.031
to +0.011), G-Acc -0.033 (-0.077 to +0.010), POPE accuracy +0.017 (-0.005 to +0.038) and
POPE ECE +0.010 (-0.025 to +0.053), all within noise. So v19 matches the image-trained
model on accuracy, and on NaturalBench is much better calibrated.

With the image removed every system falls to chance (NaturalBench 0.500, POPE about
0.50), so the answers come from the image. What text training does not teach shows here:
without the image, v19 still answers at a mean confidence of 0.652 on NaturalBench and
0.725 on POPE (ECE 0.152 and 0.225; Mapika 0.153 and 0.175; the untrained base 0.092 and
0.059).

- NaturalBench: the first 300 groups of the first shard (1,200 questions); each group is
  two images and two questions with opposite answers, and G-Acc counts a group only when
  all four are right.
- POPE adversarial: the first 600 items by question id.
- Image JevBench: the official items are sealed. These are the 60 published preview items
  that can be rebuilt exactly from their source rows (CLEVR-HOPE, Geometry3K, ArxivQA).
- One run per system, on one H100 each (fp32 for v19 and the base, bf16 for Mapika).

The script is `evaluation/vision/run.py`; [evaluation/vision/README.md](../evaluation/vision/README.md)
has the exact command and a link to the recorded runs: per-item probabilities for v19 and
the base, and Mapika's summary.

Two opt-in settings change how images are read, and fine-tuning on images builds on them
([v20-v21.md](v20-v21.md)): a pixel budget per image instead of the long-side cap
(`serve --vision --image-long-side 0 --image-max-pixels 400000`, or
`load_vision_engine(image_long_side=0, image_max_pixels=400_000)`; at 400,000 pixels v19 gets
40 of the 60 preview items right instead of 37, within noise), and per-kind temperatures
for questions over images (`image_temperature_by_kind`, fitted by
`evaluation/vision/temps.py`), which `serve --vision` applies when a checkpoint has them.

## A torso without a vision tower: grafted eyes (MiniCPM5)

MiniCPM5-2B is a text-only Llama decoder. `src/strands_decider/graft.py` gives a MiniCPM5
checkpoint eyes LLaVA-style: [`google/siglip2-so400m-patch16-384`](https://huggingface.co/google/siglip2-so400m-patch16-384)
(Apache-2.0, pinned at `dd658faac399427308559e2c3ac1e99cbe43845d`, frozen) reads each image
at 384 x 384 px as 24 x 24 patches; a 2 x 2 pixel-unshuffle and a two-layer GELU MLP (the
projector, trained, saved in the checkpoint as `projector.safetensors` and
`projector_config.json`) turn them into 144 vectors of the decoder's width. They replace the
input embeddings of 144 `<unused_token_0>` slots inside `<state>`, between plain-text markers:

```
<state>
<image><unused_token_0> x 144</image>
Checkout page after the user tapped Pay.
</state>
```

Positions stay 1-D, so the shared-prefix cache is the text one: images and state are
forwarded once (as `inputs_embeds`), the question suffixes as token ids against that cache.
Text requests take the text path untouched. `serve --vision`, `evaluation/vision/run.py`
and `text_check.py` recognise such a checkpoint by its `projector_config.json`.

Training is two stages (`training/recipe_minicpm_vision.sh`): stage 1 trains the projector
alone to caption 80,000 COCO train2014 images through the frozen base LM
(`strands_decider.graft_align`, `configs/align/strands-decider-2.5B-minicpm-v21-vl.yaml`);
stage 2 is the v19-images recipe on the v21 MiniCPM5 text checkpoint with that projector,
LoRA, head and projector trained (`configs/vision/strands-decider-2.5B-minicpm-v21-vl*.yaml`).
Checks: `tests/test_graft.py`. Exploratory; the commands are in [v20-v21.md](v20-v21.md) and
the measurements in the results release (https://github.com/Vivek0712/strands-decider/blob/b63bc26de24a935a4925ed96616954f82daf1f80/research/v21/RESULTS.md).
