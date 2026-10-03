"""Tiny random-weight bases of the two other torso families, the way tests/tiny_qwen35.py
builds a Qwen3.5: written to a directory, nothing downloaded.

* `save_llama`: a Llama decoder with an untied output head, as MiniCPM5 is, and a
  byte-level BPE tokeniser over the 256 byte symbols (no merges) that prepends `<s>`
  through its post-processor, as MiniCPM5's does.
* `save_gemma4`: a Gemma 4 multimodal checkpoint (text decoder with per-layer inputs,
  sliding and full attention and KV-shared layers, plus a vision tower), saved through
  Gemma4ForConditionalGeneration so its decoder weights sit under `model.language_model.`
  as in google/gemma-4-E2B, and a real `GemmaTokenizer` (byte-fallback BPE, no merges,
  spaces as "▁", `<bos>` prepended). With `images=True` the tokeniser also has Gemma 4's
  image tokens `<|image>`, `<|image|>`, `<image|>` as special tokens whose ids the config
  names, a 1-layer audio tower is saved too (as google/gemma-4-E2B ships one, for the image
  path to leave unread), and a `Gemma4ImageProcessorPil` at 70 soft tokens per image
  (google/gemma-4-E2B ships 280) is saved beside it, so the multimodal model answers over
  images (tests/test_vision_gemma.py). The default keeps the text-only bases of test_bases.py
  exactly as they were.

Each tokeniser reloads as itself through AutoTokenizer and tokenises every character of a
state or question, as the real ones do.
"""

from __future__ import annotations

import torch

# begin-of-image, the soft-token placeholder, end-of-image: as google/gemma-4-E2B's tokeniser
GEMMA_IMAGE_TOKENS = ("<|image>", "<|image|>", "<image|>")
ATTENTION_MLP = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def llama_tokenizer():
    import transformers
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {"<s>": 0, "</s>": 1, "<unk>": 2, **{c: i + 3 for i, c in enumerate(alphabet)}}
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.post_processor = processors.TemplateProcessing(
        single="<s> $A", pair="<s> $A <s> $B", special_tokens=[("<s>", 0)])
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="</s>", unk_token="<unk>")


def save_llama(d: str) -> str:
    """2 layers, grouped-query attention, untied output head, saved to `d`."""
    import transformers

    tok = llama_tokenizer()
    cfg = transformers.LlamaConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=False,
        bos_token_id=0, eos_token_id=1, pad_token_id=1)
    torch.manual_seed(0)
    transformers.LlamaForCausalLM(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return d


def gemma4_tokenizer(images: bool = False):
    import transformers

    specials = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>"]
    pieces = specials + [f"<0x{b:02X}>" for b in range(256)] + ["▁"]
    pieces += [chr(c) for c in range(33, 127)] + ["\n"]
    tok = transformers.GemmaTokenizer(vocab={p: i for i, p in enumerate(pieces)}, merges=[],
                                      add_bos_token=True)
    if images:
        tok.add_special_tokens({"additional_special_tokens": list(GEMMA_IMAGE_TOKENS)})
    tok.padding_side = "left"  # as google/gemma-4-E2B ships it; the package pads right itself
    return tok


AUDIO = {"hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
         "subsampling_conv_channels": [8, 4], "output_proj_dims": 32}


def save_gemma4(d: str, images: bool = False) -> str:
    """4 decoder layers (sliding, full, then two sharing their KV), a 1-layer ViT."""
    import transformers

    tok = gemma4_tokenizer(images)
    ids = dict(zip(("boi_token_id", "image_token_id", "eoi_token_id"),
                   tok.convert_tokens_to_ids(list(GEMMA_IMAGE_TOKENS)), strict=True)) if images else {}
    cfg = transformers.Gemma4Config(
        text_config={
            "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128, "head_dim": 16,
            "global_head_dim": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
            "vocab_size": len(tok), "vocab_size_per_layer_input": len(tok),
            "hidden_size_per_layer_input": 8, "num_kv_shared_layers": 2, "sliding_window": 8,
            "layer_types": ["sliding_attention", "full_attention"] * 2,
            "tie_word_embeddings": True, "pad_token_id": 0, "eos_token_id": 1, "bos_token_id": 2,
            # E2B caps at 30; a cap this low makes the soft-capping visible in random logits
            "final_logit_softcapping": 1.0,
        },
        vision_config={"hidden_size": 32, "num_hidden_layers": 1, "intermediate_size": 64,
                       "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 16,
                       "global_head_dim": 16},
        audio_config=AUDIO if images else None, **ids,
    )
    torch.manual_seed(0)
    transformers.Gemma4ForConditionalGeneration(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    if images:
        transformers.Gemma4ImageProcessorPil(max_soft_tokens=70).save_pretrained(d)
    return d
