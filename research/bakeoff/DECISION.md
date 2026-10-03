# Bake-off decision rule (fixed 2026-10-03 20:10 UTC, before any bake-off score existed)

Arms: Qwen3.5-2B (two seeds), MiniCPM5-2B, Gemma 4 E2B; identical data, budget (2,400 steps), recipe and seed 0,
each trained from its raw base. Measured on JevBench v1 public (231) with `evaluation/jevbench/v15_proxy.py`.

1. Primary: v1.5 proxy Intelligence.
2. A non-Qwen base wins only if its proxy Intelligence is above the HIGHER of the two Qwen seeds
   (the Qwen seed spread is the noise floor). Otherwise Qwen3.5-2B is kept and v21 continues v19.
3. If both non-Qwen bases clear the bar, the higher proxy Intelligence wins; within 1.0 point of each other,
   the lower Brier wins, then the faster p50 latency.
4. Tasks right, yes/no band share, Brier and ECE are reported for every arm whatever the outcome.
