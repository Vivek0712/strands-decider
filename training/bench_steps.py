"""Measure training speed: a config run for a few steps, its optimizer steps per second.

    python training/bench_steps.py --config configs/next/stage-d/cell-000.yaml --steps 100
    torchrun --standalone --nproc_per_node=4 training/bench_steps.py --config ... --steps 100

It trains the config as `strands-decider train` would, for `--steps` optimizer steps, into a
scratch output directory, with validation off, and reports the rate the trainer logs at
the last step (steps over the time since step 1 began, so the frozen-KL precompute and the
model load are excluded and reported apart), the peak GPU memory, and whether the fused
Gated DeltaNet kernels were in use (training/install_fast_kernels.sh). Run it with and
without the kernels on the same host and config before budgeting a stage.

`--set key=value` overrides any TrainConfig field, e.g. `--set base_model=Qwen/Qwen3.5-4B-Base`.
Writes the result as JSON to stdout and, with `--json`, to a file.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import time
from typing import Any


class Tee(io.TextIOBase):
    """Write to the real stdout and keep a copy."""

    def __init__(self, out: Any) -> None:
        self.out, self.buf = out, io.StringIO()

    def write(self, s: str) -> int:
        self.out.write(s)
        return self.buf.write(s)

    def flush(self) -> None:
        self.out.flush()


def kernels() -> dict[str, Any]:
    from transformers.utils.import_utils import (
        is_causal_conv1d_available,
        is_flash_linear_attention_available,
    )

    return {"causal_conv1d": bool(is_causal_conv1d_available()),
            "flash_linear_attention": bool(is_flash_linear_attention_available())}


def parse(log: str) -> dict[str, Any]:
    """The last logged rate and memory, and the frozen-KL precompute time, from train's log."""
    rates = re.findall(r"step (\d+)/\d+ .*? ([\d.]+) step/s peak_mem ([\d.]+)GiB", log)
    pre = re.findall(r"frozen-KL reference for [\d,]+ micro-batches in (\d+) s", log)
    out: dict[str, Any] = {}
    if rates:
        step, rate, mem = rates[-1]
        out.update(last_step=int(step), step_per_s=float(rate), peak_mem_gib=float(mem))
    if pre:
        out["frozen_kl_precompute_s"] = int(pre[-1])
    return out


def main(argv: list[str] | None = None) -> None:
    import torch

    from strands_decider import distributed
    from strands_decider.train import TrainConfig, train

    ap = argparse.ArgumentParser(description="Optimizer steps per second for a training config.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--set", action="append", default=[], help="key=value: override a TrainConfig field")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    cfg = TrainConfig.from_yaml(a.config).with_overrides(a.set)
    out_dir = tempfile.mkdtemp(prefix="bench-steps-")
    cfg = cfg.with_overrides([f"max_steps={a.steps}", "eval_every=0", "save_every=0",
                              f"log_every={min(20, a.steps)}", f"output_dir={out_dir}"])
    tee = Tee(sys.stdout)
    t0 = time.time()
    with contextlib.redirect_stdout(tee):
        train(cfg)
    rank, _, world = distributed.env()
    if rank != 0:
        return
    result = {"config": a.config, "overrides": a.set, "steps": a.steps, "world_size": world,
              "wall_s": round(time.time() - t0, 1), **parse(tee.buf.getvalue()), "kernels": kernels(),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "output_dir": out_dir}
    print(json.dumps(result, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    if os.environ.get("BENCH_KEEP") != "1":
        import shutil

        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
