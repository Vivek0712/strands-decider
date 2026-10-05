"""A1: new decision task families for training, every label computed by a generator.

Arm A1 of the recipe screen behind strands-decider-4B-hobson-v22 adds task families, not
rows: held-out generalisation grows with the number of families far more than with rows
per family. Four families, in the shapes the board's sealed set is known to ask about,
each asked as all three question types:

  state_tracking  an event log over a small state machine: is X in state S, which state,
                  how many end in S                                  (state.py)
  tool_guardrail  a tool, a three-rule policy and a proposed call: does it violate, which
                  rule, how many rules                               (guardrail.py)
  lead_rubric     a CRM record against a four-criterion rubric: meets X / qualified, which
                  criterion fails, the 0-4 rating                    (rubric.py)
  json_policy     a request and a policy with a stated procedure: compliant, approve /
                  escalate / deny, how many conditions hold          (policy.py)

States are drawn in four formats (indented JSON, compact JSON, key: value lines, prose),
questions in two or three phrasings (`instruction_variants`, sampled per epoch by the
collator), and gold labels are balanced within each question template. None of these
families is in unseen-v2 (evaluation/unseen_v2/), and tests/test_families.py holds that.

    python -m strands_decider.data.families --out data/families_a1.jsonl \\
        --eval-out data/families_a1_eval.jsonl

The default is 2,000 rows per family and type (24,000) for training and 150 (1,800) for
the in-family evaluation file, from seeds 0 and 1, the evaluation rows never repeating a
training row: the same seeds give the same files.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from types import ModuleType

from ..format import Example, write_jsonl
from . import guardrail, policy, rubric, state
from .common import draw

FAMILIES: dict[str, ModuleType] = {m.FAMILY: m for m in (state, guardrail, rubric, policy)}
KINDS = ("noul", "choice", "score")


def build(n_per_kind: int, seed: int = 0, families: list[str] | None = None) -> list[Example]:
    """`n_per_kind` rows of every family and type. Each (family, type) draws from its own
    generator seeded by (seed, family, type), so adding a family changes no other rows."""
    out: list[Example] = []
    for name in families or list(FAMILIES):
        mod = FAMILIES[name]
        for kind in KINDS:
            rng = random.Random(f"a1:{seed}:{name}:{kind}")
            out += draw(mod.generate, mod.TEMPLATES[kind], kind, n_per_kind, rng)
    random.Random(f"a1:{seed}:order").shuffle(out)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate the A1 training families.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-out", help="also write an in-family evaluation file (another seed)")
    ap.add_argument("--n", type=int, default=2000, help="rows per family and question type")
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seed", type=int, default=1)
    a = ap.parse_args(argv)
    train = build(a.n, a.seed)
    write_jsonl(a.out, train)
    print(f"{a.out}: {len(train):,} rows", dict(Counter(f"{e.task}/{e.kind}" for e in train)))
    if a.eval_out:
        seen = {key(e) for e in train}
        held = [e for e in build(a.n_eval, a.eval_seed) if key(e) not in seen]
        write_jsonl(a.eval_out, held)
        print(f"{a.eval_out}: {len(held):,} rows", dict(Counter(f"{e.task}/{e.kind}" for e in held)))


def key(ex: Example) -> str:
    """What makes two rows the same question: state, question and options."""
    return repr((ex.state, ex.instructions, ex.options))


if __name__ == "__main__":
    main()
