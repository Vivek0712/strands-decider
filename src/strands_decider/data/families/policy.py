"""Policy compliance over a JSON request: a written policy with hard limits and an
escalation clause, and a request with its account state.

The decision procedure is stated in the policy itself: deny if any hard condition fails;
otherwise escalate if the escalation condition holds; otherwise approve. Thresholds are
drawn per item, so the gold is computed exactly. Asked as: is the request compliant (every
hard condition holds) (yes/no), what should happen to it (choice: approve, escalate, deny),
how many of the four hard conditions does it satisfy (score, 0-4).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from .common import YES_NO, Row, join, levels, named, render, row

FAMILY = "json_policy"
Check = tuple[str, Callable[[dict[str, Any]], bool]]  # (text, holds?)
DECISIONS = ["approve", "escalate to a manager", "deny"]
SATISFIED = ["satisfies none of the four", "satisfies one", "satisfies two", "satisfies three", "satisfies all four"]


def _refund(rng: random.Random) -> tuple[str, dict[str, Any], list[Check], Check]:
    days, cap, esc = rng.choice([14, 30, 60]), rng.choice([100, 250, 500]), rng.choice([50, 150])
    req = {"order_age_days": rng.randint(1, days * 2), "amount": rng.randint(5, cap * 2),
           "item_condition": rng.choice(["unopened", "opened", "damaged"]),
           "customer_refunds_last_90_days": rng.randint(0, 4), "has_receipt": rng.random() < 0.7}
    hard: list[Check] = [
        (f"The order is at most {days} days old.", lambda r: r["order_age_days"] <= days),
        (f"The refund amount is at most {cap}.", lambda r: r["amount"] <= cap),
        ("The item is unopened or damaged.", lambda r: r["item_condition"] in ("unopened", "damaged")),
        ("The customer has a receipt.", lambda r: r["has_receipt"]),
    ]
    escalate: Check = (f"Escalate when the amount exceeds {esc} or the customer had 2 or more refunds in "
                       "the last 90 days.", lambda r: r["amount"] > esc or r["customer_refunds_last_90_days"] >= 2)
    return "refund request", req, hard, escalate


def _access(rng: random.Random) -> tuple[str, dict[str, Any], list[Check], Check]:
    levels_ = ["read", "write", "admin"]
    tenure = rng.choice([30, 90])
    req = {"requested_level": rng.choice(levels_), "team": rng.choice(["data", "web", "finance", "support"]),
           "resource_team": rng.choice(["data", "web", "finance"]), "tenure_days": rng.randint(5, 400),
           "training_completed": rng.random() < 0.7, "mfa_enabled": rng.random() < 0.75}
    hard: list[Check] = [
        ("Multi-factor authentication is enabled.", lambda r: r["mfa_enabled"]),
        ("Security training is completed.", lambda r: r["training_completed"]),
        (f"Write or admin access needs at least {tenure} days of tenure.",
         lambda r: r["requested_level"] == "read" or r["tenure_days"] >= tenure),
        ("Admin access is only for the resource's own team.",
         lambda r: r["requested_level"] != "admin" or r["team"] == r["resource_team"]),
    ]
    escalate: Check = ("Escalate any access to a team other than the requester's own.",
                       lambda r: r["team"] != r["resource_team"])
    return "access request", req, hard, escalate


def _expense(rng: random.Random) -> tuple[str, dict[str, Any], list[Check], Check]:
    meal, hours, receipt_over = rng.choice([40, 60, 80]), rng.choice([4, 6]), rng.choice([25, 75])
    esc = rng.choice([500, 1000])
    req = {"category": rng.choice(["meal", "flight", "hotel"]), "amount": rng.randint(10, 1500),
           "flight_class": rng.choice(["economy", "premium", "business"]), "flight_hours": rng.randint(1, 12),
           "receipt_attached": rng.random() < 0.7, "submitted_within_days": rng.randint(1, 60)}
    hard: list[Check] = [
        (f"Meals may cost at most {meal}.", lambda r: r["category"] != "meal" or r["amount"] <= meal),
        (f"Business class only on flights longer than {hours} hours.",
         lambda r: r["category"] != "flight" or r["flight_class"] != "business" or r["flight_hours"] > hours),
        (f"A receipt is required above {receipt_over}.",
         lambda r: r["amount"] <= receipt_over or r["receipt_attached"]),
        ("Claims must be submitted within 30 days.", lambda r: r["submitted_within_days"] <= 30),
    ]
    escalate: Check = (f"Escalate any claim above {esc}.", lambda r: r["amount"] > esc)
    return "expense claim", req, hard, escalate


DOMAINS = (_refund, _access, _expense)


def scenario(rng: random.Random) -> tuple[str, list[bool], bool, Any]:
    name, req, hard, escalate = rng.choice(DOMAINS)(rng)
    rng.shuffle(hard)
    holds = [f(req) for _, f in hard]
    policy = {f"condition_{i + 1}": text for i, (text, _) in enumerate(hard)}
    policy["escalation"] = escalate[0]
    policy["procedure"] = ("Deny if any condition fails; otherwise escalate if the escalation clause "
                           "applies; otherwise approve.")
    article = "an" if name[0] in "aeiou" else "a"
    state = join(render(rng, f"Policy for {article} {name}", policy), render(rng, name.capitalize(), req))
    return name, holds, escalate[1](req), state


def generate(rng: random.Random, kind: str, template: str) -> Row | None:
    name, holds, esc, state = scenario(rng)
    if kind == "noul":
        return row("noul", state, [f"Does this {name} satisfy every condition of the policy?",
                                   f"Is the {name} compliant with the policy's conditions?"],
                   YES_NO, int(all(holds)), FAMILY)
    if kind == "choice":
        decision = 2 if not all(holds) else 1 if esc else 0
        return row("choice", state, [f"Under the policy's procedure, what should happen to this {name}?",
                                     "Which outcome does the policy require?"], named(DECISIONS), decision, FAMILY)
    return row("score", state, [f"How many of the policy's four conditions does this {name} satisfy?",
                                "Count the conditions this request meets."], levels(SATISFIED), sum(holds), FAMILY)


TEMPLATES = {"noul": {"compliant": 1}, "choice": {"decision": 1}, "score": {"conditions": 1}}
