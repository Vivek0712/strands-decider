"""Tool-call guardrails: a tool, a three-rule policy, and a proposed call.

Each rule is a predicate on the call's arguments (a numeric cap unless an approval is
attached, a forbidden value, an allow-list, a required field), with thresholds drawn per
item, so the gold is computed exactly. Asked as: does the call violate the policy
(yes/no), which rule does it break, or none (choice, only calls breaking at most one),
how many rules does it break (score, 0-3).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from .common import YES_NO, Row, join, levels, named, render, row

FAMILY = "tool_guardrail"
Rule = tuple[str, Callable[[dict[str, Any]], bool]]  # (text, broken?)
BROKEN = ["breaks no rule", "breaks exactly one rule", "breaks two rules", "breaks all three rules"]
NONE = "None of the rules: the call is allowed"


def _transfer(rng: random.Random) -> tuple[str, dict[str, Any], list[Rule]]:
    cap, approver_cap = rng.choice([500, 1000, 2500, 5000]), rng.choice([10_000, 20_000])
    banned = rng.sample(["KP", "IR", "SY", "CU", "RU", "BY"], 2)
    args = {"amount": rng.choice([rng.randint(50, cap), rng.randint(cap + 1, 3 * cap)]),
            "currency": rng.choice(["USD", "EUR", "GBP"]),
            "destination_country": rng.choice([*banned, "DE", "FR", "US", "JP", "IN", "BR"]),
            "approval_id": rng.choice([None, f"APR-{rng.randint(1000, 9999)}"]),
            "memo": rng.choice(["", "invoice 2231", "vendor payment", "refund"])}
    rules: list[Rule] = [
        (f"Transfers above {cap} need an approval_id.",
         lambda a: a["amount"] > cap and not a["approval_id"]),
        (f"Never transfer to these countries: {', '.join(banned)}.",
         lambda a: a["destination_country"] in banned),
        ("Every transfer needs a non-empty memo.", lambda a: not a["memo"]),
    ]
    if rng.random() < 0.5:
        rules[2] = (f"No single transfer may exceed {approver_cap}, approved or not.",
                    lambda a: a["amount"] > approver_cap)
        args["amount"] = rng.choice([args["amount"], rng.randint(approver_cap // 2, approver_cap * 2)])
    return "transfer_funds", args, rules


def _delete(rng: random.Random) -> tuple[str, dict[str, Any], list[Rule]]:
    limit = rng.choice([100, 500, 1000])
    protected = rng.sample(["users", "payments", "audit_log", "invoices"], 2)
    args = {"table": rng.choice([*protected, "sessions", "cache", "drafts"]),
            "environment": rng.choice(["prod", "staging", "dev"]),
            "row_count": rng.choice([rng.randint(1, limit), rng.randint(limit + 1, limit * 5)]),
            "backup_taken": rng.random() < 0.5}
    rules: list[Rule] = [
        (f"Never delete from these tables: {', '.join(protected)}.", lambda a: a["table"] in protected),
        ("Deletes in prod require backup_taken to be true.",
         lambda a: a["environment"] == "prod" and not a["backup_taken"]),
        (f"A single delete may remove at most {limit} rows.", lambda a: a["row_count"] > limit),
    ]
    return "delete_records", args, rules


def _email(rng: random.Random) -> tuple[str, dict[str, Any], list[Rule]]:
    domain = rng.choice(["acme.example", "northwind.example", "contoso.example"])
    max_to = rng.choice([3, 5, 10])
    people = ["ana", "bo", "cy", "dee", "eli", "fay", "gus", "hal", "ivy", "jo", "kai", "lu", "mo", "ned"]
    args = {"to": [f"{n}@{rng.choice([domain, domain, 'gmail.example', 'partner.example'])}"
                   for n in rng.sample(people, rng.randint(1, max_to + 3))],
            "subject": rng.choice(["Q3 numbers", "", "Lunch", "Contract draft"]),
            "attachment_labels": rng.sample(["public", "internal", "confidential"], rng.randint(0, 2))}
    rules: list[Rule] = [
        (f"Recipients must all be on {domain} when any attachment is labelled confidential.",
         lambda a: "confidential" in a["attachment_labels"] and any(not t.endswith("@" + domain) for t in a["to"])),
        (f"At most {max_to} recipients per email.", lambda a: len(a["to"]) > max_to),
        ("The subject must not be empty.", lambda a: not a["subject"]),
    ]
    return "send_email", args, rules


def _deploy(rng: random.Random) -> tuple[str, dict[str, Any], list[Rule]]:
    need = rng.choice([1, 2])
    frozen = rng.choice(["billing", "auth", "search"])
    args = {"service": rng.choice([frozen, "web", "worker", "search", "billing"]),
            "environment": rng.choice(["prod", "staging"]),
            "approvals": rng.randint(0, 3),
            "tests_passed": rng.random() < 0.6}
    rules: list[Rule] = [
        (f"Prod deploys need at least {need} approval{'s' if need > 1 else ''}.",
         lambda a: a["environment"] == "prod" and a["approvals"] < need),
        ("tests_passed must be true for any deploy.", lambda a: not a["tests_passed"]),
        (f"The {frozen} service is in a change freeze: no deploys to prod.",
         lambda a: a["service"] == frozen and a["environment"] == "prod"),
    ]
    return "deploy_service", args, rules


TOOLS = (_transfer, _delete, _email, _deploy)


def scenario(rng: random.Random) -> tuple[str, dict[str, Any], list[Rule], Any]:
    tool, args, rules = rng.choice(TOOLS)(rng)
    rng.shuffle(rules)
    policy = {f"R{i + 1}": text for i, (text, _) in enumerate(rules)}
    state = join(render(rng, "Policy for " + tool, policy), render(rng, f"Proposed call to {tool}", args))
    return tool, args, rules, state


def generate(rng: random.Random, kind: str, template: str) -> Row | None:
    tool, args, rules, state = scenario(rng)
    broken = [i for i, (_, f) in enumerate(rules) if f(args)]
    if kind == "noul":
        return row("noul", state, ["Does this call violate the policy?",
                                   f"Would executing this {tool} call break any policy rule?",
                                   "Should the guardrail block this call?"], YES_NO, int(bool(broken)), FAMILY)
    if kind == "choice":
        if len(broken) > 1:
            return None
        names = [f"R{i + 1}" for i in range(len(rules))] + [NONE]
        return row("choice", state, ["Which policy rule does this call break?",
                                     "Name the rule this call violates, if any."],
                   named(names), broken[0] if broken else len(rules), FAMILY)
    return row("score", state, ["How many of the policy's rules does this call break?",
                                "Count the policy rules this call violates."], levels(BROKEN), len(broken), FAMILY)


TEMPLATES = {"noul": {"call": 1}, "choice": {"call": 1}, "score": {"call": 1}}
