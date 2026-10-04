"""Lead qualification: a CRM record against a four-criterion rubric.

The criteria are drawn per item from a pool (company size, budget, buyer role, purchase
timeline, industry, territory, demo request, meetings held), each with a threshold drawn
per item, so the gold is computed exactly. Asked as: rate the lead 0-4, one point per
criterion met (score); does the lead meet criterion X, or is it qualified (at least three)
(yes/no); which single criterion does it fail (choice, only leads failing exactly one).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from .common import YES_NO, Row, join, levels, named, render, row

FAMILY = "lead_rubric"
Criterion = tuple[str, str, Callable[[dict[str, Any]], bool]]  # (name, text, met?)
RATING = ["0: meets none of the criteria", "1: meets one criterion", "2: meets two criteria",
          "3: meets three criteria", "4: meets all four criteria"]
INDUSTRIES = ["healthcare", "retail", "logistics", "fintech", "manufacturing", "education", "media"]
REGIONS = ["EMEA", "NA-East", "NA-West", "APAC", "LATAM"]
ROLES = ["Intern", "Analyst", "Manager", "Director", "VP", "CTO", "CEO"]


def _pool(rng: random.Random) -> list[Criterion]:
    size, budget = rng.choice([50, 200, 500, 1000]), rng.choice([10_000, 25_000, 50_000, 100_000])
    months, meetings = rng.choice([3, 6, 9]), rng.choice([1, 2, 3])
    senior = ROLES[rng.choice([3, 4])]
    targets = rng.sample(INDUSTRIES, 3)
    territory = rng.sample(REGIONS, 2)
    return [
        ("size", f"Company has at least {size} employees.", lambda r: r["employees"] >= size),
        ("budget", f"Annual budget for the product is at least {budget}.", lambda r: r["budget"] >= budget),
        ("authority", f"The contact is {senior} or more senior ({' < '.join(ROLES)}).",
         lambda r: ROLES.index(r["contact_role"]) >= ROLES.index(senior)),
        ("timeline", f"Plans to buy within {months} months.", lambda r: r["timeline_months"] <= months),
        ("industry", f"Industry is one of: {', '.join(targets)}.", lambda r: r["industry"] in targets),
        ("territory", f"Region is one of: {', '.join(territory)}.", lambda r: r["region"] in territory),
        ("demo", "Has requested a product demo.", lambda r: r["demo_requested"]),
        ("engagement", f"Has held at least {meetings} meetings with us.", lambda r: r["meetings"] >= meetings),
    ]


def scenario(rng: random.Random) -> tuple[dict[str, Any], list[Criterion], list[bool], Any]:
    criteria = rng.sample(_pool(rng), 4)
    record = {"company": f"{rng.choice(['Blue', 'Iron', 'Swift', 'Cedar', 'Nova'])} "
                         f"{rng.choice(['Labs', 'Systems', 'Works', 'Group'])}",
              "employees": rng.choice([12, 45, 80, 150, 240, 420, 600, 900, 1500, 4000]),
              "budget": rng.choice([5_000, 12_000, 20_000, 30_000, 60_000, 90_000, 150_000]),
              "contact_role": rng.choice(ROLES), "timeline_months": rng.choice([1, 2, 4, 6, 8, 12, 18]),
              "industry": rng.choice(INDUSTRIES), "region": rng.choice(REGIONS),
              "demo_requested": rng.random() < 0.5, "meetings": rng.randint(0, 4)}
    met = [f(record) for _, _, f in criteria]
    rubric = {name: text for name, text, _ in criteria}
    state = join(render(rng, "Qualification rubric (one point per criterion met)", rubric),
                 render(rng, "Lead record", record))
    return record, criteria, met, state


def generate(rng: random.Random, kind: str, template: str) -> Row | None:
    _, criteria, met, state = scenario(rng)
    if kind == "score":
        return row("score", state, ["Rate this lead against the rubric.",
                                    "How many rubric criteria does this lead meet?",
                                    "Score the lead from 0 to 4, one point per criterion met."],
                   levels(RATING), sum(met), FAMILY)
    if kind == "noul":
        if template == "qualified":
            return row("noul", state, ["Is this lead qualified, meeting at least three of the four criteria?",
                                       "Does the lead score 3 or more on the rubric?"],
                       YES_NO, int(sum(met) >= 3), FAMILY)
        i = rng.randrange(4)
        name = criteria[i][0]
        return row("noul", state, [f"Does the lead meet the '{name}' criterion?",
                                   f"Is the '{name}' criterion satisfied by this record?"],
                   YES_NO, int(met[i]), FAMILY)
    if met.count(False) != 1:
        return None
    return row("choice", state, ["Which single criterion does this lead fail?",
                                 "The lead misses exactly one criterion. Which one?"],
               named([name for name, _, _ in criteria]), met.index(False), FAMILY)


TEMPLATES = {"noul": {"criterion": 1, "qualified": 1}, "choice": {"fails": 1}, "score": {"rating": 1}}
