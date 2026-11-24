"""Deterministic rule engine.

Rules live in a JSON file (data/rules.json), not in code. Each rule names a
field, an operator, a threshold and a point value. Scoring is a plain sum of
the points of the rules that hit, so every score can be explained by listing
its hits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .features import claim_to_features
from .models import Claim, RuleHit, ScoreResult

FLAG_THRESHOLD = 40  # score at or above this = flagged for investigation
REVIEW_THRESHOLD = 20  # score at or above this = review band

OPERATORS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


@dataclass
class Rule:
    rule_id: str
    name: str
    field: str
    operator: str
    threshold: float | bool
    points: int
    description: str = ""

    def hits(self, features: dict) -> bool:
        if self.field not in features:
            return False
        value = features[self.field]
        fn = OPERATORS.get(self.operator)
        if fn is None:
            raise ValueError(f"unknown operator {self.operator!r} in rule {self.rule_id}")
        return bool(fn(value, self.threshold))

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "field": self.field,
            "operator": self.operator,
            "threshold": self.threshold,
            "points": self.points,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        return cls(
            rule_id=d["rule_id"],
            name=d["name"],
            field=d["field"],
            operator=d["operator"],
            threshold=d["threshold"],
            points=int(d["points"]),
            description=d.get("description", ""),
        )


def load_rules(path: str | Path | None = None) -> list[Rule]:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "rules.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [Rule.from_dict(r) for r in raw["rules"]]


def risk_band(score: int) -> str:
    if score >= FLAG_THRESHOLD:
        return "high"
    if score >= REVIEW_THRESHOLD:
        return "medium"
    return "low"


class RuleEngine:
    """Scores claims against a list of rules."""

    def __init__(self, rules: list[Rule]):
        self.rules = list(rules)

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> "RuleEngine":
        return cls(load_rules(path))

    def score_claim(self, claim: Claim) -> ScoreResult:
        features = claim_to_features(claim)
        return self.score_features(features, claim_id=claim.claim_id)

    def score_features(self, features: dict, claim_id: str = "") -> ScoreResult:
        hits: list[RuleHit] = []
        total = 0
        for rule in self.rules:
            if rule.hits(features):
                total += rule.points
                hits.append(
                    RuleHit(
                        rule_id=rule.rule_id,
                        name=rule.name,
                        points=rule.points,
                        detail=f"{rule.field} {rule.operator} {rule.threshold} "
                        f"(value {features.get(rule.field)})",
                    )
                )
        band = risk_band(total)
        if hits:
            explanation = (
                f"Score {total} ({band} risk): "
                + "; ".join(f"{h.rule_id} +{h.points}" for h in hits)
                + ". No other rules hit."
            )
        else:
            explanation = f"Score 0 (low risk): no rules hit."
        return ScoreResult(
            claim_id=claim_id,
            score=total,
            risk_band=band,
            flagged=total >= FLAG_THRESHOLD,
            hits=hits,
            explanation=explanation,
        )

    def flag_features(self, features: dict) -> bool:
        return self.score_features(features).flagged
