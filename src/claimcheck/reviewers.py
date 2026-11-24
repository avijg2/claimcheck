"""Two-reviewer panel.

Default reviewers are deterministic and offline. They only use evidence
from the claim itself and the rule hits — they never invent claim facts,
never quote numbers that are not in the claim or a rule hit, and they cite
the rule IDs they relied on. A reviewer abstains when there is no evidence
to judge on.

An optional OpenAI-backed reviewer is available only when OPENAI_API_KEY is
set. It is given the same evidence-only instruction and the same claim
facts; if the key is absent it is simply not used.
"""

from __future__ import annotations

import json
import os

import httpx

from .models import Claim, PanelResult, Review
from .rules import RuleEngine


def _verdict_from_score(score: int) -> str:
    if score >= 40:
        return "escalate"
    if score >= 20:
        return "review"
    return "approve"


class RuleReviewer:
    """Reviewer 1: judges by the rule-engine score and its hits."""

    name = "rule_reviewer"

    def review(self, claim: Claim, engine: RuleEngine) -> Review:
        result = engine.score_claim(claim)
        if not result.hits:
            return Review(
                reviewer=self.name, verdict="abstain", confidence=0.0,
                reasons=["No rule hit, so there is no evidence in the claim to judge on."],
                rule_ids_cited=[],
            )
        reasons = [
            f"{h.rule_id}: {h.name} (+{h.points}) — {h.detail}"
            for h in result.hits
        ]
        reasons.append(f"Total score {result.score}, risk band {result.risk_band}.")
        return Review(
            reviewer=self.name,
            verdict=_verdict_from_score(result.score),
            confidence=min(0.95, 0.5 + 0.05 * len(result.hits)),
            reasons=reasons,
            rule_ids_cited=[h.rule_id for h in result.hits],
        )


class EvidenceReviewer:
    """Reviewer 2: checks internal consistency of the claim's own facts."""

    name = "evidence_reviewer"

    def review(self, claim: Claim, engine: RuleEngine) -> Review:
        result = engine.score_claim(claim)
        reasons: list[str] = []
        cited: list[str] = []
        if claim.repair_ratio > 1.0:
            reasons.append(
                f"Repair estimate ${claim.repair_estimate:,.0f} exceeds vehicle value "
                f"${claim.vehicle_value:,.0f} (repair ratio {claim.repair_ratio:.2f})."
            )
            cited.append("R02")
        if claim.duplicate_claim:
            reasons.append("Claim is marked as a duplicate of an earlier claim.")
            cited.append("R06")
        if claim.days_since_policy_start < 30 and claim.claim_amount > 10000:
            reasons.append(
                f"Claim of ${claim.claim_amount:,.0f} filed "
                f"{claim.days_since_policy_start} days after the policy started."
            )
            cited.append("R01")
        if claim.claimant_claims_12m >= 3:
            reasons.append(
                f"Claimant has {claim.claimant_claims_12m} claims in 12 months."
            )
            cited.append("R04")
        if not reasons:
            return Review(
                reviewer=self.name, verdict="abstain", confidence=0.0,
                reasons=["No internal inconsistency found in the claim's own facts."],
                rule_ids_cited=[],
            )
        verdict = "escalate" if result.score >= 40 else "review"
        return Review(
            reviewer=self.name, verdict=verdict,
            confidence=0.8 if len(reasons) >= 2 else 0.6,
            reasons=reasons, rule_ids_cited=sorted(set(cited)),
        )


class OpenAIReviewer:
    """Optional reviewer 3: calls OpenAI with an evidence-only instruction."""

    name = "openai_reviewer"

    def __init__(self, model: str | None = None):
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def available(self) -> bool:
        return bool(self.api_key)

    def review(self, claim: Claim, engine: RuleEngine) -> Review:
        if not self.available():
            raise RuntimeError("OPENAI_API_KEY is not set; the OpenAI reviewer is unavailable.")
        result = engine.score_claim(claim)
        evidence = {
            "claim_id": claim.claim_id,
            "score": result.score,
            "risk_band": result.risk_band,
            "rule_hits": [h.model_dump() for h in result.hits],
            "claim_amount": claim.claim_amount,
            "repair_ratio": round(claim.repair_ratio, 4),
            "days_since_policy_start": claim.days_since_policy_start,
        }
        prompt = (
            "You are a fraud-review assistant. Use ONLY the evidence in the JSON below. "
            "Cite rule IDs for every reason. Do not invent any claim facts or numbers. "
            "If the evidence is insufficient, say so and abstain. "
            "Reply with JSON: {\"verdict\": \"approve|review|escalate|abstain\", "
            "\"confidence\": 0-1, \"reasons\": [\"...\"]}.\n\n"
            + json.dumps(evidence)
        )
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model,
                      "messages": [{"role": "user", "content": prompt}],
                      "response_format": {"type": "json_object"}},
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
        data = json.loads(text)
        return Review(
            reviewer=self.name,
            verdict=data.get("verdict", "abstain"),
            confidence=float(data.get("confidence", 0.5)),
            reasons=list(data.get("reasons", [])),
            rule_ids_cited=[h.rule_id for h in result.hits],
        )


def run_panel(claim: Claim, engine: RuleEngine, use_openai: bool = False) -> PanelResult:
    """Run the two offline reviewers; add the OpenAI reviewer if asked and keyed."""
    reviews = [RuleReviewer().review(claim, engine), EvidenceReviewer().review(claim, engine)]
    if use_openai:
        reviewer = OpenAIReviewer()
        if reviewer.available():
            reviews.append(reviewer.review(claim, engine))
    decided = [r for r in reviews if r.verdict != "abstain"]
    if not decided:
        consensus = "abstain"
        note = "Both reviewers abstained: no evidence in the claim to judge on."
    elif any(r.verdict == "escalate" for r in decided):
        consensus = "escalate"
        note = "At least one reviewer escalated based on the cited rule hits."
    elif any(r.verdict == "review" for r in decided):
        consensus = "review"
        note = "Reviewers recommend a human review of the cited evidence."
    else:
        consensus = "approve"
        note = "No reviewer found evidence to escalate."
    return PanelResult(claim_id=claim.claim_id, reviews=reviews, consensus=consensus, note=note)
