"""Data models for a single insurance claim and for scoring results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """One auto-insurance claim. All fields are plain, checkable facts."""

    claim_id: str = "CLM-0001"
    policy_id: str = "POL-0001"
    claimant_id: str = "CUST-0001"
    repair_shop_id: str = "SHOP-001"
    incident_type: str = "collision"
    days_since_policy_start: int = Field(ge=0, default=365)
    claim_amount: float = Field(ge=0, default=5000.0)
    repair_estimate: float = Field(ge=0, default=3000.0)
    vehicle_value: float = Field(ge=1, default=20000.0)
    claimant_claims_12m: int = Field(ge=0, default=0)
    shop_claims_30d: int = Field(ge=0, default=2)
    prior_claims: int = Field(ge=0, default=0)
    injuries_reported: int = Field(ge=0, default=0)
    police_report_filed: bool = False
    duplicate_claim: bool = False

    @property
    def repair_ratio(self) -> float:
        """Repair estimate as a share of the vehicle's value."""
        return self.repair_estimate / self.vehicle_value if self.vehicle_value else 0.0


class RuleHit(BaseModel):
    rule_id: str
    name: str
    points: int
    detail: str


class ScoreResult(BaseModel):
    claim_id: str
    score: int
    risk_band: str
    flagged: bool
    hits: list[RuleHit] = []
    explanation: str


class Review(BaseModel):
    reviewer: str
    verdict: str  # approve | review | escalate | abstain
    confidence: float
    reasons: list[str] = []
    rule_ids_cited: list[str] = []


class PanelResult(BaseModel):
    claim_id: str
    reviews: list[Review] = []
    consensus: str
    note: str
