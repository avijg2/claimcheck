"""Feature extraction: turn a claim (or a dataframe row) into rule inputs."""

from __future__ import annotations

from .models import Claim

# Numeric fields the rule loop is allowed to propose thresholds for.
NUMERIC_FIELDS = [
    "days_since_policy_start",
    "claim_amount",
    "repair_estimate",
    "vehicle_value",
    "repair_ratio",
    "claimant_claims_12m",
    "shop_claims_30d",
    "prior_claims",
    "injuries_reported",
]


def claim_to_features(claim: Claim) -> dict:
    """Return a flat dict of numeric / boolean features for one claim."""
    return {
        "days_since_policy_start": claim.days_since_policy_start,
        "claim_amount": claim.claim_amount,
        "repair_estimate": claim.repair_estimate,
        "vehicle_value": claim.vehicle_value,
        "repair_ratio": round(claim.repair_ratio, 4),
        "claimant_claims_12m": claim.claimant_claims_12m,
        "shop_claims_30d": claim.shop_claims_30d,
        "prior_claims": claim.prior_claims,
        "injuries_reported": claim.injuries_reported,
        "police_report_filed": claim.police_report_filed,
        "duplicate_claim": claim.duplicate_claim,
    }


def add_features(df):
    """Add derived columns (repair_ratio) to a claims dataframe, in place copy."""
    out = df.copy()
    out["repair_ratio"] = (out["repair_estimate"] / out["vehicle_value"]).round(4)
    return out
