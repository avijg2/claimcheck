"""Synthetic auto-insurance claims generator.

Everything here is invented data, generated from a fixed seed (42 by
default) so runs are reproducible. No real claims, people, or companies are
involved. Fraud is injected through five typologies designed for this lab:

- inflated_repair: repair estimate close to or above the vehicle value
- early_claim: claim filed days after the policy started
- velocity: many claims from the same claimant / repair shop in a short window
- duplicate: a repeat of an earlier claim on the same policy
- staged: several fraud signals at once (early + inflated + velocity)

Legitimate claims are drawn from calmer distributions, so the signals are
informative but not perfect — exactly the situation a rule lab should study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TYPOLOGIES = ["inflated_repair", "early_claim", "velocity", "duplicate", "staged"]
COLUMNS = [
    "claim_id", "policy_id", "claimant_id", "repair_shop_id", "incident_type",
    "days_since_policy_start", "claim_amount", "repair_estimate", "vehicle_value",
    "claimant_claims_12m", "shop_claims_30d", "prior_claims", "injuries_reported",
    "police_report_filed", "duplicate_claim", "is_fraud", "typology",
]


def _legit_row(rng: np.random.Generator, i: int) -> dict:
    vehicle_value = float(rng.uniform(8000, 45000))
    repair_ratio = float(rng.uniform(0.15, 0.65))
    repair_estimate = round(vehicle_value * repair_ratio, 2)
    return {
        "claim_id": f"CLM-{i:05d}",
        "policy_id": f"POL-{rng.integers(0, 2000):04d}",
        "claimant_id": f"CUST-{rng.integers(0, 3000):04d}",
        "repair_shop_id": f"SHOP-{rng.integers(0, 80):03d}",
        "incident_type": str(rng.choice(["collision", "theft", "weather", "vandalism"])),
        "days_since_policy_start": int(rng.integers(45, 1500)),
        "claim_amount": round(float(repair_estimate * rng.uniform(1.0, 1.25)), 2),
        "repair_estimate": repair_estimate,
        "vehicle_value": round(vehicle_value, 2),
        "claimant_claims_12m": int(rng.choice([0, 0, 0, 1, 1, 2])),
        "shop_claims_30d": int(rng.integers(1, 8)),
        "prior_claims": int(rng.choice([0, 0, 0, 1])),
        "injuries_reported": int(rng.choice([0, 0, 0, 1, 2])),
        "police_report_filed": bool(rng.random() < 0.6),
        "duplicate_claim": False,
        "is_fraud": False,
        "typology": "legitimate",
    }


def _fraud_row(rng: np.random.Generator, i: int, typology: str) -> dict:
    row = _legit_row(rng, i)
    row["is_fraud"] = True
    row["typology"] = typology
    vehicle_value = float(rng.uniform(8000, 30000))
    row["vehicle_value"] = round(vehicle_value, 2)
    if typology == "inflated_repair":
        row["repair_estimate"] = round(vehicle_value * float(rng.uniform(0.9, 1.15)), 2)
        row["claim_amount"] = round(row["repair_estimate"] * float(rng.uniform(1.1, 1.4)), 2)
        row["injuries_reported"] = int(rng.choice([2, 3, 4]))
    elif typology == "early_claim":
        row["days_since_policy_start"] = int(rng.integers(2, 28))
        row["repair_estimate"] = round(vehicle_value * float(rng.uniform(0.6, 0.9)), 2)
        row["claim_amount"] = round(row["repair_estimate"] * 1.2, 2)
    elif typology == "velocity":
        row["claimant_claims_12m"] = int(rng.integers(3, 7))
        row["shop_claims_30d"] = int(rng.integers(10, 22))
        row["prior_claims"] = int(rng.integers(2, 5))
    elif typology == "duplicate":
        row["duplicate_claim"] = True
        row["repair_estimate"] = round(vehicle_value * float(rng.uniform(0.5, 0.8)), 2)
        row["claim_amount"] = round(row["repair_estimate"] * 1.15, 2)
    elif typology == "staged":
        row["days_since_policy_start"] = int(rng.integers(5, 25))
        row["repair_estimate"] = round(vehicle_value * float(rng.uniform(0.9, 1.1)), 2)
        row["claim_amount"] = round(float(rng.uniform(16000, 28000)), 2)
        row["claimant_claims_12m"] = int(rng.integers(3, 6))
        row["shop_claims_30d"] = int(rng.integers(10, 18))
        row["injuries_reported"] = int(rng.integers(3, 6))
        row["police_report_filed"] = False
    return row


def generate_claims(n: int = 5000, fraud_rate: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """Generate n synthetic claims with the given fraud prevalence."""
    rng = np.random.default_rng(seed)
    n_fraud = int(round(n * fraud_rate))
    rows = []
    for i in range(n):
        if i < n_fraud:
            typology = TYPOLOGIES[i % len(TYPOLOGIES)]
            rows.append(_fraud_row(rng, i, typology))
        else:
            rows.append(_legit_row(rng, i))
    df = pd.DataFrame(rows, columns=COLUMNS)
    # Shuffle so fraud is not grouped at the top.
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_claims(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Add a train/dev/test split column (60/20/20, stratified by fraud)."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    splits = np.empty(len(out), dtype=object)
    for label in (False, True):
        idx = np.where(out["is_fraud"].to_numpy() == label)[0]
        shuffled = rng.permutation(idx)
        n = len(shuffled)
        n_train, n_dev = int(0.6 * n), int(0.8 * n)
        splits[shuffled[:n_train]] = "train"
        splits[shuffled[n_train:n_dev]] = "dev"
        splits[shuffled[n_dev:]] = "test"
    out["split"] = splits
    return out
