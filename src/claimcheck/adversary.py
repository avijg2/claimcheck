"""Adversary evasion test.

Take fraud claims and nudge each fraud signal just below its rule
threshold — the repair ratio a hair under the limit, the claim filed a day
after the early-claim window, velocity one claim under the limit, and so on.
Evasion rate is the share of these doctored fraud claims the engine no
longer flags. A good rule set should catch some of them anyway (signals
overlap), but the number is reported honestly, not hidden.
"""

from __future__ import annotations

import pandas as pd

from .evals import confusion
from .features import add_features
from .rules import RuleEngine


def make_adversarial(df: pd.DataFrame, engine: RuleEngine, seed: int = 7) -> pd.DataFrame:
    """Return a doctored copy of fraud claims, signals nudged under thresholds.

    Each rule's signal is nudged on a random 80% of rows (seeded): a real
    adversary does not know every threshold exactly, and does not manage to
    hide every signal on every claim.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    out = add_features(df.copy())
    # Learned thresholds can be fractional; keep numeric columns float so
    # clipped values below/above a threshold can always be stored.
    for col in out.columns:
        if col in ("claim_id", "policy_id", "claimant_id", "repair_shop_id",
                   "incident_type", "typology", "split"):
            continue
        if out[col].dtype.kind in "iuf":
            out[col] = out[col].astype(float)
    for rule in engine.rules:
        field, op, thr = rule.field, rule.operator, rule.threshold
        if field not in out.columns:
            continue
        mask = rng.random(len(out)) < 0.8
        try:
            thr_f = float(thr)
        except (TypeError, ValueError):
            if isinstance(thr, bool):
                out.loc[mask, field] = False
            continue
        if op == ">":
            out.loc[mask, field] = out.loc[mask, field].clip(upper=thr_f * 0.99)
            if field == "repair_ratio":
                out["repair_estimate"] = (out["repair_ratio"] * out["vehicle_value"]).round(2)
        elif op == ">=":
            out.loc[mask, field] = out.loc[mask, field].clip(upper=thr_f - 1)
            if field == "repair_ratio":
                out["repair_estimate"] = (out["repair_ratio"] * out["vehicle_value"]).round(2)
        elif op == "<":
            out.loc[mask, field] = out.loc[mask, field].clip(lower=thr_f + 1)
        elif op == "<=":
            out.loc[mask, field] = out.loc[mask, field].clip(lower=thr_f + 1)
    return out


def evasion_rate(engine: RuleEngine, fraud_df: pd.DataFrame) -> float:
    """Share of (doctored) fraud claims the engine does NOT flag."""
    if len(fraud_df) == 0:
        return 0.0
    metrics = confusion(engine, fraud_df.assign(is_fraud=True))
    # confusion on all-fraud data: recall = flagged share, so evasion = 1 - recall.
    return round(1.0 - metrics["recall"], 4)
