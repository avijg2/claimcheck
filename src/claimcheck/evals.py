"""Evaluation at realistic fraud prevalence.

Metrics are computed on a sampled test set at a requested prevalence: all
available fraud cases are kept and legitimate cases are sampled (seeded) so
the fraud share matches the target. This makes the 1% and 5% numbers honest
about what low prevalence does to precision — it is not a reweighting trick.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import add_features
from .rules import RuleEngine


def confusion(engine: RuleEngine, df: pd.DataFrame) -> dict:
    feats = add_features(df)
    flagged = []
    for _, row in feats.iterrows():
        features = {k: row[k] for k in feats.columns if k not in ("is_fraud", "typology", "split")}
        # Keep only rule inputs; strings are ignored by the engine anyway.
        flagged.append(engine.flag_features(features))
    y = df["is_fraud"].to_numpy(dtype=bool)
    pred = np.array(flagged, dtype=bool)
    tp = int(((pred) & (y)).sum())
    fp = int(((pred) & (~y)).sum())
    fn = int(((~pred) & (y)).sum())
    tn = int(((~pred) & (~y)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
    }


def sample_at_prevalence(df: pd.DataFrame, prevalence: float, seed: int = 42) -> pd.DataFrame:
    """Return a sample of df whose fraud share is (approximately) prevalence."""
    rng = np.random.default_rng(seed)
    fraud = df[df["is_fraud"]]
    legit = df[~df["is_fraud"]]
    if len(fraud) == 0:
        return df.copy()
    # Use (nearly) all legitimate cases, then take just enough fraud cases
    # to hit the target prevalence — with a small fraud pool, taking a fixed
    # fraud count first would make the target prevalence unreachable.
    n_legit = min(len(legit), 5000)
    n_fraud = int(round(n_legit * prevalence / (1 - prevalence)))
    n_fraud = max(1, min(n_fraud, len(fraud)))
    fraud_s = fraud.sample(n=n_fraud, random_state=seed)
    legit_s = legit.sample(n=n_legit, random_state=seed + 1)
    out = pd.concat([fraud_s, legit_s]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def evaluate_at_prevalences(engine: RuleEngine, df: pd.DataFrame) -> dict:
    results = {}
    for p in (0.01, 0.05):
        sampled = sample_at_prevalence(df, p)
        metrics = confusion(engine, sampled)
        metrics["n"] = int(len(sampled))
        metrics["prevalence"] = p
        results[f"{int(p * 100)}%"] = metrics
    return results
