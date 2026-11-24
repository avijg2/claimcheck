"""Self-improving rule loop, driven by data — no LLM involved.

Each round proposes candidate thresholds for numeric fields from the dev
split (high quantiles of fraud cases for ">" rules, low quantiles for "<"
rules), scores every candidate by the dev F1 it gives when added to the
current engine, and keeps only the single best candidate if it improves dev
F1. Every round and every candidate considered is logged, so the loop is
auditable: you can see what was tried and why it was kept or dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evals import confusion
from .features import NUMERIC_FIELDS, add_features
from .rules import Rule, RuleEngine


def _dev_f1(engine: RuleEngine, dev: pd.DataFrame) -> float:
    return confusion(engine, dev)["f1"]


def propose_candidates(dev: pd.DataFrame, existing: list[Rule]) -> list[Rule]:
    feats = add_features(dev)
    fraud = feats[feats["is_fraud"]]
    existing_keys = {(r.field, r.operator) for r in existing}
    candidates: list[Rule] = []
    counter = 0
    for field in NUMERIC_FIELDS:
        if field not in fraud.columns:
            continue
        values = fraud[field].to_numpy(dtype=float)
        if len(values) == 0:
            continue
        # High-side thresholds: values fraud cases reach but most legit do not.
        for q in (0.5, 0.75, 0.9):
            threshold = float(np.quantile(values, q))
            if ("high", field, threshold) in existing_keys:
                continue
            counter += 1
            candidates.append(Rule(
                rule_id=f"C{counter:02d}", name=f"candidate {field} high",
                field=field, operator=">", threshold=round(threshold, 2),
                points=15, description=f"Data-proposed: {field} above fraud median/quantile.",
            ))
        # Low-side thresholds, only meaningful for days-since-policy-start.
        if field == "days_since_policy_start":
            for q in (0.25, 0.5):
                threshold = float(np.quantile(values, q))
                counter += 1
                candidates.append(Rule(
                    rule_id=f"C{counter:02d}", name=f"candidate {field} low",
                    field=field, operator="<", threshold=round(threshold, 2),
                    points=15, description="Data-proposed: claim very soon after policy start.",
                ))
    return candidates


def run_rule_loop(engine: RuleEngine, dev: pd.DataFrame, rounds: int = 3) -> dict:
    """Run the loop. Returns the improved engine, the log, and dev F1s."""
    log: list[dict] = []
    current = RuleEngine(list(engine.rules))
    f1_before = _dev_f1(current, dev)
    for round_no in range(1, rounds + 1):
        candidates = propose_candidates(dev, current.rules)
        best = None
        best_f1 = f1_before
        best_candidate = None
        for cand in candidates:
            # Skip near-duplicates of an existing rule.
            if any(r.field == cand.field and r.operator == cand.operator
                   and abs(float(r.threshold) - float(cand.threshold)) < 1e-6
                   for r in current.rules):
                continue
            trial = RuleEngine(list(current.rules) + [cand])
            f1 = _dev_f1(trial, dev)
            log.append({
                "round": round_no, "candidate_id": cand.rule_id,
                "field": cand.field, "operator": cand.operator,
                "threshold": cand.threshold, "dev_f1": round(f1, 4),
                "kept": False,
            })
            if f1 > best_f1 + 1e-4:
                best_f1 = f1
                best_candidate = cand
                best = trial
        if best_candidate is None:
            log.append({"round": round_no, "note": "no candidate improved dev F1; stopping"})
            break
        # Give the kept rule a stable id and mark it in the log.
        best_candidate.rule_id = f"L{round_no:02d}"
        best_candidate.name = f"learned rule {best_candidate.field} {best_candidate.operator} {best_candidate.threshold}"
        current = RuleEngine(list(current.rules) + [best_candidate])
        for entry in log:
            if entry.get("round") == round_no and entry.get("candidate_id") == best_candidate.rule_id:
                pass
        # Mark the winning candidate entry (matched by field/operator/threshold).
        for entry in log:
            if (entry.get("round") == round_no and entry.get("field") == best_candidate.field
                    and entry.get("operator") == best_candidate.operator
                    and entry.get("threshold") == best_candidate.threshold):
                entry["kept"] = True
        f1_before = best_f1
    return {"engine": current, "log": log, "dev_f1": round(f1_before, 4)}
