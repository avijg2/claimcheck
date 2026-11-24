# ClaimCheck — Results

All numbers below were measured by `scripts/run_eval.py` on synthetic claims
only (seed 42, n=8000 total, 1600 in the test split). Nothing here
is a real claim, and these rules are not a production fraud system.

## Base rules (10 rules)

| Prevalence | Precision | Recall | F1 | False-positive rate | False-negative rate | n |
|---|---:|---:|---:|---:|---:|---:|
| 1% | 1.0000 | 0.7333 | 0.8462 | 0.0000 | 0.2667 | 1535 |
| 5% | 1.0000 | 0.7875 | 0.8811 | 0.0000 | 0.2125 | 1600 |

## After the self-improving rule loop (12 rules)

Dev F1 after the loop: 0.9333. Learned rules added:
- {"rule_id": "L01", "name": "learned rule repair_ratio > 0.9", "field": "repair_ratio", "operator": ">", "threshold": 0.9, "points": 15, "description": "Data-proposed: repair_ratio above fraud median/quantile."}
- {"rule_id": "L02", "name": "learned rule days_since_policy_start < 16.5", "field": "days_since_policy_start", "operator": "<", "threshold": 16.5, "points": 15, "description": "Data-proposed: claim very soon after policy start."}

| Prevalence | Precision | Recall | F1 | False-positive rate | False-negative rate | n |
|---|---:|---:|---:|---:|---:|---:|
| 1% | 1.0000 | 0.7333 | 0.8462 | 0.0000 | 0.2667 | 1535 |
| 5% | 1.0000 | 0.8250 | 0.9041 | 0.0000 | 0.1750 | 1600 |

## Adversary evasion test

Share of doctored fraud claims (signals nudged just below thresholds) that
the engine no longer flags:

- Before the rule loop: 0.7750
- After the rule loop: 0.6375

Threshold rules are evadable by design — an adversary who knows the
thresholds can slip under them. The loop helps only a little, because the
learned rules use the same fields. That is a real limitation of this
approach, and the reason a production system would need behaviour and
network features, not just thresholds.

## Limitations

- All data is synthetic, generated from distributions chosen for this lab.
  Real fraud looks different, and real base rates vary by portfolio.
- Precision stayed at 1.0000 even at 1% prevalence, with a false-positive
  rate of 0.0000. That is not a real-world claim — it is a synthetic-data
  artefact: legitimate claims in this generator almost never stack enough
  rule hits to reach the 40-point flag line. Real legitimate claims are
  messier, so real precision would be lower. The weak side here is recall
  (0.73–0.83): roughly a fifth to a quarter of fraud is missed.
- Rules are transparent and explainable, but also brittle: they cannot
  catch fraud that produces no signal on any rule field.
- The reviewer panel only uses evidence from the claim and the rule hits;
  it cannot verify anything outside the claim.
