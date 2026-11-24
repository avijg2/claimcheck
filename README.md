# ClaimCheck

**One sentence:** ClaimCheck scores synthetic auto-insurance claims with transparent, human-readable rules, improves those rules from data, stress-tests them against an adversary, and has two reviewers check the evidence — so you can see exactly why a claim was flagged.

This is a deterministic rule lab, not a black-box model. There is no machine-learning classifier here on purpose: every score is a sum of rule points, every rule is a line in `data/rules.json`, and every flag can be explained by listing the rules that hit. That makes the failure modes visible, which is the point.

## Architecture

```
synthetic claims (seed 42)
        │
        ▼
features (repair_ratio, velocity, …) ──► rule engine (data/rules.json)
        │                                        │ score + rule hits
        ├─► self-improving rule loop (dev split, F1-gated)
        ├─► adversary evasion test (signals nudged under thresholds)
        ├─► reviewer panel (rule reviewer + evidence reviewer, optional OpenAI)
        └─► evaluation at 1% and 5% fraud prevalence
        │
        ▼
FastAPI service: /health · /score · /review · /metrics
```

## How to run

```bash
pip install -r requirements.txt
pip install -e .

# Generate the synthetic dataset
python scripts/generate_data.py

# Score three example claims and run the reviewer panel
python scripts/demo.py

# Full evaluation — writes RESULTS.md and data/eval_summary.json
python scripts/run_eval.py

# API
uvicorn claimcheck.api:app --reload
# GET  /health   — service status and rule count
# POST /score    — one claim → score, risk band, rule hits, explanation
# POST /review   — one claim → two-reviewer panel consensus
# GET  /metrics  — eval summary on a synthetic test split

# Tests
python -m pytest -q
```

## Results

Measured by `scripts/run_eval.py` on synthetic claims only (seed 42). Full numbers in [RESULTS.md](RESULTS.md).

| Prevalence | Precision | Recall | F1 | False-positive rate | False-negative rate |
|---|---:|---:|---:|---:|---:|
| 1% (after rule loop) | 1.0000 | 0.7333 | 0.8462 | 0.0000 | 0.2667 |
| 5% (after rule loop) | 1.0000 | 0.8250 | 0.9041 | 0.0000 | 0.1750 |

Base rules (before the loop): 1% precision 1.0000 / recall 0.7333 / F1 0.8462; 5% precision 1.0000 / recall 0.7875 / F1 0.8811. The loop added 2 learned rules (repair_ratio > 0.9, days_since_policy_start < 16.5), dev F1 0.9333. Adversary evasion: 0.7750 before the loop, 0.6375 after — threshold rules are evadable, and the loop only helps a little.

## What the numbers mean

- Precision is 1.0000 at both prevalences with a false-positive rate of 0.0000 — but that is a synthetic-data artefact, not a real-world claim: legitimate claims in this generator almost never stack enough rule hits to reach the 40-point flag line. Real precision would be lower.
- The weak side is recall (0.73 at 1%, 0.83 after the loop at 5%): roughly a fifth of fraud is missed, and the adversary test shows why — 64% of doctored fraud claims evade the improved rules.
- The reviewer panel abstains when there is no evidence. A low score with no rule hits is not "approved with confidence" — it is "nothing to judge on".

## Limitations

- **All data is synthetic.** Distributions were chosen for this lab; real fraud looks different, and real base rates vary by portfolio.
- **Rules are not a production fraud system.** They are transparent and explainable, but brittle: fraud that produces no signal on any rule field is invisible to them.
- **No real claims, people, or companies are involved.** Claim IDs, shops, and claimants are invented.
- The optional OpenAI reviewer is only used when `OPENAI_API_KEY` is set, and it is instructed to use only the claim facts and rule hits it is given.

## Project layout

```
src/claimcheck/   models, rule engine, features, data generation,
                  rule loop, adversary test, reviewers, evals, FastAPI app
data/rules.json   the base rules (human-readable, editable)
scripts/          generate_data.py · run_eval.py · demo.py
examples/         sample claims + expected outputs
tests/            pytest suite (rule engine, data, evals, loop, adversary, reviewers, API)
```
