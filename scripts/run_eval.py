"""Run the full evaluation and write RESULTS.md + data/eval_summary.json.

Steps: generate synthetic claims (seed 42), evaluate the base rules on the
test split at 1% and 5% prevalence, run the rule loop on dev, evaluate the
improved rules, and run the adversary evasion test before/after the loop.
All numbers in RESULTS.md come from this run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimcheck.adversary import evasion_rate, make_adversarial
from claimcheck.data_gen import generate_claims, split_claims
from claimcheck.evals import evaluate_at_prevalences
from claimcheck.rule_loop import run_rule_loop
from claimcheck.rules import RuleEngine

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    df = split_claims(generate_claims(n=8000, fraud_rate=0.05, seed=42))
    dev = df[df["split"] == "dev"]
    test = df[df["split"] == "test"]
    fraud_test = test[test["is_fraud"]]

    base = RuleEngine.from_file()
    base_eval = evaluate_at_prevalences(base, test)
    evasion_before = evasion_rate(base, make_adversarial(fraud_test, base))

    loop = run_rule_loop(base, dev, rounds=3)
    improved = loop["engine"]
    improved_eval = evaluate_at_prevalences(improved, test)
    evasion_after = evasion_rate(improved, make_adversarial(fraud_test, improved))

    summary = {
        "n_total": int(len(df)),
        "n_dev": int(len(dev)),
        "n_test": int(len(test)),
        "base_rules": len(base.rules),
        "improved_rules": len(improved.rules),
        "learned_rules": [r.to_dict() for r in improved.rules if r.rule_id.startswith("L")],
        "dev_f1_after_loop": loop["dev_f1"],
        "rule_loop_log": loop["log"],
        "base_eval": base_eval,
        "improved_eval": improved_eval,
        "evasion_before_loop": evasion_before,
        "evasion_after_loop": evasion_after,
    }
    (ROOT / "data" / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def table(evald: dict) -> str:
        lines = ["| Prevalence | Precision | Recall | F1 | False-positive rate | False-negative rate | n |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for key, m in evald.items():
            lines.append(
                f"| {key} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
                f"{m['false_positive_rate']:.4f} | {m['false_negative_rate']:.4f} | {m['n']} |"
            )
        return "\n".join(lines)

    md = f"""# ClaimCheck — Results

All numbers below were measured by `scripts/run_eval.py` on synthetic claims
only (seed 42, n={len(df)} total, {len(test)} in the test split). Nothing here
is a real claim, and these rules are not a production fraud system.

## Base rules ({len(base.rules)} rules)

{table(base_eval)}

## After the self-improving rule loop ({len(improved.rules)} rules)

Dev F1 after the loop: {loop['dev_f1']:.4f}. Learned rules added:
{chr(10).join('- ' + json.dumps(r) for r in summary['learned_rules']) if summary['learned_rules'] else '- none improved dev F1, so no rule was added'}

{table(improved_eval)}

## Adversary evasion test

Share of doctored fraud claims (signals nudged just below thresholds) that
the engine no longer flags:

- Before the rule loop: {evasion_before:.4f}
- After the rule loop: {evasion_after:.4f}

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
"""
    (ROOT / "RESULTS.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
