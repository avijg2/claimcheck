"""Demo: score three example claims, run the reviewer panel, print results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimcheck.models import Claim
from claimcheck.reviewers import run_panel
from claimcheck.rules import RuleEngine

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    engine = RuleEngine.from_file()
    with open(ROOT / "examples" / "sample_claims.json", encoding="utf-8") as f:
        examples = json.load(f)
    for ex in examples:
        claim = Claim(**ex)
        result = engine.score_claim(claim)
        panel = run_panel(claim, engine)
        print(f"\n=== {claim.claim_id} ({ex.get('label', 'claim')}) ===")
        print(f"Score: {result.score} | risk: {result.risk_band} | flagged: {result.flagged}")
        print(f"Explanation: {result.explanation}")
        print(f"Panel consensus: {panel.consensus} — {panel.note}")
        for r in panel.reviews:
            print(f"  - {r.reviewer}: {r.verdict} (confidence {r.confidence:.2f})")
            for reason in r.reasons:
                print(f"      {reason}")


if __name__ == "__main__":
    main()
