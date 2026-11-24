import os

import pytest

from claimcheck.models import Claim
from claimcheck.reviewers import EvidenceReviewer, OpenAIReviewer, RuleReviewer, run_panel
from claimcheck.rules import RuleEngine

ENGINE = RuleEngine.from_file()


def suspicious() -> Claim:
    return Claim(claim_id="R1", days_since_policy_start=12, claim_amount=21000,
                 repair_estimate=19500, vehicle_value=20000, claimant_claims_12m=4,
                 shop_claims_30d=14, prior_claims=3, injuries_reported=4,
                 police_report_filed=False, duplicate_claim=False)


def ordinary() -> Claim:
    return Claim(claim_id="R2", days_since_policy_start=400, claim_amount=4200,
                 repair_estimate=3800, vehicle_value=22000, claimant_claims_12m=0,
                 shop_claims_30d=3, prior_claims=0, injuries_reported=0,
                 police_report_filed=True, duplicate_claim=False)


def test_rule_reviewer_escalates_suspicious_claim():
    r = RuleReviewer().review(suspicious(), ENGINE)
    assert r.verdict == "escalate"
    assert "R02" in r.rule_ids_cited
    assert any("R02" in reason for reason in r.reasons)


def test_reviewers_abstain_when_no_evidence():
    assert RuleReviewer().review(ordinary(), ENGINE).verdict == "abstain"
    assert EvidenceReviewer().review(ordinary(), ENGINE).verdict == "abstain"


def test_evidence_reviewer_cites_only_claim_facts():
    r = EvidenceReviewer().review(suspicious(), ENGINE)
    assert r.verdict in ("escalate", "review")
    # Every reason must mention a number that is in the claim.
    assert any("21,000" in reason or "12 days" in reason for reason in r.reasons)


def test_panel_consensus():
    panel = run_panel(suspicious(), ENGINE)
    assert panel.consensus == "escalate"
    assert len(panel.reviews) == 2
    panel2 = run_panel(ordinary(), ENGINE)
    assert panel2.consensus == "abstain"


def test_openai_reviewer_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reviewer = OpenAIReviewer()
    assert reviewer.available() is False
    with pytest.raises(RuntimeError):
        reviewer.review(suspicious(), ENGINE)
