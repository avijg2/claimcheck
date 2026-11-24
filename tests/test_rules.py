import json
from pathlib import Path

import pytest

from claimcheck.models import Claim
from claimcheck.rules import Rule, RuleEngine, load_rules, risk_band

ROOT = Path(__file__).resolve().parents[1]


def ordinary_claim() -> Claim:
    return Claim(claim_id="T1", days_since_policy_start=400, claim_amount=4200,
                 repair_estimate=3800, vehicle_value=22000, claimant_claims_12m=0,
                 shop_claims_30d=3, prior_claims=0, injuries_reported=0,
                 police_report_filed=True, duplicate_claim=False)


def test_load_rules_from_json():
    rules = load_rules()
    assert len(rules) == 10
    assert {r.rule_id for r in rules} == {f"R{i:02d}" for i in range(1, 11)}


def test_rule_operators():
    r = Rule("X", "x", "claim_amount", ">", 100, 5)
    assert r.hits({"claim_amount": 101}) is True
    assert r.hits({"claim_amount": 100}) is False
    r2 = Rule("X", "x", "duplicate_claim", "==", True, 5)
    assert r2.hits({"duplicate_claim": True}) is True
    assert r2.hits({"duplicate_claim": False}) is False
    with pytest.raises(ValueError):
        Rule("X", "x", "claim_amount", "~~", 1, 5).hits({"claim_amount": 2})


def test_ordinary_claim_scores_zero():
    engine = RuleEngine.from_file()
    result = engine.score_claim(ordinary_claim())
    assert result.score == 0
    assert result.risk_band == "low"
    assert result.flagged is False
    assert result.hits == []


def test_suspicious_claim_is_flagged():
    engine = RuleEngine.from_file()
    claim = Claim(claim_id="T2", days_since_policy_start=12, claim_amount=21000,
                  repair_estimate=19500, vehicle_value=20000, claimant_claims_12m=4,
                  shop_claims_30d=14, prior_claims=3, injuries_reported=4,
                  police_report_filed=False, duplicate_claim=False)
    result = engine.score_claim(claim)
    assert result.score >= 100
    assert result.risk_band == "high"
    assert result.flagged is True
    assert "R02" in [h.rule_id for h in result.hits]


def test_duplicate_claim_scores_30_and_is_not_flagged():
    engine = RuleEngine.from_file()
    claim = ordinary_claim()
    claim.duplicate_claim = True
    result = engine.score_claim(claim)
    assert result.score == 30
    assert result.risk_band == "medium"
    assert result.flagged is False


def test_risk_band_boundaries():
    assert risk_band(0) == "low"
    assert risk_band(19) == "low"
    assert risk_band(20) == "medium"
    assert risk_band(39) == "medium"
    assert risk_band(40) == "high"


def test_explanation_lists_hits():
    engine = RuleEngine.from_file()
    claim = ordinary_claim()
    claim.duplicate_claim = True
    result = engine.score_claim(claim)
    assert "R06" in result.explanation
    assert "30" in result.explanation


def test_rules_json_is_valid_and_human_readable():
    raw = json.loads((ROOT / "data" / "rules.json").read_text())
    assert "rules" in raw
    for r in raw["rules"]:
        assert set(r) >= {"rule_id", "field", "operator", "threshold", "points"}
