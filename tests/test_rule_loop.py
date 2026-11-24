from claimcheck.data_gen import generate_claims, split_claims
from claimcheck.evals import confusion
from claimcheck.rule_loop import propose_candidates, run_rule_loop
from claimcheck.rules import RuleEngine


def _dev():
    df = split_claims(generate_claims(n=1200, fraud_rate=0.05, seed=42))
    return df[df["split"] == "dev"]


def test_propose_candidates_are_rules():
    engine = RuleEngine.from_file()
    cands = propose_candidates(_dev(), engine.rules)
    assert len(cands) > 5
    for c in cands:
        assert c.operator in (">", "<")
        assert c.points > 0


def test_rule_loop_never_hurts_dev_f1():
    engine = RuleEngine.from_file()
    dev = _dev()
    before = confusion(engine, dev)["f1"]
    out = run_rule_loop(engine, dev, rounds=3)
    assert out["dev_f1"] >= before - 1e-9
    assert len(out["engine"].rules) >= len(engine.rules)
    assert isinstance(out["log"], list) and len(out["log"]) > 0


def test_rule_loop_log_records_kept_flags():
    engine = RuleEngine.from_file()
    out = run_rule_loop(engine, _dev(), rounds=2)
    kept = [e for e in out["log"] if e.get("kept")]
    assert len(kept) == len([r for r in out["engine"].rules if r.rule_id.startswith("L")])
