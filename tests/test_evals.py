from claimcheck.data_gen import generate_claims, split_claims
from claimcheck.evals import confusion, evaluate_at_prevalences, sample_at_prevalence
from claimcheck.rules import RuleEngine


def _df():
    return split_claims(generate_claims(n=1000, fraud_rate=0.05, seed=42))


def test_confusion_counts_add_up():
    engine = RuleEngine.from_file()
    df = _df()
    test = df[df["split"] == "test"]
    m = confusion(engine, test)
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == len(test)
    for k in ("precision", "recall", "f1", "false_positive_rate", "false_negative_rate"):
        assert 0.0 <= m[k] <= 1.0


def test_precision_collapses_at_low_prevalence():
    engine = RuleEngine.from_file()
    df = _df()
    test = df[df["split"] == "test"]
    out = evaluate_at_prevalences(engine, test)
    assert set(out) == {"1%", "5%"}
    # Same rules, rarer fraud: precision at 1% should be no better than at 5%.
    assert out["1%"]["precision"] <= out["5%"]["precision"] + 1e-9
    assert out["1%"]["false_positive_rate"] <= 0.2


def test_sample_at_prevalence():
    df = _df()
    s = sample_at_prevalence(df, 0.01, seed=42)
    assert abs(s["is_fraud"].mean() - 0.01) < 0.01
    s5 = sample_at_prevalence(df, 0.05, seed=42)
    assert abs(s5["is_fraud"].mean() - 0.05) < 0.01


def test_engine_beats_chance_on_synthetic_data():
    engine = RuleEngine.from_file()
    df = _df()
    test = df[df["split"] == "test"]
    m = confusion(engine, test)
    assert m["recall"] > 0.5
    assert m["precision"] > 0.5
