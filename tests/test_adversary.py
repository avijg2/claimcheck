from claimcheck.adversary import evasion_rate, make_adversarial
from claimcheck.data_gen import generate_claims, split_claims
from claimcheck.rules import RuleEngine


def _fraud():
    df = split_claims(generate_claims(n=1000, fraud_rate=0.05, seed=42))
    return df[df["is_fraud"]]


def test_adversarial_nudges_signals_down():
    engine = RuleEngine.from_file()
    fraud = _fraud()
    adv = make_adversarial(fraud, engine)
    assert len(adv) == len(fraud)
    # Repair ratios should not increase after doctoring.
    from claimcheck.features import add_features

    fraud_feats = add_features(fraud)
    assert adv["repair_ratio"].max() <= fraud_feats["repair_ratio"].max() + 1e-9
    assert adv["claimant_claims_12m"].max() <= fraud["claimant_claims_12m"].max()


def test_evasion_rate_is_between_zero_and_one():
    engine = RuleEngine.from_file()
    fraud = _fraud()
    rate = evasion_rate(engine, fraud)
    assert 0.0 <= rate <= 1.0
    adv_rate = evasion_rate(engine, make_adversarial(fraud, engine))
    # Doctoring signals should make evasion worse (or at worst equal).
    assert adv_rate >= rate - 1e-9


def test_adversarial_does_not_create_new_fraud_labels():
    engine = RuleEngine.from_file()
    fraud = _fraud()
    adv = make_adversarial(fraud, engine)
    assert adv["is_fraud"].all()
