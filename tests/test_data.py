import pandas as pd

from claimcheck.data_gen import generate_claims, split_claims
from claimcheck.features import add_features, claim_to_features
from claimcheck.models import Claim


def test_generate_is_reproducible():
    a = generate_claims(n=500, seed=42)
    b = generate_claims(n=500, seed=42)
    assert a.equals(b)
    c = generate_claims(n=500, seed=7)
    assert not a.equals(c)


def test_generate_fraud_rate_and_columns():
    df = generate_claims(n=1000, fraud_rate=0.05, seed=42)
    assert len(df) == 1000
    assert abs(df["is_fraud"].mean() - 0.05) < 0.01
    assert set(df["typology"].unique()) <= {
        "legitimate", "inflated_repair", "early_claim", "velocity", "duplicate", "staged"}
    assert df["is_fraud"].sum() == (df["typology"] != "legitimate").sum()


def test_split_is_stratified_60_20_20():
    df = split_claims(generate_claims(n=1000, seed=42))
    counts = df["split"].value_counts(normalize=True)
    assert abs(counts["train"] - 0.6) < 0.03
    assert abs(counts["dev"] - 0.2) < 0.03
    assert abs(counts["test"] - 0.2) < 0.03
    # Fraud present in every split.
    assert df.groupby("split")["is_fraud"].sum().min() > 0


def test_repair_ratio_feature():
    claim = Claim(repair_estimate=10000, vehicle_value=20000)
    assert claim.repair_ratio == 0.5
    feats = claim_to_features(claim)
    assert feats["repair_ratio"] == 0.5
    df = add_features(pd.DataFrame([{"repair_estimate": 5000, "vehicle_value": 10000}]))
    assert df.iloc[0]["repair_ratio"] == 0.5
