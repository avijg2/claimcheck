from fastapi.testclient import TestClient

from claimcheck.api import app

client = TestClient(app)

ORDINARY = {
    "claim_id": "A1", "days_since_policy_start": 400, "claim_amount": 4200,
    "repair_estimate": 3800, "vehicle_value": 22000, "claimant_claims_12m": 0,
    "shop_claims_30d": 3, "prior_claims": 0, "injuries_reported": 0,
    "police_report_filed": True, "duplicate_claim": False,
}

SUSPICIOUS = dict(ORDINARY, claim_id="A2", days_since_policy_start=12,
                  claim_amount=21000, repair_estimate=19500, vehicle_value=20000,
                  claimant_claims_12m=4, shop_claims_30d=14, prior_claims=3,
                  injuries_reported=4, police_report_filed=False)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["rules"] == 10


def test_score_ordinary_and_suspicious():
    assert client.post("/score", json=ORDINARY).json()["score"] == 0
    resp = client.post("/score", json=SUSPICIOUS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["flagged"] is True
    assert body["risk_band"] == "high"


def test_review_endpoint():
    resp = client.post("/review", json=SUSPICIOUS)
    assert resp.status_code == 200
    assert resp.json()["consensus"] == "escalate"


def test_metrics_endpoint_has_both_prevalences():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["eval"]) == {"1%", "5%"}
    assert body["n_test"] > 0
    assert "Synthetic" in body["note"]


def test_score_rejects_negative_amount():
    bad = dict(ORDINARY, claim_amount=-5)
    assert client.post("/score", json=bad).status_code == 422
