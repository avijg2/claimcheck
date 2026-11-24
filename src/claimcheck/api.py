"""FastAPI service for ClaimCheck."""

from __future__ import annotations

from fastapi import FastAPI

from .data_gen import generate_claims, split_claims
from .evals import evaluate_at_prevalences
from .models import Claim, PanelResult, ScoreResult
from .reviewers import run_panel
from .rules import RuleEngine

app = FastAPI(title="ClaimCheck", version="0.1.0")
engine = RuleEngine.from_file()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "claimcheck", "rules": len(engine.rules)}


@app.post("/score", response_model=ScoreResult)
def score(claim: Claim) -> ScoreResult:
    return engine.score_claim(claim)


@app.post("/review", response_model=PanelResult)
def review(claim: Claim, use_openai: bool = False) -> PanelResult:
    return run_panel(claim, engine, use_openai=use_openai)


@app.get("/metrics")
def metrics() -> dict:
    """Eval summary on a fresh synthetic test split (seed 42)."""
    df = split_claims(generate_claims(n=2000, fraud_rate=0.05, seed=42))
    test = df[df["split"] == "test"]
    return {
        "n_test": int(len(test)),
        "fraud_in_test": int(test["is_fraud"].sum()),
        "eval": evaluate_at_prevalences(engine, test),
        "note": "Synthetic data only (seed 42). Metrics are for the rule lab, not a production system.",
    }
