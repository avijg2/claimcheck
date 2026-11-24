"""Generate synthetic claims and write data/claims.csv with a train/dev/test split.

Usage: python scripts/generate_data.py [--n 5000] [--fraud-rate 0.05] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimcheck.data_gen import generate_claims, split_claims


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--fraud-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = split_claims(generate_claims(n=args.n, fraud_rate=args.fraud_rate, seed=args.seed))
    out = Path(__file__).resolve().parents[1] / "data" / "claims.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} synthetic claims to {out}")
    print(df["split"].value_counts().to_dict())
    print(f"Fraud rate: {df['is_fraud'].mean():.3f}")


if __name__ == "__main__":
    main()
