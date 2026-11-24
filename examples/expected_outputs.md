# Expected outputs for the example claims

Run `python scripts/demo.py` to reproduce these.

## CLM-EX-01 — ordinary claim

- Score: 0, risk band low, flagged: no.
- No rules hit. Both reviewers abstain (no evidence to judge on);
  panel consensus is `abstain`.

## CLM-EX-02 — suspicious claim

- Score: 125 or higher, risk band high, flagged: yes.
- Hits include R01 (claim soon after policy start), R02 (inflated repair),
  R03 (very large amount), R04 (claimant velocity), R05 (shop velocity),
  R07 (many injuries), R08 (repeat history), R09 (high repair estimate),
  and R10 (no police report).
- Panel consensus is `escalate`.

## CLM-EX-03 — duplicate claim

- Score: 30, risk band medium, flagged: no (below the 40-point flag line).
- Only R06 (duplicate claim, +30) hits.
- Panel consensus is `review`: the duplicate alone is worth a human look,
  but one signal is not enough to escalate on its own.
