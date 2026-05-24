# Poker44 SN126 conservative ML miner

Public model repository for Bittensor **Poker44 subnet 126** (netuid 126).

## Model

- **Name:** `poker44-conservative-ml`
- **Type:** ExtraTrees chunk classifier + `conservative_sigmoid` calibration
- **Training data:** Public Poker44 benchmark API only (`https://api.poker44.net/api/v1/benchmark`)
- **Holdout dates (selection only):** 2026-04-30, 2026-05-04, 2026-05-08

## Runtime (on your miner VPS)

Clone [Poker44/Poker44-subnet](https://github.com/Poker44/Poker44-subnet), install deps, copy artifacts from this repo, set:

```bash
export POKER44_MODEL_REPO_URL=https://github.com/kyan9400/poker44-conservative-ml
export POKER44_MODEL_REPO_COMMIT=<this-repo-commit>
export POKER44_RISK_MODE=conservative
export POKER44_MODEL_MODE=ml
export POKER44_CALIBRATION_MODE=identity
export POKER44_CONFIDENCE_FALLBACK=0
export POKER44_ARTIFACTS_DIR=$(pwd)/scripts/local/artifacts/conservative
```

## Files

Implementation files hashed in the miner `model_manifest` match paths under this repository.
