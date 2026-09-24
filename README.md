# Poker44 SN126 conservative ML miner

Public model repository for Bittensor **Poker44 subnet 126** (netuid 126).
It publishes the trained bot-detection model, the exact inference code the
miner runs, and the training script that produced the artifact, so validators
can verify the `model_manifest` a miner advertises against open source.

> **Which branch to use:** `master` (the default branch) is the published,
> canonical line described by this README. The `main` branch holds a separate
> line of work with no shared history; nothing here refers to it. Pin the
> `master` commit you deploy (see [Runtime](#runtime-on-your-miner-vps)).

## What is in this repository

| Path | Purpose |
|------|---------|
| `neurons/miner.py` | Miner neuron. Delegates scoring to `MinerRuntime`; also holds the reference heuristic (`Miner.score_chunk`) used as fallback. |
| `scripts/local/miner_runtime.py` | Env-driven runtime: loads the ML scorer, applies optional calibration, builds the `model_manifest`. |
| `scripts/local/ml_inference.py` | `MLScorer`: loads the pickle, vectorises chunks, applies the score transform. |
| `scripts/local/features.py` | 225 chunk-level behavioural features computed from miner-visible hands. |
| `scripts/local/scoring_policy.py` | Optional low-confidence blending toward the heuristic (off by default). |
| `scripts/local/train_conservative.py` | Training script with date holdout and holdout-tuned score transform. |
| `scripts/local/artifacts/conservative/best_model.pkl` | Pickled bundle: `{"model", "feature_names", "model_type", "granularity", "hand_aggregation"}`. |
| `scripts/local/artifacts/conservative/best_model_metadata.json` | Feature list, training/holdout dates, holdout metrics recorded at training time. |
| `scripts/local/artifacts/conservative/best_score_transform.json` | The transform the runtime applies to raw probabilities (this is what production uses). |
| `requirements-inference.txt` | `numpy>=1.24`, `scikit-learn>=1.3` (the only extra packages inference needs). |
| `MODEL_CARD.md` | One-table model card. |

The runtime modules import `bittensor` and the `poker44.*` package from
[Poker44/Poker44-subnet](https://github.com/Poker44/Poker44-subnet); they are
meant to be dropped into a clone of that repository, not run standalone.

## Model

| Field | Value |
|-------|-------|
| Name | `poker44-conservative-ml` |
| Task | Bot detection: one risk score in `[0, 1]` per chunk of 40 hands |
| Estimator | `sklearn.ensemble.ExtraTreesClassifier` (800 trees, `max_depth=18`, `min_samples_leaf=2`, `class_weight={0: 0.85, 1: 1.45}`) |
| Granularity | chunk (no per-hand aggregation) |
| Features | 225 chunk-level aggregates (see `feature_names` in the metadata JSON) |
| Training data | Public Poker44 benchmark API only (`https://api.poker44.net/api/v1/benchmark`), dates 2026-05-01, 05-02, 05-03, 05-05, 05-06, 05-07 |
| Holdout dates (selection only) | 2026-04-30, 2026-05-04, 2026-05-08 |
| Tuning objective | `robust` (worst-date / worst-batch stability), `--recall-focus` sample weights |
| Private validator data | Not used for training; inference never reads labels |
| Score transform | `conservative_sigmoid` with `a=3.5`, `b=0.14`, `offset=+0.01` |

### Holdout metrics (recorded at training time)

From `best_model_metadata.json`, grouped by 40-hand release chunk on the three
holdout dates (35 groups):

| Metric | Value |
|--------|-------|
| reward_mean | 0.9962 |
| reward_min | 0.9025 |
| fpr_max | 0.05 |
| bot_recall_mean | 0.9971 |
| ap_mean | 1.0 |
| per-date reward | 2026-04-30: 1.0000, 2026-05-04: 0.9919, 2026-05-08: 0.9968 |

These numbers were measured with the transform as originally tuned
(`offset=+0.04`). The offset was later lowered (see changelog) and the
metadata file was left as the training-time record, so
`best_model_metadata.json` still shows `0.04` while
`best_score_transform.json` (the file the runtime actually reads) has `0.01`.

### Changelog

- **Fix 1** (commit `4ad3e12`): `conservative_sigmoid` offset `+0.04 -> +0.01`.
  Removes the false-positive-rate tail that appeared under date shift in a
  leave-one-date-out check across nine dates. The diagnosis report and the
  production backup file referenced in the transform's `note` field are not
  part of this repository.
- Initial publish (commit `2a7f032`): model, metadata, transform and runtime code.

## How scoring works

For every chunk the validator sends (`DetectionSynapse.chunks`):

1. `features.extract_chunk_features` builds the 225-feature vector from the
   miner-visible hands.
2. `ExtraTreesClassifier.predict_proba` gives the raw bot probability.
3. `scoring_policy.apply_confidence_fallback` is a no-op unless
   `POKER44_CONFIDENCE_FALLBACK=1`.
4. `MLScorer.apply_transform` applies `sigmoid(a * (p - b)) + offset`, then
   clamps to `[0, 1]`.
5. `POKER44_CALIBRATION_MODE` (default `identity`) is applied and the result
   is clamped again.
6. `risk_scores` are returned; `predictions` are `score >= 0.5`.

If the model fails to load or scoring raises, the runtime logs a warning and
falls back to the reference heuristic in `neurons/miner.py`.

## Runtime (on your miner VPS)

1. Clone [Poker44/Poker44-subnet](https://github.com/Poker44/Poker44-subnet)
   and install its dependencies, then `pip install -r requirements-inference.txt`.
2. Copy `neurons/miner.py` and the whole `scripts/local/` tree from this
   repository into the subnet clone, at the same paths. The manifest hashes
   `neurons/miner.py`, `scripts/local/miner_runtime.py`, `features.py`,
   `ml_inference.py` and `train_conservative.py`, so the copies must be
   byte-identical to the commit you pin.
3. Export the runtime configuration (replace the commit with
   `git rev-parse HEAD` from this repository's `master`):

```bash
export POKER44_MODEL_REPO_URL=https://github.com/kyan9400/poker44-conservative-ml
export POKER44_MODEL_REPO_COMMIT=<master-commit-you-deployed>
export POKER44_RISK_MODE=conservative
export POKER44_MODEL_MODE=ml
export POKER44_CALIBRATION_MODE=identity
export POKER44_CONFIDENCE_FALLBACK=0
export POKER44_ARTIFACTS_DIR=$(pwd)/scripts/local/artifacts/conservative
```

4. Start the miner as documented by the subnet repository
   (`python neurons/miner.py ...`). Startup logs print the resolved mode,
   artifacts directory, manifest compliance status and digest.

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `POKER44_MODEL_MODE` | `heuristic` | `heuristic`, `ml`, or `blend` (ML blended with the heuristic by `POKER44_BLEND_ALPHA`). Use `ml`. |
| `POKER44_RISK_MODE` | `conservative` | Selects `scripts/local/artifacts/<mode>/` when `POKER44_ARTIFACTS_DIR` is unset. Only `conservative` is published here. |
| `POKER44_ARTIFACTS_DIR` | derived | Explicit directory containing `best_model.pkl`, `best_model_metadata.json`, `best_score_transform.json`. |
| `POKER44_CALIBRATION_MODE` | `identity` | Post-transform calibration: `identity`, `rank`, `linear:k=..`, `power:gamma=..`, `sigmoid:a=..,b=..`. Keep `identity`. |
| `POKER44_BLEND_ALPHA` | `0.5` | ML weight in `blend` mode. |
| `POKER44_CONFIDENCE_FALLBACK` | `0` | Enable low-confidence blending toward the heuristic. Off by default because it can push borderline humans above 0.5. |
| `POKER44_CONFIDENCE_MARGIN` / `_BLEND_HEUR` / `_DISAGREE` | `0.12` / `0.55` / `0.25` | Fallback tuning; only read when the fallback is enabled. |
| `POKER44_MODEL_REPO_URL` | git remote `origin` | Repository URL written into the manifest. `local://` is rejected unless `POKER44_ALLOW_LOCAL_MANIFEST=1`. |
| `POKER44_MODEL_REPO_COMMIT` | `git rev-parse HEAD` | Commit written into the manifest. Set it explicitly on the VPS. |
| `POKER44_MODEL_NAME` | `poker44-conservative-ml` | Manifest model name. |
| `POKER44_PREDICTION_LOG` | unset | Optional path; when set, one JSON line per scored batch is appended there. |

## Retraining

`scripts/local/train_conservative.py` reproduces the artifact set. It needs the
feature CSVs (`scripts/local/artifacts/poker44_training_rows.csv` and
optionally `poker44_hand_training_rows.csv`), which are **not** included, and
the `poker44.score.scoring.reward` function from the subnet repository for
holdout scoring.

```bash
python scripts/local/train_conservative.py --objective robust --recall-focus
```

It trains logistic, HistGradientBoosting and ExtraTrees candidates on the
non-holdout dates, searches `raw_probability` / `conservative_sigmoid` / `blend`
transforms on the holdout groups, rejects anything with `fpr_max > 0.10`, and
writes the winner to `scripts/local/artifacts/conservative/` (plus a copy one
level up, which is git-ignored).

## Verifying the artifact

```
sha256  3e230c3afdb8348cb1b49706c5e964646d8b8dbf3e987efa90fe4555da649556  scripts/local/artifacts/conservative/best_model.pkl
```

The pickle loads with `numpy>=1.24` and `scikit-learn>=1.3` (checked with
numpy 2.2 / scikit-learn 1.7).

## License

MIT, see [LICENSE](LICENSE).
