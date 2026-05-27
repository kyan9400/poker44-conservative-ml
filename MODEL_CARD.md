# Model card — poker44-conservative-ml

| Field | Value |
|-------|--------|
| Task | Bot detection on poker hand chunks |
| Output | Risk score in [0, 1] per chunk |
| Framework | scikit-learn ExtraTrees |
| Features | 225 chunk-level behavioral aggregates |
| Transform | conservative_sigmoid (holdout-tuned) |
| Private validator data | Not used for training |
