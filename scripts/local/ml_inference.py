#!/usr/bin/env python3
"""Load trained ML artifacts and score miner-visible chunks."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scripts.local.features import (
    extract_hand_features,
    feature_vector,
    prepare_chunk_from_raw,
)
from scripts.local.scoring_policy import apply_confidence_fallback

HAND_AGG_FUNCS = {
    "mean": lambda p: float(np.mean(p)),
    "max": lambda p: float(np.max(p)),
    "p75": lambda p: float(np.percentile(p, 75)),
    "p90": lambda p: float(np.percentile(p, 90)),
    "mix_70max_30mean": lambda p: 0.7 * float(np.max(p)) + 0.3 * float(np.mean(p)),
}

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "scripts" / "local" / "artifacts"


def _artifact_dir() -> Path:
    import os

    explicit = os.getenv("POKER44_ARTIFACTS_DIR", "").strip()
    if explicit:
        return Path(explicit)
    mode = os.getenv("POKER44_RISK_MODE", "conservative").strip().lower()
    if mode == "aggressive":
        return ARTIFACTS / "aggressive_max"
    return ARTIFACTS / "conservative" if (ARTIFACTS / "conservative" / "best_model.pkl").exists() else ARTIFACTS


class MLScorer:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        transform_path: Optional[Path] = None,
    ) -> None:
        base = _artifact_dir()
        self.model_path = Path(model_path or base / "best_model.pkl")
        self.metadata_path = Path(metadata_path or base / "best_model_metadata.json")
        self.transform_path = Path(transform_path or base / "best_score_transform.json")
        self._bundle: Optional[dict] = None
        self._metadata: Optional[dict] = None
        self._transform: Optional[dict] = None
        self._isotonic = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model: {self.model_path}")
        with self.model_path.open("rb") as f:
            self._bundle = pickle.load(f)
        if self.metadata_path.exists():
            self._metadata = json.loads(self.metadata_path.read_text())
        if self.transform_path.exists():
            self._transform = json.loads(self.transform_path.read_text())

    @property
    def feature_names(self) -> List[str]:
        if not self._bundle:
            raise RuntimeError("Model not loaded")
        return list(self._bundle["feature_names"])

    @property
    def model_type(self) -> str:
        if not self._bundle:
            return "unknown"
        return str(self._bundle.get("model_type", "unknown"))

    @property
    def granularity(self) -> str:
        if not self._bundle:
            return "chunk"
        return str(self._bundle.get("granularity", "chunk"))

    @property
    def hand_aggregation(self) -> str:
        if not self._bundle:
            return "mean"
        agg = self._bundle.get("hand_aggregation")
        if agg:
            return str(agg)
        if self._metadata:
            return str(self._metadata.get("hand_aggregation") or "mean")
        return "mean"

    def _vectorize(self, chunk: List[dict]) -> np.ndarray:
        feats = feature_vector(chunk, raw=False, feature_names=self.feature_names)
        return np.array([[float(feats.get(n, 0.0)) for n in self.feature_names]], dtype=np.float64)

    def _vectorize_hand(self, hand: dict) -> np.ndarray:
        feats = extract_hand_features(hand)
        return np.array(
            [[float(feats.get(n, 0.0)) for n in self.feature_names]], dtype=np.float64
        )

    def _predict_proba_vector(self, X: np.ndarray) -> float:
        if not self._bundle:
            self.load()
        bundle = self._bundle
        if bundle.get("ensemble_models"):
            probs = []
            for model in bundle["ensemble_models"]:
                if hasattr(model, "predict_proba"):
                    probs.append(float(model.predict_proba(X)[0, 1]))
                else:
                    probs.append(float(model.predict(X)[0]))
            return float(np.mean(probs))
        model = bundle["model"]
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(X)[0, 1])
        return float(model.predict(X)[0])

    def predict_proba_chunk(self, chunk: List[dict]) -> float:
        if not self._bundle:
            self.load()
        if self.granularity == "hand":
            model = self._bundle["model"]
            hands = [h for h in chunk if isinstance(h, dict)]
            if not hands:
                return 0.5
            probs: List[float] = []
            for hand in hands:
                X = self._vectorize_hand(hand)
                if hasattr(model, "predict_proba"):
                    probs.append(float(model.predict_proba(X)[0, 1]))
                else:
                    probs.append(float(model.predict(X)[0]))
            agg_fn = HAND_AGG_FUNCS.get(self.hand_aggregation, HAND_AGG_FUNCS["mean"])
            return float(agg_fn(np.asarray(probs, dtype=float)))
        X = self._vectorize(chunk)
        return self._predict_proba_vector(X)

    def apply_transform(
        self,
        scores: List[float],
        heuristic_scores: List[float],
        *,
        transform_override: Optional[dict] = None,
    ) -> List[float]:
        spec = transform_override if transform_override is not None else self._transform
        arr = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
        heur = np.clip(np.asarray(heuristic_scores, dtype=float), 0.0, 1.0)

        if not spec:
            return [max(0.0, min(1.0, float(x))) for x in arr]

        name = spec.get("transform", "raw_probability")
        params = spec.get("params") or {}

        if name in {"raw_probability", "identity"}:
            pass
        elif name == "clipped_probability":
            lo = float(params.get("low", 0.0))
            hi = float(params.get("high", 0.85))
            arr = np.clip(arr, lo, hi)
        elif name == "conservative_sigmoid":
            a = float(params.get("a", 6.0))
            b = float(params.get("b", 0.35))
            arr = 1.0 / (1.0 + np.exp(-a * (arr - b)))
        elif name in {"blend_ml_heuristic", "blend"} or name.startswith("blend_alpha"):
            alpha = float(params.get("alpha", 0.8))
            arr = alpha * arr + (1.0 - alpha) * heur
        elif name.startswith("scale_"):
            arr = arr * float(params.get("scale", 1.0))
        elif name.startswith("offset_"):
            arr = arr + float(params.get("offset", 0.0))
        elif name in {"rank_within_batch_40", "rank"}:
            # Deprecated for mainnet — kept for diagnostics only.
            if len(arr) > 1:
                order = np.argsort(np.argsort(arr))
                arr = order.astype(float) / (len(arr) - 1)

        if "offset" in params and name not in {"rank_within_batch_40", "rank"}:
            if name not in {"offset_"}:  # offset_* already applied above
                arr = arr + float(params["offset"])

        return [max(0.0, min(1.0, float(x))) for x in arr]

    def score_chunks(
        self,
        chunks: List[List[dict]],
        *,
        mode: str = "ml",
        blend_alpha: float = 0.5,
    ) -> Tuple[List[float], List[float], List[float]]:
        """Return (final_scores, ml_probs, heuristic_scores)."""
        from neurons.miner import Miner  # lazy: avoids import cycle with miner_runtime

        ml_probs: List[float] = []
        heur: List[float] = []
        for chunk in chunks:
            visible = chunk
            if chunk and isinstance(chunk[0], dict) and "label" in chunk[0]:
                visible = prepare_chunk_from_raw(chunk)
            heur.append(Miner.score_chunk(visible))
            try:
                ml_probs.append(self.predict_proba_chunk(visible))
            except Exception:
                ml_probs.append(heur[-1])

        if mode == "heuristic":
            final = heur
        elif mode == "blend":
            final = [
                max(0.0, min(1.0, blend_alpha * m + (1 - blend_alpha) * h))
                for m, h in zip(ml_probs, heur)
            ]
        else:
            final = list(ml_probs)

        final, _n_adj = apply_confidence_fallback(final, heur)
        final = self.apply_transform(final, heur)
        return final, ml_probs, heur


_SCORER: Optional[MLScorer] = None


def get_scorer() -> MLScorer:
    global _SCORER
    if _SCORER is None:
        _SCORER = MLScorer()
        _SCORER.load()
    return _SCORER
