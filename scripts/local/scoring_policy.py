"""Low-confidence blending and optional ensemble helpers for miner scoring."""

from __future__ import annotations

import os
from typing import List, Tuple


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def confidence_fallback_enabled() -> bool:
    # Default off: fallback can push borderline humans above 0.5 (FPR >= 0.10 → zero reward).
    return os.getenv("POKER44_CONFIDENCE_FALLBACK", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def apply_confidence_fallback(
    ml_probs: List[float],
    heur_scores: List[float],
    *,
    margin: float | None = None,
    blend_heuristic: float | None = None,
    disagree_delta: float | None = None,
) -> Tuple[List[float], int]:
    """
    When ML probability is near 0.5 (uncertain), blend toward heuristic.
    When ML and heuristic strongly disagree, favor the more extreme signal slightly.

    Returns (adjusted_scores, n_adjusted).
    """
    if not confidence_fallback_enabled():
        return list(ml_probs), 0

    margin = margin if margin is not None else _env_float("POKER44_CONFIDENCE_MARGIN", 0.12)
    blend_h = blend_heuristic if blend_heuristic is not None else _env_float(
        "POKER44_CONFIDENCE_BLEND_HEUR", 0.55
    )
    disagree_delta = disagree_delta if disagree_delta is not None else _env_float(
        "POKER44_CONFIDENCE_DISAGREE", 0.25
    )

    out: List[float] = []
    adjusted = 0
    for ml, heur in zip(ml_probs, heur_scores):
        score = ml
        uncertain = abs(ml - 0.5) < margin
        disagree = abs(ml - heur) >= disagree_delta
        if uncertain:
            score = blend_h * heur + (1.0 - blend_h) * ml
            # Keep uncertain humans below the 0.5 decision threshold when heuristic is neutral-low.
            if heur <= 0.52:
                score = min(score, 0.48)
            adjusted += 1
        elif disagree:
            # Prefer sharper signal when one side is confident
            if abs(heur - 0.5) > abs(ml - 0.5):
                score = 0.35 * ml + 0.65 * heur
            else:
                score = 0.65 * ml + 0.35 * heur
            adjusted += 1
        out.append(max(0.0, min(1.0, float(score))))
    return out, adjusted
