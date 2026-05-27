#!/usr/bin/env python3
"""Feature extraction for Poker44 inner chunks (miner-visible hands)."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

from poker44.validator.payload_view import prepare_hand_for_miner

ACTION_TYPES = ("fold", "check", "call", "bet", "raise", "small_blind", "big_blind", "ante")
STREETS = ("preflop", "flop", "turn", "river")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent


def extract_hand_features(hand: Mapping[str, Any]) -> Dict[str, float]:
    """Numeric features for one miner-visible hand dict."""
    actions = hand.get("actions") if isinstance(hand.get("actions"), list) else []
    players = hand.get("players") if isinstance(hand.get("players"), list) else []
    streets = hand.get("streets") if isinstance(hand.get("streets"), list) else []
    metadata = hand.get("metadata") if isinstance(hand.get("metadata"), dict) else {}

    action_counts = Counter()
    amounts: List[float] = []
    pot_ratios: List[float] = []
    streets_seen: List[str] = []
    seats: List[int] = []

    for action in actions:
        if not isinstance(action, dict):
            continue
        atype = str(action.get("action_type") or "").lower()
        action_counts[atype] += 1
        amt = _safe_float(action.get("normalized_amount_bb"))
        if amt > 0:
            amounts.append(amt)
        pot_before = _safe_float(action.get("pot_before"))
        pot_after = _safe_float(action.get("pot_after"))
        if pot_before > 0:
            pot_ratios.append((pot_after - pot_before) / pot_before)
        street = str(action.get("street") or "").lower()
        if street:
            streets_seen.append(street)
        seat = int(_safe_float(action.get("actor_seat")))
        if seat > 0:
            seats.append(seat)

    meaningful = sum(action_counts.get(k, 0) for k in ("call", "check", "bet", "raise", "fold"))
    meaningful = max(meaningful, 1)

    preflop = sum(1 for s in streets_seen if s == "preflop")
    postflop = sum(1 for s in streets_seen if s in ("flop", "turn", "river"))

    stacks = [_safe_float(p.get("starting_stack")) for p in players if isinstance(p, dict)]
    max_seats = _safe_float(metadata.get("max_seats"), 6.0)
    outcome = hand.get("outcome") if isinstance(hand.get("outcome"), dict) else {}
    showdown = 1.0 if outcome.get("showdown") else 0.0
    all_in_actions = sum(
        1 for a in actions if isinstance(a, dict) and bool(a.get("is_all_in"))
    )
    blind_actions = sum(
        action_counts.get(k, 0) for k in ("small_blind", "big_blind", "ante")
    )
    streets_with_action = len(set(streets_seen))
    action_per_street = len(actions) / max(streets_with_action, 1)

    return {
        "hand_action_count": float(len(actions)),
        "hand_meaningful_actions": float(meaningful),
        "hand_street_count": float(len(streets)),
        "hand_player_count": float(len(players)),
        "hand_max_seats": max_seats,
        "hand_call_ratio": action_counts.get("call", 0) / meaningful,
        "hand_check_ratio": action_counts.get("check", 0) / meaningful,
        "hand_fold_ratio": action_counts.get("fold", 0) / meaningful,
        "hand_raise_ratio": action_counts.get("raise", 0) / meaningful,
        "hand_bet_ratio": action_counts.get("bet", 0) / meaningful,
        "hand_aggression_ratio": (action_counts.get("bet", 0) + action_counts.get("raise", 0)) / meaningful,
        "hand_passive_ratio": (action_counts.get("call", 0) + action_counts.get("check", 0)) / meaningful,
        "hand_preflop_action_ratio": preflop / max(len(actions), 1),
        "hand_postflop_action_ratio": postflop / max(len(actions), 1),
        "hand_action_entropy": _entropy(action_counts),
        "hand_unique_seats": float(len(set(seats))),
        "hand_amount_mean": sum(amounts) / len(amounts) if amounts else 0.0,
        "hand_amount_std": (
            (sum((a - sum(amounts) / len(amounts)) ** 2 for a in amounts) / len(amounts)) ** 0.5
            if len(amounts) > 1
            else 0.0
        ),
        "hand_amount_max": max(amounts) if amounts else 0.0,
        "hand_amount_min": min(amounts) if amounts else 0.0,
        "hand_pot_ratio_mean": sum(pot_ratios) / len(pot_ratios) if pot_ratios else 0.0,
        "hand_stack_mean": sum(stacks) / len(stacks) if stacks else 0.0,
        "hand_stack_std": (
            (sum((s - sum(stacks) / len(stacks)) ** 2 for s in stacks) / len(stacks)) ** 0.5
            if len(stacks) > 1
            else 0.0
        ),
        "hand_showdown": showdown,
        "hand_all_in_ratio": all_in_actions / max(len(actions), 1),
        "hand_blind_ratio": blind_actions / max(len(actions), 1),
        "hand_streets_with_action": float(streets_with_action),
        "hand_action_per_street": action_per_street,
        "hand_passive_aggressive_gap": abs(
            (action_counts.get("call", 0) + action_counts.get("check", 0))
            - (action_counts.get("bet", 0) + action_counts.get("raise", 0))
        ) / meaningful,
    }


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _aggregate(values: Sequence[float], prefix: str) -> Dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_p75": 0.0,
            f"{prefix}_range": 0.0,
        }
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    sorted_vals = sorted(values)
    med = _percentile(sorted_vals, 0.5)
    p75 = _percentile(sorted_vals, 0.75)
    vmin, vmax = sorted_vals[0], sorted_vals[-1]
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": var**0.5,
        f"{prefix}_min": vmin,
        f"{prefix}_max": vmax,
        f"{prefix}_median": med,
        f"{prefix}_p75": p75,
        f"{prefix}_range": vmax - vmin,
    }


def extract_chunk_features(miner_visible_chunk: List[dict]) -> Dict[str, float]:
    """Aggregate hand features into one feature dict per inner chunk."""
    if not miner_visible_chunk:
        return {"chunk_hand_count": 0.0}

    hand_feats = [extract_hand_features(h) for h in miner_visible_chunk if isinstance(h, dict)]
    if not hand_feats:
        return {"chunk_hand_count": 0.0}

    out: Dict[str, float] = {"chunk_hand_count": float(len(hand_feats))}
    keys = [k for k in hand_feats[0] if k.startswith("hand_")]
    for key in keys:
        vals = [hf[key] for hf in hand_feats]
        agg = _aggregate(vals, key.replace("hand_", "chunk_"))
        out.update(agg)

    # Chunk-level behavioural diversity
    action_entropies = [hf["hand_action_entropy"] for hf in hand_feats]
    aggression = [hf["hand_aggression_ratio"] for hf in hand_feats]
    out.update(_aggregate(action_entropies, "chunk_entropy"))
    out.update(_aggregate(aggression, "chunk_aggression"))

    # Line diversity: fraction of unique (call,check,raise,fold) ratio quadruples
    signatures = []
    for hf in hand_feats:
        signatures.append(
            (
                round(hf["hand_call_ratio"], 2),
                round(hf["hand_check_ratio"], 2),
                round(hf["hand_raise_ratio"], 2),
                round(hf["hand_fold_ratio"], 2),
            )
        )
    out["chunk_line_diversity"] = len(set(signatures)) / max(len(signatures), 1)

    # Cross-hand contrast signals (help bot vs human separation in a batch)
    if aggression:
        out["chunk_aggression_spread"] = max(aggression) - min(aggression)
        out["chunk_aggression_p90"] = _percentile(sorted(aggression), 0.9)
    fold_ratios = [hf["hand_fold_ratio"] for hf in hand_feats]
    if fold_ratios:
        out["chunk_fold_ratio_spread"] = max(fold_ratios) - min(fold_ratios)
        out["chunk_high_fold_hand_frac"] = sum(1 for v in fold_ratios if v >= 0.45) / len(
            fold_ratios
        )
    raise_ratios = [hf["hand_raise_ratio"] for hf in hand_feats]
    if raise_ratios:
        out["chunk_high_raise_hand_frac"] = sum(1 for v in raise_ratios if v >= 0.15) / len(
            raise_ratios
        )
    showdowns = [hf.get("hand_showdown", 0.0) for hf in hand_feats]
    if showdowns:
        out["chunk_showdown_frac"] = sum(showdowns) / len(showdowns)

    return out


def prepare_chunk_from_raw(raw_hands: List[dict]) -> List[dict]:
    """Project raw benchmark hands to miner-visible form."""
    visible: List[dict] = []
    for hand in raw_hands:
        if not isinstance(hand, dict):
            continue
        try:
            visible.append(prepare_hand_for_miner(hand))
        except Exception:
            continue
    return visible


def feature_vector(
    chunk: List[dict],
    *,
    raw: bool = False,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Extract features; set raw=True to run prepare_hand_for_miner first."""
    visible = prepare_chunk_from_raw(chunk) if raw else chunk
    feats = extract_chunk_features(visible)
    if feature_names is not None:
        return {name: float(feats.get(name, 0.0)) for name in feature_names}
    return feats


def default_hand_feature_names(sample_hand: Optional[dict] = None) -> List[str]:
    if sample_hand is None:
        sample_hand = {
            "metadata": {"max_seats": 6},
            "players": [{"seat": 1, "starting_stack": 100.0}],
            "streets": [{"street": "preflop"}],
            "actions": [
                {
                    "action_type": "call",
                    "street": "preflop",
                    "actor_seat": 1,
                    "normalized_amount_bb": 1.0,
                    "pot_before": 1.0,
                    "pot_after": 2.0,
                }
            ],
            "outcome": {},
        }
    feats = extract_hand_features(sample_hand)
    return sorted(feats.keys())


def default_feature_names(sample_chunk: Optional[List[dict]] = None) -> List[str]:
    """Stable ordered feature name list."""
    if sample_chunk is None:
        sample_chunk = [
            {
                "metadata": {"max_seats": 6},
                "players": [{"seat": 1, "starting_stack": 100.0}],
                "streets": [{"street": "preflop"}],
                "actions": [
                    {
                        "action_type": "call",
                        "street": "preflop",
                        "actor_seat": 1,
                        "normalized_amount_bb": 1.0,
                        "pot_before": 1.0,
                        "pot_after": 2.0,
                    }
                ],
                "outcome": {},
            }
        ]
    feats = extract_chunk_features(sample_chunk)
    return sorted(feats.keys())


# Columns that must never be model inputs
LEAKAGE_COLUMNS = frozenset(
    {
        "label",
        "label_str",
        "sourceDate",
        "chunkId",
        "release_chunk_id",
        "inner_index",
        "chunkHash",
        "windowStart",
        "windowEnd",
    }
)
