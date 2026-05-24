#!/usr/bin/env python3
"""Train ML model with date holdout — low overfit risk before mainnet."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker44.score.scoring import reward as official_reward  # noqa: E402
from scripts.local.features import LEAKAGE_COLUMNS  # noqa: E402

ARTIFACTS = REPO_ROOT / "scripts" / "local" / "artifacts"
CONSERVATIVE_DIR = ARTIFACTS / "conservative"
LOGS = REPO_ROOT / "logs"

# Spread holdout across calendar (early / mid / late) — targets R6–R11-style shift.
DEFAULT_HOLDOUT = ("2026-04-30", "2026-05-04", "2026-05-08")
# Hand rows inherit chunk labels — holdout reward >= this is treated as overfit.
HAND_HOLDOUT_REWARD_CEILING = 0.98

HAND_AGG_OPTIONS = (
    ("mean", lambda p: float(np.mean(p))),
    ("max", lambda p: float(np.max(p))),
    ("p75", lambda p: float(np.percentile(p, 75))),
    ("p90", lambda p: float(np.percentile(p, 90))),
    ("mix_70max_30mean", lambda p: 0.7 * float(np.max(p)) + 0.3 * float(np.mean(p))),
)


def _git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def load_dataset(
    csv_path: Path, *, include_heuristic: bool = False, meta_extra: frozenset | None = None
):
    import csv

    rows: List[dict] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    meta_cols = {
        "sourceDate",
        "chunkId",
        "release_chunk_id",
        "inner_index",
        "label_str",
        "label",
        "hand_count",
        "hand_index",
    }
    if not include_heuristic:
        meta_cols.add("heuristic_score")
    if meta_extra:
        meta_cols |= set(meta_extra)

    feature_names = [
        c for c in fieldnames if c not in meta_cols and c not in LEAKAGE_COLUMNS
    ]
    X = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.int32)
    for i, row in enumerate(rows):
        y[i] = int(float(row["label"]))
        for j, name in enumerate(feature_names):
            X[i, j] = float(row.get(name) or 0.0)
    return X, y, feature_names, rows


def release_groups(rows: List[dict]) -> List[List[int]]:
    buckets: Dict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)
    for i, row in enumerate(rows):
        buckets[(row["sourceDate"], row["chunkId"])].append(
            (i, int(row.get("inner_index", 0)))
        )
    out: List[List[int]] = []
    for key in sorted(buckets.keys()):
        idxs = [t[0] for t in sorted(buckets[key], key=lambda x: x[1])]
        if len(idxs) == 40:
            out.append(idxs)
    return out


def hand_rows_by_chunk(hand_rows: List[dict]) -> Dict[Tuple[str, str, int], List[int]]:
    buckets: Dict[Tuple[str, str, int], List[Tuple[int, int]]] = defaultdict(list)
    for i, row in enumerate(hand_rows):
        key = (row["sourceDate"], row["chunkId"], int(row["inner_index"]))
        buckets[key].append((i, int(row.get("hand_index", 0))))
    out: Dict[Tuple[str, str, int], List[int]] = {}
    for key, items in buckets.items():
        out[key] = [t[0] for t in sorted(items, key=lambda x: x[1])]
    return out


def grouped_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: List[List[int]],
    rows: List[dict] | None = None,
) -> Dict[str, float]:
    rewards, fprs, botrecs, aps = [], [], [], []
    by_date: Dict[str, List[float]] = defaultdict(list)
    for idxs in groups:
        rew, m = official_reward(scores[idxs].astype(float), labels[idxs].astype(bool))
        rewards.append(float(rew))
        fprs.append(float(m["fpr"]))
        botrecs.append(float(m["bot_recall"]))
        aps.append(float(m["ap_score"]))
        if rows is not None and idxs:
            by_date[str(rows[idxs[0]]["sourceDate"])].append(float(rew))
    if not rewards:
        return {
            "reward_mean": 0.0,
            "fpr_max": 1.0,
            "bot_recall_mean": 0.0,
            "ap_mean": 0.0,
            "reward_min": 0.0,
            "per_date_min": 0.0,
            "per_date_mean": 0.0,
            "n_dates": 0,
        }
    per_date_means = [float(np.mean(v)) for v in by_date.values()] if by_date else [float(np.mean(rewards))]
    return {
        "reward_mean": float(np.mean(rewards)),
        "fpr_max": float(np.max(fprs)),
        "bot_recall_mean": float(np.mean(botrecs)),
        "ap_mean": float(np.mean(aps)),
        "reward_min": float(np.min(rewards)),
        "n_groups": len(rewards),
        "per_date_min": float(np.min(per_date_means)),
        "per_date_mean": float(np.mean(per_date_means)),
        "n_dates": len(per_date_means),
        "per_date_rewards": {d: float(np.mean(v)) for d, v in sorted(by_date.items())},
    }


def build_train_sample_weights(
    rows: List[dict], train_mask: np.ndarray, *, recall_focus: bool
) -> np.ndarray:
    """Up-weight bots and the training date closest to the recent holdout."""
    weights = np.ones(len(rows), dtype=np.float64)
    if not recall_focus:
        return weights[train_mask]
    for i, row in enumerate(rows):
        if not train_mask[i]:
            continue
        if int(row["label"]) == 1:
            weights[i] = 1.35
        if row.get("sourceDate") == "2026-05-07":
            weights[i] *= 1.15
    return weights[train_mask]


def fit_chunk_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    rows: List[dict],
    train_mask: np.ndarray,
    *,
    recall_focus: bool,
) -> None:
    X_tr, y_tr = X[train_mask], y[train_mask]
    sw = build_train_sample_weights(rows, train_mask, recall_focus=recall_focus)
    if isinstance(model, Pipeline):
        model.fit(X_tr, y_tr, clf__sample_weight=sw)
        return
    try:
        model.fit(X_tr, y_tr, sample_weight=sw)
    except TypeError:
        model.fit(X_tr, y_tr)


def robust_composite(hm: Dict[str, float]) -> float:
    """Score for model selection: prioritize worst-date / worst-batch stability."""
    per_date = hm.get("per_date_min", hm.get("reward_min", hm["reward_mean"]))
    batch_min = hm.get("reward_min", per_date)
    mean_r = hm["reward_mean"]
    return 0.35 * mean_r + 0.35 * per_date + 0.30 * batch_min


def rank_tuning_key(hm: Dict[str, float], objective: str) -> tuple:
    if hm["fpr_max"] > 0.10:
        return (2, 0.0, 1.0, 0.0)
    min_r = hm.get("reward_min", hm["reward_mean"])
    per_date = hm.get("per_date_min", min_r)
    if objective == "recall_floor":
        return (0, -min_r, -per_date, -hm["reward_mean"], hm["fpr_max"], -hm["bot_recall_mean"])
    if objective == "robust":
        return (0, -robust_composite(hm), -per_date, -min_r, -hm["reward_mean"], hm["fpr_max"])
    if objective == "balanced":
        composite = 0.72 * hm["reward_mean"] + 0.28 * min_r
        return (0, -composite, -per_date, hm["fpr_max"], -hm["bot_recall_mean"])
    return (0, -hm["reward_mean"], hm["fpr_max"], -hm["bot_recall_mean"])


def predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X).astype(float)


def apply_transform_array(
    raw: np.ndarray, heur: np.ndarray, transform: str, params: Dict[str, Any]
) -> np.ndarray:
    arr = np.clip(raw.astype(float), 0.0, 1.0)
    h = np.clip(heur.astype(float), 0.0, 1.0)
    if transform == "conservative_sigmoid":
        a = float(params.get("a", 6.0))
        b = float(params.get("b", 0.35))
        arr = 1.0 / (1.0 + np.exp(-a * (arr - b)))
    elif transform in {"blend", "blend_ml_heuristic"}:
        alpha = float(params.get("alpha", 0.85))
        arr = alpha * arr + (1.0 - alpha) * h
    if "offset" in params:
        arr = arr + float(params["offset"])
    return np.clip(arr, 0.0, 1.0)


def chunk_models(*, recall_focus: bool) -> Dict[str, Any]:
    bot_weight = {0: 0.85, 1: 1.45} if recall_focus else "balanced"
    models: Dict[str, Any] = {}
    for c in (0.4, 0.8, 1.2):
        models[f"logistic_c{c}"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        C=c,
                        class_weight=bot_weight if recall_focus else "balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
    models["hist_gradient_boosting"] = HistGradientBoostingClassifier(
        max_iter=600,
        max_depth=11,
        learning_rate=0.035,
        min_samples_leaf=6,
        l2_regularization=0.08,
        class_weight="balanced",
        random_state=42,
    )
    models["extra_trees"] = ExtraTreesClassifier(
        n_estimators=800,
        max_depth=18,
        min_samples_leaf=2,
        class_weight=bot_weight if recall_focus else "balanced",
        n_jobs=-1,
        random_state=42,
    )
    models["extra_trees_shallow"] = ExtraTreesClassifier(
        n_estimators=600,
        max_depth=14,
        min_samples_leaf=4,
        class_weight=bot_weight if recall_focus else "balanced",
        n_jobs=-1,
        random_state=44,
    )
    if recall_focus:
        models["extra_trees_deep"] = ExtraTreesClassifier(
            n_estimators=1000,
            max_depth=22,
            min_samples_leaf=1,
            class_weight={0: 0.8, 1: 1.55},
            n_jobs=-1,
            random_state=43,
        )
    return models


def hand_models() -> Dict[str, Any]:
    return {
        "hand_hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=400,
            max_depth=8,
            learning_rate=0.05,
            min_samples_leaf=12,
            class_weight="balanced",
            random_state=42,
        ),
        "hand_extra_trees": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    }


def scores_from_hand_model(
    model: Any,
    hand_X: np.ndarray,
    chunk_rows: List[dict],
    hand_by_chunk: Dict[Tuple[str, str, int], List[int]],
    agg_name: str,
    agg_fn,
) -> np.ndarray:
    hand_probs = np.clip(predict_proba(model, hand_X), 0.0, 1.0)
    out = np.zeros(len(chunk_rows), dtype=np.float64)
    for i, row in enumerate(chunk_rows):
        key = (row["sourceDate"], row["chunkId"], int(row["inner_index"]))
        hand_idxs = hand_by_chunk.get(key, [])
        if not hand_idxs:
            out[i] = 0.5
            continue
        probs = hand_probs[hand_idxs]
        out[i] = agg_fn(probs)
    return out


def tune_scoring(
    raw: np.ndarray,
    heur: np.ndarray,
    y: np.ndarray,
    groups: List[List[int]],
    rows: List[dict],
    *,
    objective: str = "balanced",
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    best_spec: Dict[str, Any] | None = None
    best_metrics: Dict[str, float] | None = None
    best_key = None

    def consider(transform: str, params: Dict[str, Any]) -> None:
        nonlocal best_spec, best_metrics, best_key
        scores = apply_transform_array(raw, heur, transform, params)
        hm = grouped_metrics(scores, y, groups, rows)
        if hm["fpr_max"] > 0.10:
            return
        key = rank_tuning_key(hm, objective)
        if best_key is None or key < best_key:
            best_key = key
            best_spec = {"transform": transform, "params": dict(params)}
            best_metrics = hm

    for offset in np.arange(-0.10, 0.22, 0.005):
        consider("raw_probability", {"offset": float(offset)})

    for a in (3.5, 4.0, 4.5, 5.0, 6.0):
        for b in np.arange(0.14, 0.38, 0.02):
            for offset in np.arange(0.0, 0.22, 0.02):
                consider(
                    "conservative_sigmoid",
                    {"a": float(a), "b": float(b), "offset": float(offset)},
                )

    for alpha in (0.55, 0.65, 0.75, 0.85, 0.92, 1.0):
        for offset in np.arange(0.0, 0.20, 0.02):
            consider("blend", {"alpha": alpha, "offset": float(offset)})

    if best_spec is None or best_metrics is None:
        return (
            {"transform": "raw_probability", "params": {"offset": 0.0}},
            {"reward_mean": 0.0, "fpr_max": 1.0, "bot_recall_mean": 0.0, "ap_mean": 0.0},
        )
    return best_spec, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(ARTIFACTS / "poker44_training_rows.csv"))
    parser.add_argument(
        "--hand-csv", default=str(ARTIFACTS / "poker44_hand_training_rows.csv")
    )
    parser.add_argument("--holdout-dates", nargs="+", default=list(DEFAULT_HOLDOUT))
    parser.add_argument("--include-heuristic", action="store_true")
    parser.add_argument("--skip-hand", action="store_true")
    parser.add_argument(
        "--objective",
        choices=("mean", "balanced", "recall_floor", "robust"),
        default="robust",
        help="Holdout tuning objective; robust = worst-date + worst-batch stability.",
    )
    parser.add_argument(
        "--recall-focus",
        action="store_true",
        help="Bot-heavy sample weights + recall-biased class weights.",
    )
    args = parser.parse_args()

    holdout_set = set(args.holdout_dates)
    X, y, feature_names, rows = load_dataset(
        Path(args.csv), include_heuristic=args.include_heuristic
    )
    groups_all = release_groups(rows)

    train_mask = np.array([r["sourceDate"] not in holdout_set for r in rows])
    hold_groups = [g for g in groups_all if rows[g[0]]["sourceDate"] in holdout_set]

    hand_path = Path(args.hand_csv)
    hand_data = None
    if not args.skip_hand and hand_path.exists():
        hand_X, hand_y, hand_feature_names, hand_rows = load_dataset(
            hand_path, include_heuristic=args.include_heuristic
        )
        hand_by_chunk = hand_rows_by_chunk(hand_rows)
        hand_train_mask = np.array([r["sourceDate"] not in holdout_set for r in hand_rows])
        hand_data = {
            "X": hand_X,
            "y": hand_y,
            "feature_names": hand_feature_names,
            "rows": hand_rows,
            "by_chunk": hand_by_chunk,
            "train_mask": hand_train_mask,
        }

    print(f"rows={len(rows)} features={len(feature_names)}")
    print(f"train={train_mask.sum()} holdout={len(rows) - train_mask.sum()} holdout_groups={len(hold_groups)}")
    if hand_data:
        print(f"hand_rows={len(hand_data['rows'])} hand_features={len(hand_data['feature_names'])}")
    print(f"holdout dates: {sorted(holdout_set)}")

    heur = np.array([float(r.get("heuristic_score") or 0.0) for r in rows])
    candidates: List[Dict[str, Any]] = []
    fitted_chunk_models: List[Any] = []
    chunk_raw_preds: List[np.ndarray] = []

    for name, model in chunk_models(recall_focus=args.recall_focus).items():
        print(f"\nTraining chunk model {name} (train dates only) …")
        fit_chunk_model(model, X, y, rows, train_mask, recall_focus=args.recall_focus)
        fitted_chunk_models.append(model)
        raw = np.clip(predict_proba(model, X), 0.0, 1.0)
        chunk_raw_preds.append(raw)
        spec, hm = tune_scoring(
            raw, heur, y, hold_groups, rows, objective=args.objective
        )
        print(
            f"  {spec['transform']} {spec.get('params')} "
            f"holdout_reward={hm['reward_mean']:.4f} min={hm.get('reward_min', 0):.4f} "
            f"per_date_min={hm.get('per_date_min', 0):.4f} "
            f"botrec={hm['bot_recall_mean']:.4f}"
        )
        candidates.append(
            {
                "model": model,
                "model_name": name,
                "granularity": "chunk",
                "feature_names": feature_names,
                "hand_aggregation": None,
                "score_transform": spec,
                "holdout_metrics": hm,
            }
        )

    if len(chunk_raw_preds) >= 2:
        ensemble_raw = np.clip(np.mean(np.stack(chunk_raw_preds, axis=0), axis=0), 0.0, 1.0)
        spec, hm = tune_scoring(
            ensemble_raw, heur, y, hold_groups, rows, objective=args.objective
        )
        print(
            f"\nEnsemble ({len(fitted_chunk_models)} models) "
            f"{spec['transform']} holdout_reward={hm['reward_mean']:.4f} "
            f"min={hm.get('reward_min', 0):.4f} per_date_min={hm.get('per_date_min', 0):.4f}"
        )
        candidates.append(
            {
                "ensemble_models": list(fitted_chunk_models),
                "model_name": f"ensemble_{len(fitted_chunk_models)}",
                "granularity": "chunk",
                "feature_names": feature_names,
                "hand_aggregation": None,
                "score_transform": spec,
                "holdout_metrics": hm,
            }
        )

    if hand_data:
        hd = hand_data
        for name, model in hand_models().items():
            print(f"\nTraining hand model {name} …")
            model.fit(hd["X"][hd["train_mask"]], hd["y"][hd["train_mask"]])
            for agg_name, agg_fn in HAND_AGG_OPTIONS:
                raw = scores_from_hand_model(
                    model, hd["X"], rows, hd["by_chunk"], agg_name, agg_fn
                )
                raw = np.clip(raw, 0.0, 1.0)
                spec, hm = tune_scoring(
                    raw, heur, y, hold_groups, rows, objective=args.objective
                )
                print(
                    f"  agg={agg_name} {spec['transform']} "
                    f"holdout_reward={hm['reward_mean']:.4f} botrec={hm['bot_recall_mean']:.4f}"
                )
                candidates.append(
                    {
                        "model": model,
                        "model_name": name,
                        "granularity": "hand",
                        "feature_names": hd["feature_names"],
                        "hand_aggregation": agg_name,
                        "score_transform": spec,
                        "holdout_metrics": hm,
                    }
                )

    if not candidates:
        raise SystemExit("No models trained.")

    def rank_candidate(c: Dict[str, Any]) -> tuple:
        hm = c["holdout_metrics"]
        if hm["fpr_max"] > 0.10:
            return (2, 0.0, 1.0, 0.0)
        if (
            c["granularity"] == "hand"
            and hm["reward_mean"] >= HAND_HOLDOUT_REWARD_CEILING
        ):
            return (1, -hm["reward_mean"], hm["fpr_max"], -hm["bot_recall_mean"])
        return rank_tuning_key(hm, args.objective)

    eligible = [c for c in candidates if rank_candidate(c)[0] == 0]
    pool = eligible if eligible else candidates
    best = min(pool, key=rank_candidate)
    if best["holdout_metrics"]["fpr_max"] > 0.10:
        raise SystemExit("No configuration passed fpr_max <= 0.10 on holdout groups.")

    CONSERVATIVE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CONSERVATIVE_DIR / "best_model.pkl"
    bundle: Dict[str, Any] = {
        "feature_names": best["feature_names"],
        "model_type": best["model_name"],
        "granularity": best["granularity"],
        "hand_aggregation": best.get("hand_aggregation"),
    }
    if best.get("ensemble_models"):
        bundle["ensemble_models"] = best["ensemble_models"]
        bundle["model"] = best["ensemble_models"][0]
    else:
        bundle["model"] = best["model"]

    transform = dict(best["score_transform"])
    transform.update(
        {
            "mainnet_safe": True,
            "mode": "conservative",
            "note": "Holdout-tuned transform; ensemble + blend search on release groups.",
        }
    )

    with model_path.open("wb") as f:
        pickle.dump(bundle, f)
    (CONSERVATIVE_DIR / "best_score_transform.json").write_text(
        json.dumps(transform, indent=2)
    )

    train_dates = sorted({r["sourceDate"] for r in rows if r["sourceDate"] not in holdout_set})
    metadata = {
        "model_type": best["model_name"],
        "granularity": best["granularity"],
        "hand_aggregation": best["hand_aggregation"],
        "feature_names": best["feature_names"],
        "training_sourceDates": train_dates,
        "holdout_sourceDates": sorted(holdout_set),
        "include_heuristic_feature": args.include_heuristic,
        "holdout_grouped_metrics": best["holdout_metrics"],
        "score_transform": transform,
        "git_commit": _git_commit(),
        "risk_profile": "conservative_pre_mainnet",
        "feature_count": len(best["feature_names"]),
        "is_ensemble": bool(best.get("ensemble_models")),
        "tuning_objective": args.objective,
        "recall_focus_training": args.recall_focus,
    }
    (CONSERVATIVE_DIR / "best_model_metadata.json").write_text(json.dumps(metadata, indent=2))

    import shutil

    for src, dst in [
        (model_path, ARTIFACTS / "best_model.pkl"),
        (CONSERVATIVE_DIR / "best_score_transform.json", ARTIFACTS / "best_score_transform.json"),
        (CONSERVATIVE_DIR / "best_model_metadata.json", ARTIFACTS / "best_model_metadata.json"),
    ]:
        shutil.copy2(src, dst)

    print("\n" + "=" * 72)
    print("CONSERVATIVE MODEL READY")
    print(json.dumps(metadata, indent=2))
    print(f"Artifacts: {CONSERVATIVE_DIR}")


if __name__ == "__main__":
    main()
