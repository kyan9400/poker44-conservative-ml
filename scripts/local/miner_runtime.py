"""Shared Poker44 miner runtime for local axon and mainnet neurons/miner.py."""

from __future__ import annotations

import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Tuple

import bittensor as bt

from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def detect_repo_commit(repo_root: Path | None = None) -> str:
    explicit = os.getenv("POKER44_MODEL_REPO_COMMIT", "").strip()
    if explicit:
        return explicit
    root = repo_root or REPO_ROOT
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def detect_repo_url(repo_root: Path | None = None) -> str:
    """Resolve a mainnet-safe repo URL (never local:// unless explicitly allowed)."""
    explicit = os.getenv("POKER44_MODEL_REPO_URL", "").strip()
    allow_local = os.getenv("POKER44_ALLOW_LOCAL_MANIFEST", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if explicit:
        if explicit.startswith("local://") and not allow_local:
            bt.logging.warning(
                "POKER44_MODEL_REPO_URL is local:// but POKER44_ALLOW_LOCAL_MANIFEST is not set; "
                "using git remote or default public repo instead."
            )
        else:
            return explicit.rstrip("/")

    root = repo_root or REPO_ROOT
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return "https://github.com/Poker44/Poker44-subnet"

    if result.returncode != 0:
        return "https://github.com/Poker44/Poker44-subnet"

    remote = result.stdout.strip()
    if remote.startswith("git@"):
        # git@github.com:Org/repo.git -> https://github.com/Org/repo
        host_path = remote.split(":", 1)
        if len(host_path) == 2:
            host = host_path[0].split("@", 1)[-1]
            path = host_path[1].removesuffix(".git")
            return f"https://{host}/{path}"
    if remote.startswith("https://") or remote.startswith("http://"):
        return remote.removesuffix(".git")
    return "https://github.com/Poker44/Poker44-subnet"


def parse_calibration_mode(
    spec: str,
) -> Tuple[str, Callable[[List[float]], List[float]], dict]:
    raw = (spec or "").strip().lower()
    if not raw or raw == "identity":
        return "identity", (lambda xs: list(xs)), {}

    def _kv(args_text: str) -> dict:
        kv = {}
        for part in args_text.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"missing '=' in {part!r}")
            key, value = part.split("=", 1)
            kv[key.strip()] = float(value.strip())
        return kv

    try:
        if raw == "rank":
            def fn(xs: List[float]) -> List[float]:
                n = len(xs)
                if n <= 1:
                    return list(xs)
                order = sorted(range(n), key=lambda i: xs[i])
                ranks = [0.0] * n
                for rank, idx in enumerate(order):
                    ranks[idx] = rank / (n - 1)
                return ranks

            return "rank", fn, {}

        if ":" not in raw:
            raise ValueError("missing ':' separator")
        head, args_text = raw.split(":", 1)
        head = head.strip()

        if head == "linear":
            params = _kv(args_text)
            scale = params["k"]

            def fn(xs: List[float]) -> List[float]:
                return [max(0.0, min(1.0, x * scale)) for x in xs]

            return f"linear:k={scale}", fn, {"k": scale}

        if head == "power":
            params = _kv(args_text)
            gamma = params["gamma"]

            def fn(xs: List[float]) -> List[float]:
                return [max(0.0, min(1.0, max(0.0, x) ** gamma)) for x in xs]

            return f"power:gamma={gamma}", fn, {"gamma": gamma}

        if head == "sigmoid":
            params = _kv(args_text)
            a = params["a"]
            b = params["b"]

            def fn(xs: List[float]) -> List[float]:
                return [1.0 / (1.0 + math.exp(-a * (x - b))) for x in xs]

            return f"sigmoid:a={a},b={b}", fn, {"a": a, "b": b}

        raise ValueError(f"unknown family {head!r}")
    except Exception as exc:
        bt.logging.warning(
            f"POKER44_CALIBRATION_MODE={spec!r} parse failed ({exc}); using identity."
        )
        return "identity", (lambda xs: list(xs)), {}


def _configure_artifacts_dir() -> None:
    risk_mode = os.getenv("POKER44_RISK_MODE", "conservative").strip().lower()
    if os.getenv("POKER44_ARTIFACTS_DIR", "").strip():
        return
    if risk_mode == "aggressive":
        os.environ["POKER44_ARTIFACTS_DIR"] = str(
            REPO_ROOT / "scripts" / "local" / "artifacts" / "aggressive_max"
        )
    else:
        conservative = REPO_ROOT / "scripts" / "local" / "artifacts" / "conservative"
        if (conservative / "best_model.pkl").exists():
            os.environ["POKER44_ARTIFACTS_DIR"] = str(conservative)


class MinerRuntime:
    """Env-driven scoring + manifest for conservative ML or heuristic fallback."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root or REPO_ROOT
        self.model_mode = "heuristic"
        self.blend_alpha = 0.5
        self.cal_mode_name = "identity"
        self.cal_fn: Callable[[List[float]], List[float]] = lambda xs: list(xs)
        self.cal_params: dict = {}
        self.ml_scorer: object | None = None
        self.manifest: dict = {}
        self.manifest_compliance: dict = {}
        self.manifest_digest = ""

    def configure(self) -> None:
        _configure_artifacts_dir()
        self.model_mode = os.getenv("POKER44_MODEL_MODE", "heuristic").strip().lower()
        self.blend_alpha = float(os.getenv("POKER44_BLEND_ALPHA", "0.5"))
        if self.model_mode not in {"heuristic", "ml", "blend"}:
            bt.logging.warning(
                f"Unknown POKER44_MODEL_MODE={self.model_mode!r}; using heuristic."
            )
            self.model_mode = "heuristic"

        if self.model_mode in {"ml", "blend"}:
            try:
                from scripts.local.ml_inference import MLScorer

                scorer = MLScorer()
                scorer.load()
                self.ml_scorer = scorer
                bt.logging.info(
                    f"ML model loaded type={scorer.model_type} path={scorer.model_path}"
                )
            except Exception as exc:
                bt.logging.warning(f"ML load failed ({exc}); heuristic fallback.")
                self.ml_scorer = None

        cal_spec = os.getenv("POKER44_CALIBRATION_MODE", "identity")
        self.cal_mode_name, self.cal_fn, self.cal_params = parse_calibration_mode(cal_spec)
        self.manifest = self.build_manifest()
        self.manifest_compliance = evaluate_manifest_compliance(self.manifest)
        self.manifest_digest = manifest_digest(self.manifest)

    def build_manifest(self) -> dict:
        commit = detect_repo_commit(self.repo_root)
        if commit and not os.getenv("POKER44_MODEL_REPO_COMMIT"):
            os.environ["POKER44_MODEL_REPO_COMMIT"] = commit
        repo_url = detect_repo_url(self.repo_root)
        uses_ml = self.model_mode in {"ml", "blend"} and self.ml_scorer is not None

        if uses_ml:
            ml_type = self.ml_scorer.model_type if self.ml_scorer else "unloaded"
            impl_files = [
                self.repo_root / "neurons" / "miner.py",
                self.repo_root / "scripts" / "local" / "miner_runtime.py",
                self.repo_root / "scripts" / "local" / "features.py",
                self.repo_root / "scripts" / "local" / "ml_inference.py",
                self.repo_root / "scripts" / "local" / "train_conservative.py",
            ]
            defaults = {
                "model_name": os.getenv(
                    "POKER44_MODEL_NAME", "poker44-conservative-ml"
                ),
                "model_version": f"ml-1+{self.model_mode}",
                "framework": "sklearn",
                "license": "MIT",
                "repo_url": repo_url,
                "repo_commit": commit,
                "notes": (
                    f"Supervised conservative ML miner (mode={self.model_mode}, "
                    f"family={ml_type}). Trained on public benchmark API only."
                ),
                "open_source": True,
                "inference_mode": "local",
                "training_data_statement": (
                    "Supervised classifier trained on public Poker44 benchmark "
                    "releases via GET https://api.poker44.net/api/v1/benchmark. "
                    "Hold-out dates used for model selection only. "
                    "No validator-private evaluation data."
                ),
                "training_data_sources": [
                    "https://api.poker44.net/api/v1/benchmark",
                ],
                "private_data_attestation": (
                    "Inference does not load labels. Training used only public "
                    "benchmark groundTruthLabels offline."
                ),
            }
        else:
            impl_files = [
                self.repo_root / "neurons" / "miner.py",
                self.repo_root / "scripts" / "local" / "miner_runtime.py",
            ]
            defaults = {
                "model_name": "poker44-reference-heuristic",
                "model_version": f"1+cal:{self.cal_mode_name}",
                "framework": "python-heuristic",
                "license": "MIT",
                "repo_url": repo_url,
                "repo_commit": commit,
                "notes": "Reference heuristic miner (POKER44_MODEL_MODE=heuristic).",
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Reference heuristic miner. No training step. Uses runtime chunk features."
                ),
                "training_data_sources": ["none"],
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data."
                ),
            }

        return build_local_model_manifest(
            repo_root=self.repo_root,
            implementation_files=impl_files,
            defaults=defaults,
        )

    def score_chunks(self, chunks: List[list]) -> List[float]:
        from neurons.miner import Miner  # lazy: avoids circular import at module load

        heur_scores: List[float] = []
        output_scores: List[float] = []

        try:
            if self.model_mode in {"ml", "blend"} and self.ml_scorer is not None:
                model_scores, _, heur_scores = self.ml_scorer.score_chunks(
                    chunks, mode=self.model_mode, blend_alpha=self.blend_alpha
                )
                output_scores = model_scores
            else:
                heur_scores = [Miner.score_chunk(chunk) for chunk in chunks]
                output_scores = heur_scores
        except Exception as exc:
            bt.logging.error(f"Scoring failed ({exc}); heuristic fallback.")
            heur_scores = [Miner.score_chunk(chunk) for chunk in chunks]
            output_scores = heur_scores

        final = self._clamp01_list(self.cal_fn(output_scores)) if output_scores else []
        self._maybe_log_predictions(chunks, output_scores, heur_scores)
        if chunks and len(final) != len(chunks):
            bt.logging.warning(
                f"Score count mismatch ({len(final)} vs {len(chunks)}); padding."
            )
            fill = heur_scores or [Miner.score_chunk(c) for c in chunks]
            final = []
            for i, _chunk in enumerate(chunks):
                if i < len(output_scores):
                    final.append(self._clamp01_list(self.cal_fn([output_scores[i]]))[0])
                elif i < len(fill):
                    final.append(self._clamp01_list(self.cal_fn([fill[i]]))[0])
                else:
                    final.append(Miner.score_chunk(_chunk))
        return final

    @staticmethod
    def _clamp01_list(values: List[float]) -> List[float]:
        return [max(0.0, min(1.0, float(v))) for v in values]

    def _maybe_log_predictions(
        self,
        chunks: List[list],
        model_scores: List[float],
        heur_scores: List[float],
    ) -> None:
        log_path = os.getenv("POKER44_PREDICTION_LOG", "").strip()
        if not log_path:
            return
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "mode": self.model_mode,
            "n_chunks": len(chunks),
            "scores": model_scores[:20],
            "heuristic": heur_scores[:20] if heur_scores else [],
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def log_startup(self) -> None:
        bt.logging.info(
            f"Miner runtime | mode={self.model_mode} calibration={self.cal_mode_name} "
            f"artifacts={os.getenv('POKER44_ARTIFACTS_DIR', 'default')}"
        )
        bt.logging.info(
            f"Manifest status={self.manifest_compliance.get('status')} "
            f"missing={self.manifest_compliance.get('missing_fields')} "
            f"digest={self.manifest_digest}"
        )
        bt.logging.info(
            f"Manifest repo={self.manifest.get('repo_url')} "
            f"commit={self.manifest.get('repo_commit')}"
        )
        if str(self.manifest.get("repo_url", "")).startswith("local://"):
            bt.logging.warning(
                "repo_url is still local:// — set POKER44_MODEL_REPO_URL before mainnet."
            )
