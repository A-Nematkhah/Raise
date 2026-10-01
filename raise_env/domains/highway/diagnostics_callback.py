"""Purely observational SB3 callbacks for highway PPO diagnostics.

Never alters training control flow: ``_on_step`` always returns True;
file I/O failures are swallowed with a warning. Ground-truth checkpoint
results are written *only* to the per-candidate JSONL — never into
candidate.metadata or any selection/fitness path.

Reward-component trends (EUREKA): per-step dicts arrive via
``info['raise_reward_components']``. This callback is the *only* place that
aggregates them — once per rollout at ``_on_rollout_end`` (and optionally
later at GroundTruth fractions). Env / SandboxedReward keep last-step only.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

logger = logging.getLogger(__name__)

# SB3 LoggerRecorder keys vary slightly by version; try prefixed then bare.
_KEY_ALIASES = {
    "ep_rew_mean": ("rollout/ep_rew_mean", "ep_rew_mean"),
    "explained_variance": ("train/explained_variance", "explained_variance"),
    "approx_kl": ("train/approx_kl", "approx_kl"),
    "entropy_loss": ("train/entropy_loss", "entropy_loss"),
    "value_loss": ("train/value_loss", "value_loss"),
    "policy_gradient_loss": (
        "train/policy_gradient_loss",
        "policy_gradient_loss",
    ),
    "clip_fraction": ("train/clip_fraction", "clip_fraction"),
    "fps": ("time/fps", "fps"),
}


def _logger_get(name_to_value: Any, field: str) -> Optional[float]:
    if not isinstance(name_to_value, dict):
        return None
    for key in _KEY_ALIASES.get(field, (field,)):
        if key in name_to_value:
            try:
                return float(name_to_value[key])
            except (TypeError, ValueError):
                return None
    return None


def _append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


class RolloutDiagnosticsCallback(BaseCallback):
    """Append one free-signal JSON line per PPO rollout (Part 1).

    Also snapshots mean reward-component values once per rollout for EUREKA
    reflection (``component_history``).
    """

    def __init__(
        self,
        *,
        candidate_id: str,
        log_path: str,
        continuous_actions: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.candidate_id = str(candidate_id)
        self.log_path = str(log_path)
        self.continuous_actions = bool(continuous_actions)
        self._throttle_buf: List[float] = []
        self._speed_buf: List[float] = []
        # Per-rollout running sums/counts (keys may appear mid-rollout).
        self._comp_sums: Dict[str, float] = {}
        self._comp_counts: Dict[str, int] = {}
        self._keys_seen_this_rollout: set[str] = set()
        self._warned_new_keys: set[str] = set()
        # Across rollouts: name -> list of per-rollout means.
        self.component_history: Dict[str, List[float]] = {}

    def _ingest_step_components(self, comps: Mapping[str, Any]) -> None:
        """Accumulate one step's components; missing keys are skipped (not zeroed)."""
        if not isinstance(comps, Mapping) or not comps:
            return
        step_keys = {str(k) for k in comps.keys()}
        if self._keys_seen_this_rollout:
            for name in step_keys - self._keys_seen_this_rollout:
                if name not in self._warned_new_keys:
                    self._warned_new_keys.add(name)
                    warnings.warn(
                        f"reward component key {name!r} appeared mid-rollout for "
                        f"{self.candidate_id}; using union of keys (missing steps "
                        "excluded from that key's mean)",
                        stacklevel=2,
                    )
        self._keys_seen_this_rollout |= step_keys
        for key, raw in comps.items():
            name = str(key)
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if val != val:  # NaN
                continue
            self._comp_sums[name] = self._comp_sums.get(name, 0.0) + val
            self._comp_counts[name] = self._comp_counts.get(name, 0) + 1

    def _flush_rollout_component_means(self) -> Dict[str, float]:
        """Compute per-component means for this rollout; append to history."""
        means: Dict[str, float] = {}
        for name, total in self._comp_sums.items():
            n = int(self._comp_counts.get(name, 0))
            if n <= 0:
                continue
            mean = float(total) / float(n)
            means[name] = mean
            self.component_history.setdefault(name, []).append(mean)
        self._comp_sums.clear()
        self._comp_counts.clear()
        self._keys_seen_this_rollout.clear()
        self._warned_new_keys.clear()
        return means

    def component_trend_summary(self) -> Dict[str, Any]:
        """Serializable trends for candidate.metadata / evidence_block."""
        out: Dict[str, Any] = {}
        for name, vals in self.component_history.items():
            series = [float(v) for v in vals]
            if not series:
                continue
            out[name] = {
                "values": series,
                "max": float(max(series)),
                "mean": float(sum(series) / len(series)),
                "min": float(min(series)),
            }
        return out

    def _on_step(self) -> bool:
        try:
            infos = self.locals.get("infos")
            if infos:
                for info in infos:
                    if not isinstance(info, dict):
                        continue
                    comps = info.get("raise_reward_components")
                    if isinstance(comps, dict):
                        self._ingest_step_components(comps)
                    if self.continuous_actions and "raise_speed" in info:
                        self._speed_buf.append(float(info["raise_speed"]))
            if self.continuous_actions:
                actions = self.locals.get("actions")
                if actions is not None:
                    arr = np.asarray(actions, dtype=np.float64).reshape(-1)
                    if arr.size >= 1:
                        self._throttle_buf.append(float(arr[0]))
        except Exception:  # noqa: BLE001
            pass
        return True

    def _on_rollout_end(self) -> None:
        try:
            rollout_means = self._flush_rollout_component_means()
            ntv = getattr(getattr(self.model, "logger", None), "name_to_value", None)
            record: dict = {
                "candidate_id": self.candidate_id,
                "steps": int(self.num_timesteps),
                "ep_rew_mean": _logger_get(ntv, "ep_rew_mean"),
                "explained_variance": _logger_get(ntv, "explained_variance"),
                "approx_kl": _logger_get(ntv, "approx_kl"),
                "entropy_loss": _logger_get(ntv, "entropy_loss"),
                "value_loss": _logger_get(ntv, "value_loss"),
                "policy_gradient_loss": _logger_get(ntv, "policy_gradient_loss"),
                "clip_fraction": _logger_get(ntv, "clip_fraction"),
                "fps": _logger_get(ntv, "fps"),
            }
            if rollout_means:
                record["reward_component_means"] = rollout_means
            if self.continuous_actions:
                record.update(self._continuous_rollout_stats())
                self._throttle_buf.clear()
                self._speed_buf.clear()
            _append_jsonl(self.log_path, record)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "diagnostics_rollout write failed for %s: %s",
                self.candidate_id,
                exc,
            )

    def _continuous_rollout_stats(self) -> dict:
        out: dict = {"action_mode": "continuous"}
        if self._throttle_buf:
            thr = np.asarray(self._throttle_buf, dtype=np.float64)
            hist, edges = np.histogram(thr, bins=10, range=(-1.0, 1.0))
            out["throttle_mean"] = float(np.mean(thr))
            out["throttle_std"] = float(np.std(thr))
            out["throttle_hist"] = hist.astype(int).tolist()
            out["throttle_hist_edges"] = [float(x) for x in edges]
        if self._speed_buf:
            spd = np.asarray(self._speed_buf, dtype=np.float64)
            out["speed_mean"] = float(np.mean(spd))
            out["speed_std"] = float(np.std(spd))
            out["speed_p10"] = float(np.percentile(spd, 10))
            out["speed_p50"] = float(np.percentile(spd, 50))
            out["speed_p90"] = float(np.percentile(spd, 90))
            shist, sedges = np.histogram(spd, bins=10)
            out["speed_hist"] = shist.astype(int).tolist()
            out["speed_hist_edges"] = [float(x) for x in sedges]
        return out


class GroundTruthCheckpointCallback(BaseCallback):
    """Cheap holdout eval at fixed fractions of the train budget (Part 2).

    Results go only to ``diagnostics_groundtruth.jsonl`` — never metadata.
    Component-trend snapshots at GT fractions are deferred (optional).
    """

    def __init__(
        self,
        *,
        candidate_id: str,
        log_path: str,
        train_env_steps: int,
        reward_fn: Any,
        seed: int,
        n_episodes: int = 4,
        fractions: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.candidate_id = str(candidate_id)
        self.log_path = str(log_path)
        self.reward_fn = reward_fn
        self.seed = int(seed)
        self.n_episodes = max(1, int(n_episodes))
        total = max(1, int(train_env_steps))
        self._thresholds: List[int] = []
        for frac in fractions:
            thr = int(round(float(frac) * total))
            thr = max(1, min(total, thr))
            if not self._thresholds or thr > self._thresholds[-1]:
                self._thresholds.append(thr)
        if self._thresholds[-1] < total:
            self._thresholds.append(total)
        self._next_idx = 0

    def _on_step(self) -> bool:
        self._flush_due_checkpoints(force_all=False)
        return True

    def _on_training_end(self) -> None:
        # Ensure the final (100%) mark is recorded even if learn() stopped
        # between rollouts without another _on_step past the last threshold.
        self._flush_due_checkpoints(force_all=True)

    def _flush_due_checkpoints(self, *, force_all: bool) -> None:
        steps = int(self.num_timesteps)
        while self._next_idx < len(self._thresholds):
            thr = self._thresholds[self._next_idx]
            if steps >= thr:
                self._run_and_log(steps)
                self._next_idx += 1
                continue
            # At training end, still record the final (100%) mark if we stopped
            # a few steps short of the exact threshold due to rollout alignment.
            if force_all and self._next_idx == len(self._thresholds) - 1:
                self._run_and_log(steps)
                self._next_idx += 1
                continue
            break

    def _run_and_log(self, steps: int) -> None:
        try:
            from domains.highway.adapter import _eval_metrics
            from domains.highway.env_wrapper import (
                HOLDOUT_SEED_OFFSET,
                RewardInjectedHighwayEnv,
                holdout_env_config,
            )

            env = RewardInjectedHighwayEnv(
                self.reward_fn,
                seed=self.seed + 7,
                config=holdout_env_config(),
            )
            try:
                metrics = _eval_metrics(
                    env,
                    self.model,
                    n_episodes=self.n_episodes,
                    seed=self.seed,
                    seed_offset=HOLDOUT_SEED_OFFSET,
                )
            finally:
                try:
                    env.close()
                except Exception:  # noqa: BLE001
                    pass

            extras = getattr(metrics, "_highway_extras", None) or {}
            record = {
                "candidate_id": self.candidate_id,
                "steps": int(steps),
                "SR": float(metrics.sr),
                "CR": float(metrics.cr),
                "TR": float(metrics.tr),
                "mean_speed": float(extras.get("mean_speed", metrics.itr)),
                "mean_progress": float(extras.get("mean_progress", metrics.pl)),
                "soft_success": float(extras.get("soft_success", 0.0)),
                "speed_p10": float(extras.get("speed_p10", 0.0)),
                "speed_p90": float(extras.get("speed_p90", 0.0)),
            }
            _append_jsonl(self.log_path, record)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "diagnostics_groundtruth write failed for %s @ steps=%s: %s",
                self.candidate_id,
                steps,
                exc,
            )


def combine_train_callbacks(callbacks: Sequence[BaseCallback]) -> BaseCallback:
    """Combine progress + diagnostic callbacks without dropping any."""
    cbs = [c for c in callbacks if c is not None]
    if len(cbs) == 1:
        return cbs[0]
    return CallbackList(list(cbs))
