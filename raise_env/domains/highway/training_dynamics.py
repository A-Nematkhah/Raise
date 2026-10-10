"""
Training-rollout outcome statistics per window of the PPO budget.

Reads the ``info`` dicts SB3 already collects (``RewardInjectedHighwayEnv``
flags + ``Monitor`` episode stats); no extra environment steps or evaluations.
These are outcomes of the stochastic behaviour policy in training traffic,
not separate checkpoint evaluations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


class TrainingDynamicsRecorder:
    def __init__(self, total_timesteps: int, *, n_windows: int = 4) -> None:
        self.total = max(1, int(total_timesteps))
        self.n_windows = max(1, int(n_windows))
        self._episodes: List[List[Dict[str, float]]] = [[] for _ in range(self.n_windows)]
        self._speed_sum: Dict[int, float] = {}
        self._steps: Dict[int, int] = {}

    def _window(self, num_timesteps: int) -> int:
        return min(self.n_windows - 1, int(num_timesteps) * self.n_windows // self.total)

    def observe(self, infos: Sequence[Any], dones: Optional[Sequence[Any]], num_timesteps: int) -> None:
        if dones is None:
            return
        for i, info in enumerate(infos or ()):
            if not isinstance(info, dict):
                continue
            self._speed_sum[i] = self._speed_sum.get(i, 0.0) + float(info.get("raise_speed", 0.0) or 0.0)
            self._steps[i] = self._steps.get(i, 0) + 1
            if not bool(dones[i]):
                continue
            if info.get("raise_collision"):
                outcome = "collision"
            elif info.get("raise_off_road"):
                outcome = "off_road"
            else:
                outcome = "success"
            monitor = info.get("episode") or {}
            steps = self._steps.get(i, 0)
            self._episodes[self._window(num_timesteps)].append(
                {
                    "outcome": outcome,
                    "return": float(monitor.get("r", float("nan"))),
                    "length": float(monitor.get("l", steps)),
                    "mean_speed": self._speed_sum.get(i, 0.0) / max(1, steps),
                }
            )
            self._speed_sum[i] = 0.0
            self._steps[i] = 0

    def summary(self) -> Dict[str, Any]:
        windows = []
        for w, eps in enumerate(self._episodes):
            lo = self.total * w // self.n_windows
            hi = self.total * (w + 1) // self.n_windows
            row: Dict[str, Any] = {"steps": [lo, hi], "n_episodes": len(eps)}
            if eps:
                n = float(len(eps))
                returns = [e["return"] for e in eps if e["return"] == e["return"]]
                row.update(
                    SR=sum(e["outcome"] == "success" for e in eps) / n,
                    CR=sum(e["outcome"] == "collision" for e in eps) / n,
                    TR=sum(e["outcome"] == "off_road" for e in eps) / n,
                    mean_return=(sum(returns) / len(returns)) if returns else None,
                    mean_length=sum(e["length"] for e in eps) / n,
                    mean_speed=sum(e["mean_speed"] for e in eps) / n,
                )
            windows.append(row)
        return {
            "source": "training_rollouts",
            "total_timesteps": self.total,
            "n_episodes": sum(len(e) for e in self._episodes),
            "windows": windows,
        }


__all__ = ["TrainingDynamicsRecorder"]
