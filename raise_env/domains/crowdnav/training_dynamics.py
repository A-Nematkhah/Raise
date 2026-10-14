"""
Training-rollout outcome statistics per window of the PPO budget.

Reads the ``info`` dicts SB3 already collects (``RewardInjectedCrowdNavEnv``
flags + ``Monitor`` episode stats); no extra environment steps or evaluations.
These are outcomes of the stochastic behaviour policy on training scenarios,
not separate checkpoint evaluations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


class TrainingDynamicsRecorder:
    def __init__(self, total_timesteps: int, *, n_windows: int = 4) -> None:
        self.total = max(1, int(total_timesteps))
        self.n_windows = max(1, int(n_windows))
        self._episodes: List[List[Dict[str, Any]]] = [[] for _ in range(self.n_windows)]

    def _window(self, num_timesteps: int) -> int:
        return min(self.n_windows - 1, int(num_timesteps) * self.n_windows // self.total)

    def observe(self, infos: Sequence[Any], dones: Optional[Sequence[Any]], num_timesteps: int) -> None:
        if dones is None:
            return
        for i, info in enumerate(infos or ()):
            if not isinstance(info, dict) or not bool(dones[i]):
                continue
            outcome = info.get("raise_outcome") or "timeout"
            monitor = info.get("episode") or {}
            self._episodes[self._window(num_timesteps)].append(
                {
                    "outcome": str(outcome),
                    "return": float(monitor.get("r", float("nan"))),
                    "length": float(monitor.get("l", 0.0)),
                    "time": float(info.get("raise_global_time", 0.0) or 0.0),
                }
            )

    def summary(self) -> Dict[str, Any]:
        windows = []
        for w, eps in enumerate(self._episodes):
            lo = self.total * w // self.n_windows
            hi = self.total * (w + 1) // self.n_windows
            row: Dict[str, Any] = {"steps": [lo, hi], "n_episodes": len(eps)}
            if eps:
                n = float(len(eps))
                returns = [e["return"] for e in eps if e["return"] == e["return"]]
                success_times = [e["time"] for e in eps if e["outcome"] == "success"]
                row.update(
                    SR=sum(e["outcome"] == "success" for e in eps) / n,
                    CR=sum(e["outcome"] == "collision" for e in eps) / n,
                    TR=sum(e["outcome"] == "timeout" for e in eps) / n,
                    mean_return=(sum(returns) / len(returns)) if returns else None,
                    mean_length=sum(e["length"] for e in eps) / n,
                    nav_time=(sum(success_times) / len(success_times)) if success_times else None,
                )
            windows.append(row)
        return {
            "source": "training_rollouts",
            "total_timesteps": self.total,
            "n_episodes": sum(len(e) for e in self._episodes),
            "windows": windows,
        }


__all__ = ["TrainingDynamicsRecorder"]
