"""Count overtakes from simulator ground truth (not from the reward's view)."""

from __future__ import annotations

from typing import Any, Dict


class OvertakeTracker:
    """
    Per-episode pass counter along the road's longitudinal axis.

    ``passes`` — a vehicle that was ahead of the ego is now behind it.

    Call ``observe(env)`` after reset and after every non-terminal step.
    """

    def __init__(self) -> None:
        self.passes = 0
        self._side: Dict[int, int] = {}

    def observe(self, env: Any) -> None:
        base = getattr(env, "unwrapped", env)
        ego = getattr(base, "vehicle", None)
        road = getattr(base, "road", None)
        if ego is None or road is None:
            return
        ego_x = float(ego.position[0])
        for other in road.vehicles:
            if other is ego:
                continue
            dx = float(other.position[0]) - ego_x
            if dx == 0.0:
                continue
            side = 1 if dx > 0.0 else -1
            key = id(other)
            prev = self._side.get(key)
            if prev == 1 and side == -1:
                self.passes += 1
            self._side[key] = side
