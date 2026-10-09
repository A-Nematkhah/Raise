"""Count overtakes from simulator ground truth (not from the reward's view)."""

from __future__ import annotations

from typing import Any, Dict, Set


class OvertakeTracker:
    """
    Per-episode pass counter along the road's longitudinal axis.

    ``passes`` — vehicles that were ahead of the ego and are currently behind
    it. Losing the position again cancels that vehicle's pass, so two cars
    swapping back and forth do not inflate the count.

    Call ``observe(env)`` after reset and after every non-terminal step.
    """

    def __init__(self) -> None:
        self._side: Dict[int, int] = {}
        self._passed: Set[int] = set()

    @property
    def passes(self) -> int:
        return len(self._passed)

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
                self._passed.add(key)
            elif prev == -1 and side == 1:
                self._passed.discard(key)
            self._side[key] = side
