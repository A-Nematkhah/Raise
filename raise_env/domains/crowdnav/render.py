"""Headless matplotlib frames of a CrowdNav scene (upstream ``render`` needs a display / ffmpeg)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle

# Same view and colours as upstream ``CrowdSim.render(mode='video')``.
_VIEW_HALF_WIDTH = 6.0
_ROBOT_COLOR = "yellow"
_GOAL_COLOR = "red"


def render_frame(
    *,
    robot_xy: Tuple[float, float],
    robot_radius: float,
    goal_xy: Tuple[float, float],
    humans: Sequence[Tuple[float, float, float]],
    global_time: float,
    title: Optional[str] = None,
    trail: Sequence[Tuple[float, float]] = (),
    size_px: int = 480,
) -> np.ndarray:
    """RGB ``uint8`` frame: robot (filled), humans ``(px, py, radius)`` (outlined), goal star."""
    dpi = 80
    fig = Figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.set_xlim(-_VIEW_HALF_WIDTH, _VIEW_HALF_WIDTH)
    ax.set_ylim(-_VIEW_HALF_WIDTH, _VIEW_HALF_WIDTH)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if len(trail) > 1:
        xs, ys = zip(*trail)
        ax.plot(xs, ys, color="orange", lw=1.0, alpha=0.8)
    ax.plot([goal_xy[0]], [goal_xy[1]], marker="*", color=_GOAL_COLOR, markersize=15, linestyle="None")
    ax.add_artist(Circle(robot_xy, robot_radius, fill=True, color=_ROBOT_COLOR, ec="black"))
    for i, (hx, hy, hr) in enumerate(humans):
        ax.add_artist(Circle((hx, hy), hr, fill=False, color="tab:blue"))
        ax.text(hx - 0.11, hy - 0.11, str(i), fontsize=8)
    ax.text(-_VIEW_HALF_WIDTH + 0.3, _VIEW_HALF_WIDTH - 0.8, f"Time: {global_time:.2f}", fontsize=10)
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


__all__ = ["render_frame"]
