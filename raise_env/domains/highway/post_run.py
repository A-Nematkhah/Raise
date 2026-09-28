"""Post-run plots + highway animation for a completed RAISE output dir."""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _load_plot_raise_run():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(root, "scripts", "plot_raise_run.py")
    spec = importlib.util.spec_from_file_location("plot_raise_run", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_highway_run_artifacts(
    run_dir: str,
    *,
    episodes: int = 2,
    seed: int = 425,
    skip_viz: bool = False,
) -> Dict[str, Any]:
    """
    Write ``plots/*.png`` and ``plots/viz/*.gif`` (best policy rollout).

    Failures are logged; never raises to the training pipeline.
    """
    run_dir = os.path.abspath(run_dir)
    plots_dir = os.path.join(run_dir, "plots")
    result: Dict[str, Any] = {
        "run_dir": run_dir,
        "plots": [],
        "viz": None,
        "ok": False,
    }
    try:
        plot_mod = _load_plot_raise_run()
        written = plot_mod.run_plots(run_dir, plots_dir, show=False)
        result["plots"] = list(written)
        logger.info("Wrote %d plot(s) under %s", len(written), plots_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("plot_raise_run failed: %s", exc)
        result["plots_error"] = str(exc)

    if not skip_viz:
        try:
            from domains.highway.viz import visualize_best_policy

            viz = visualize_best_policy(
                run_dir,
                output_dir=os.path.join(plots_dir, "viz"),
                episodes=int(episodes),
                seed=int(seed),
            )
            result["viz"] = viz
            if viz.get("ok"):
                logger.info(
                    "Highway viz ok candidate=%s files=%d",
                    viz.get("candidate_id"),
                    len(viz.get("written") or []),
                )
            else:
                logger.warning("Highway viz skipped: %s", viz.get("reason"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("highway visualize failed: %s", exc)
            result["viz_error"] = str(exc)

    result["ok"] = bool(result["plots"]) or bool((result.get("viz") or {}).get("ok"))
    return result
