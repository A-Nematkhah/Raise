"""Short smoke: build env + SB3 PPO for each highway action_mode."""

from __future__ import annotations

import os
import sys
import tempfile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import raise_paths  # noqa: E402,F401

import numpy as np

from domains.highway.action_config import ActionModeSettings, set_action_settings
from domains.highway.env_wrapper import (
    RewardInjectedHighwayEnv,
    make_base_env,
    training_env_config,
)
from domains.highway.prompts import D5_SEED_FUNCTION
from raise_core.domains import load_domain, make_validator_for_domain


def _smoke_mode(mode: str, *, lateral: bool = True, steps: int = 2048) -> dict:
    set_action_settings(
        ActionModeSettings(mode=mode, continuous_lateral=lateral)
    )
    pack = load_domain("highway")
    reward = make_validator_for_domain(pack).validate_code(D5_SEED_FUNCTION)
    cfg = training_env_config()
    env = make_base_env(seed=0, config=cfg)
    info = {
        "mode": mode,
        "lateral": lateral,
        "action_space": str(env.action_space),
        "action_cfg": dict(env.unwrapped.config["action"]),
    }
    env.close()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _thunk():
        return RewardInjectedHighwayEnv(reward, seed=1, config=cfg)

    venv = DummyVecEnv([_thunk])
    model = PPO(
        "MlpPolicy",
        venv,
        verbose=0,
        n_steps=64,
        batch_size=64,
        seed=0,
        device="cpu",
    )
    with tempfile.TemporaryDirectory() as td:
        from domains.highway.diagnostics_callback import RolloutDiagnosticsCallback

        log_path = os.path.join(td, "diagnostics_rollout.jsonl")
        cb = RolloutDiagnosticsCallback(
            candidate_id="smoke",
            log_path=log_path,
            continuous_actions=(mode == "continuous"),
        )
        model.learn(total_timesteps=steps, callback=cb)
        info["diag_bytes"] = os.path.getsize(log_path) if os.path.isfile(log_path) else 0
        if mode == "continuous":
            # Collect observed speeds under random-ish deterministic zeros then accel.
            speeds = []
            e2 = RewardInjectedHighwayEnv(reward, seed=2, config=cfg)
            obs, _ = e2.reset(seed=2)
            for t in range(30):
                if lateral:
                    a = np.array([0.8, 0.0], dtype=np.float32)
                else:
                    a = np.array([0.8], dtype=np.float32)
                # Use model predict for continuous Box
                act, _ = model.predict(obs, deterministic=False)
                obs, _r, te, tr, info_step = e2.step(act)
                speeds.append(float(info_step.get("raise_speed", 0.0)))
                if te or tr:
                    break
            e2.close()
            info["speed_min"] = float(min(speeds)) if speeds else None
            info["speed_max"] = float(max(speeds)) if speeds else None
            info["n_unique_round1"] = len({round(s, 1) for s in speeds})
    venv.close()
    return info


def main() -> int:
    results = []
    results.append(_smoke_mode("meta_default", steps=1024))
    results.append(_smoke_mode("meta_fine", steps=1024))
    results.append(_smoke_mode("continuous", lateral=True, steps=2048))
    results.append(_smoke_mode("continuous", lateral=False, steps=2048))
    set_action_settings(ActionModeSettings(mode="meta_default"))
    for r in results:
        print(r)
        assert r["diag_bytes"] > 0, r
        if r["mode"] == "continuous":
            assert "target_speeds" not in r["action_cfg"], r
            assert r["n_unique_round1"] >= 2 or (
                r["speed_max"] is not None
                and r["speed_min"] is not None
                and (r["speed_max"] - r["speed_min"]) > 0.5
            ), r
        else:
            assert r["action_cfg"]["type"] == "DiscreteMetaAction", r
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
