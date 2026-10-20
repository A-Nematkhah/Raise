from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from domains.crowdnav.adapter import CrowdNavPPOTrainer
from domains.crowdnav.features import CrowdNavFeaturesExtractor
from domains.crowdnav.prompts import D5_SEED_FUNCTION
from domains.crowdnav.reward_checks import make_crowdnav_validator
from domains.crowdnav.settings import spaces_compatible
from raise_core.candidate import RewardCandidate


def _observation_space(humans: int = 3, path_points: int = 1) -> spaces.Dict:
    return spaces.Dict(
        {
            "robot_node": spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32),
            "temporal_edges": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            "visible_masks": spaces.Box(
                np.zeros(humans, dtype=np.bool_),
                np.ones(humans, dtype=np.bool_),
                dtype=np.bool_,
            ),
            "spatial_edges": spaces.Box(
                -np.inf, np.inf, shape=(humans, 2 * path_points), dtype=np.float32
            ),
            "detected_human_num": spaces.Box(0, humans, shape=(), dtype=np.int64),
        }
    )


def test_crowdnav_extractor_masks_invisible_humans_and_handles_empty_mask():
    extractor = CrowdNavFeaturesExtractor(_observation_space())
    extractor.eval()
    observations = {
        "robot_node": th.zeros((1, 7)),
        "temporal_edges": th.zeros((1, 2)),
        "visible_masks": th.tensor([[True, False, True]]),
        "spatial_edges": th.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]]),
        "detected_human_num": th.tensor([2]),
    }
    changed_invisible = dict(observations)
    changed_invisible["spatial_edges"] = observations["spatial_edges"].clone()
    changed_invisible["spatial_edges"][0, 1] = th.tensor([1000.0, -1000.0])

    features = extractor(observations)
    changed_features = extractor(changed_invisible)
    assert features.shape == (1, 128)
    assert th.isfinite(features).all()
    assert th.allclose(features, changed_features)

    no_visible = dict(observations)
    no_visible["visible_masks"] = th.zeros((1, 3), dtype=th.bool)
    assert th.isfinite(extractor(no_visible)).all()


def test_crowdnav_extractor_accepts_predicted_human_paths():
    extractor = CrowdNavFeaturesExtractor(_observation_space(path_points=6))
    observations = {
        "robot_node": th.zeros((1, 7)),
        "temporal_edges": th.zeros((1, 2)),
        "visible_masks": th.tensor([[True, False, True]]),
        "spatial_edges": th.zeros((1, 3, 12)),
        "detected_human_num": th.tensor([2]),
    }
    features = extractor(observations)
    assert features.shape == (1, 128)
    assert th.isfinite(features).all()


def test_spaces_compatible_checks_dict_keys_and_nested_shapes():
    assert spaces_compatible(_observation_space(), _observation_space())
    assert not spaces_compatible(_observation_space(3), _observation_space(4))
    assert not spaces_compatible(
        _observation_space(),
        spaces.Dict(
            {
                **_observation_space().spaces,
                "extra": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            }
        ),
    )


@pytest.mark.slow
def test_real_ppo_trains_and_reevaluates_crowdnav_dict_observations(tmp_path):
    reward_fn = make_crowdnav_validator().validate_code(D5_SEED_FUNCTION)
    assert reward_fn is not None
    candidate = RewardCandidate(
        candidate_id="dict_ppo_smoke",
        code=D5_SEED_FUNCTION,
        reward_fn=reward_fn,
        valid=True,
    )
    config = SimpleNamespace(
        train_env_steps=256,
        eval_episodes=1,
        seed=7,
        n_envs=1,
        eval_mode="holdout_only",
        eval_split="val",
        eval_deterministic=True,
        eval_seed_base=7,
        device="cpu",
        output_root=str(tmp_path),
        warm_start=False,
        ppo_n_steps=32,
        ppo_n_epochs=1,
        ppo_ent_coef=0.0,
        record_training_dynamics=False,
        action_mode="continuous",
        robot_kinematics="holonomic",
        action_speed_samples=5,
        action_rotation_samples=16,
        env_config_path=None,
        human_num_override=4,
    )
    trainer = CrowdNavPPOTrainer(stage="stage2")
    metrics = trainer.train_and_eval(candidate, round_index=0, config=config)

    checkpoint = candidate.metadata["checkpoint_path"]
    assert candidate.metadata["actual_train_steps"] >= 256
    assert checkpoint.endswith("model.zip")
    assert metrics is not None
    assert candidate.metadata["last_metrics"]["eval_context"]["scenario"] == "val"
    reevaluated = trainer.reevaluate(candidate, round_index=0, config=config)
    assert reevaluated is not None
