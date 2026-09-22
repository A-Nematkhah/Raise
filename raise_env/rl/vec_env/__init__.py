"""Minimal vec_env surface used by EvoNav / CrowdNav++.

Active code imports ``VecPretextNormalize`` (and its ``VecEnvWrapper`` /
``RunningMeanStd`` dependencies). OpenAI Baselines supplies the live
``DummyVecEnv`` / ``ShmemVecEnv`` wrappers via ``rl.networks``.
"""

from .vec_env import (
    AlreadySteppingError,
    CloudpickleWrapper,
    NotSteppingError,
    VecEnv,
    VecEnvObservationWrapper,
    VecEnvWrapper,
)

__all__ = [
    "AlreadySteppingError",
    "NotSteppingError",
    "VecEnv",
    "VecEnvWrapper",
    "VecEnvObservationWrapper",
    "CloudpickleWrapper",
]
