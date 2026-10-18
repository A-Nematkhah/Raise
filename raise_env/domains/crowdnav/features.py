"""Feature extraction for CrowdNav++ variable-human Dict observations."""

from __future__ import annotations

from typing import Dict

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class CrowdNavFeaturesExtractor(BaseFeaturesExtractor):
    """Encode robot context and masked-pool the visible human positions."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 128,
        human_embedding_dim: int = 32,
    ) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("CrowdNavFeaturesExtractor requires a gymnasium Dict space")
        required = {
            "robot_node",
            "temporal_edges",
            "visible_masks",
            "spatial_edges",
            "detected_human_num",
        }
        missing = required.difference(observation_space.spaces)
        if missing:
            raise ValueError(f"CrowdNav observation space is missing keys: {sorted(missing)}")

        robot_shape = observation_space["robot_node"].shape
        temporal_shape = observation_space["temporal_edges"].shape
        human_shape = observation_space["spatial_edges"].shape
        mask_shape = observation_space["visible_masks"].shape
        count_shape = observation_space["detected_human_num"].shape
        if len(robot_shape) != 1 or len(temporal_shape) != 1:
            raise ValueError("robot_node and temporal_edges must be one-dimensional")
        if len(human_shape) != 2 or human_shape[1] != 2:
            raise ValueError("spatial_edges must have shape (max_humans, 2)")
        if mask_shape != (human_shape[0],) or count_shape not in ((), (1,)):
            raise ValueError("visible_masks and detected_human_num have incompatible shapes")
        if features_dim < 1 or human_embedding_dim < 1:
            raise ValueError("feature dimensions must be positive")

        super().__init__(observation_space, features_dim)
        context_dim = robot_shape[0] + temporal_shape[0] + 1
        self.human_encoder = nn.Sequential(
            nn.Linear(2, human_embedding_dim),
            nn.Tanh(),
            nn.Linear(human_embedding_dim, human_embedding_dim),
            nn.Tanh(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.Tanh(),
        )
        self.combiner = nn.Sequential(
            nn.Linear(64 + human_embedding_dim, features_dim),
            nn.Tanh(),
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        robot = observations["robot_node"].float()
        temporal = observations["temporal_edges"].float()
        detected = observations["detected_human_num"].float().reshape(robot.shape[0], -1)
        context = self.context_encoder(th.cat((robot, temporal, detected), dim=1))

        human_embeddings = self.human_encoder(observations["spatial_edges"].float())
        visible = observations["visible_masks"].to(dtype=human_embeddings.dtype).unsqueeze(-1)
        pooled_humans = (human_embeddings * visible).sum(dim=1) / visible.sum(dim=1).clamp_min(1.0)
        return self.combiner(th.cat((context, pooled_humans), dim=1))


__all__ = ["CrowdNavFeaturesExtractor"]
