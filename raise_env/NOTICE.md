# NOTICE

## Vendored: vita-epfl/CrowdNav simulator

`third_party/crowdnav/` contains the `crowd_sim` package and the default
configuration files of

**CrowdNav**, https://github.com/vita-epfl/CrowdNav, commit
`20d678085c06831e658a65b9e20c8bb6f6ecdc10`, MIT License,
Copyright (c) 2018 VITA lab at EPFL (see `third_party/crowdnav/LICENSE`).

The files are byte-identical to that commit; see
`third_party/crowdnav/UPSTREAM.md` for what is vendored and what RAISE
re-implements. Please cite:

> Chen, Changan; Liu, Yuejiang; Kreiss, Sven; Alahi, Alexandre.
> **"Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based
> Deep Reinforcement Learning."** *IEEE International Conference on Robotics
> and Automation (ICRA)*, 2019. https://arxiv.org/abs/1809.08835

## Runtime dependency: Python-RVO2

The simulated humans use ORCA through Python-RVO2
(https://github.com/sybrenstuvel/Python-RVO2, Apache License 2.0), installed
from source at commit `c2c46ba8d59556aa10faf03479293236efea154d`; it is not
vendored.

## History

This directory started as a local copy of the MIT-licensed
**CrowdNav_Prediction_AttnGraph** repository
(https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph; Liu et al.,
"Intention Aware Robot Crowd Navigation with Attention-Based Interaction
Graph", ICRA 2023). That code was removed earlier; none of it remains.
RAISE later targeted HighwayEnv and now targets the vita-epfl CrowdNav
simulator above.
