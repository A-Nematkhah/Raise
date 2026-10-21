# NOTICE

## Historical simulator: vita-epfl/CrowdNav

**CrowdNav**, https://github.com/vita-epfl/CrowdNav, commit
`20d678085c06831e658a65b9e20c8bb6f6ecdc10`, MIT License,
Copyright (c) 2018 VITA lab at EPFL. Earlier revisions included a minimal
vendor of this simulator; those files were removed after the CrowdNav++
wrapper migration and remain only in Git history. Please cite:

> Chen, Changan; Liu, Yuejiang; Kreiss, Sven; Alahi, Alexandre.
> **"Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based
> Deep Reinforcement Learning."** *IEEE International Conference on Robotics
> and Automation (ICRA)*, 2019. https://arxiv.org/abs/1809.08835

The active simulator is CrowdNav++ in `third_party/crowdnavpp/`.
`raise_paths.py` guards against importing the legacy package and exposes only
the active CrowdNav++ `crowd_sim` package.

## Vendored: CrowdNav++

`third_party/crowdnavpp/` contains the unmodified CrowdNav++ variable-human
simulator (`crowd_sim/`) and only the Python configuration and policy modules
required to import and run it. The upstream repository is
**CrowdNav_Prediction_AttnGraph**,
https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph, commit
`390773137be04ed14e27620dc5fd7c5e1a5b1f62`, MIT License, Copyright (c) 2023
Shuijing Liu (see `third_party/crowdnavpp/LICENSE` and
`third_party/crowdnavpp/NOTICE.md`).

No `rl/`, `pytorch_a2c_ppo_acktr_gail/`, GST implementation, or prediction
weights are vendored. The simulator paper is:

> Liu, Shuijing; Chang, Changan; Liang, Yifan; Gao, Xinyu; Rehg, James M.;
> Huang, Zhe; Chakraborty, Neeloy; Hong, Kaiwen; Liang, Weihang; Livingston
> McPherson, D.; Geng, Junyi; Driggs-Campbell, Katherine. **"Intention Aware
> Robot Crowd Navigation with Attention-Based Interaction Graph."** *IEEE
> International Conference on Robotics and Automation (ICRA)*, 2023,
> 12015-12021.

## Runtime dependency: Python-RVO2

The simulated humans use ORCA through Python-RVO2
(https://github.com/sybrenstuvel/Python-RVO2, Apache License 2.0), installed
from source at commit `c2c46ba8d59556aa10faf03479293236efea154d`; it is not
vendored.

## History

This directory started as a local copy of the MIT-licensed
**CrowdNav_Prediction_AttnGraph** repository
(https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph); only the
minimal simulator runtime described above is vendored again under
`third_party/crowdnavpp/`. Earlier revisions also used the vita-epfl CrowdNav
simulator; the only supported target today is CrowdNav++ in this tree.
