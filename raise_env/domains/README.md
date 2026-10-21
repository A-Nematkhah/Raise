# CrowdNav++ domain pack

RAISE targets crowd-aware robot navigation in the vendored CrowdNav++
`CrowdSimVarNum` simulator. Implementation lives in `domains/crowdnav/` (see
[`crowdnav/spec.md`](crowdnav/spec.md) for the reward state contract, Score1,
and fitness). The algorithm loads this pack through `raise_core.domains.get_pack()`.
