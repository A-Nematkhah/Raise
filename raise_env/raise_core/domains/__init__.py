"""
CrowdNav++ domain pack wiring for RAISE.

``raise_core`` loads the simulation pack only through ``get_pack()``; CrowdNav
implementation lives under ``domains.crowdnav``.
"""

from __future__ import annotations

from typing import Any, Optional

from raise_core.domains.base import DomainPack, EnvAdapter
from raise_core.sandbox.validator import RewardValidator


def get_pack(*, with_adapter: bool = True) -> DomainPack:
    """Return the CrowdNav++ ``DomainPack`` (only environment in this repo)."""
    from domains.crowdnav.pack import get_pack as _crowdnav_get_pack

    pack = _crowdnav_get_pack(with_adapter=with_adapter)
    if not isinstance(pack, DomainPack):
        raise TypeError("domains.crowdnav.pack.get_pack() must return DomainPack")
    return pack


def make_score_fn(
    pack: DomainPack,
    *,
    mode: str = "dataset",
    dataset_path: Optional[str] = None,
) -> Any:
    """Resolve the Stage I Score1 function (``pack.make_score_fn`` returns ``(fn, dataset)``)."""
    path = dataset_path
    if path is None or not str(path).strip():
        path = pack.stage1_dataset_default
    if pack.make_score_fn is None:
        raise NotImplementedError(f"Domain pack {pack.name!r} has no make_score_fn")
    result = pack.make_score_fn(mode=mode, dataset_path=path)
    if isinstance(result, tuple):
        return result[0]
    return result


def make_stage2_trainer(pack: DomainPack, *, use_stub: bool = False) -> Any:
    """Resolve the Stage II trainer from the pack."""
    if pack.make_stage2_trainer is None:
        raise NotImplementedError(f"Domain pack {pack.name!r} has no Stage II trainer")
    return pack.make_stage2_trainer(use_stub=use_stub)


def make_stage3_trainer(pack: DomainPack, *, use_stub: bool = False) -> Any:
    """Resolve the Stage III trainer from the pack."""
    if pack.make_stage3_trainer is None:
        raise NotImplementedError(f"Domain pack {pack.name!r} has no Stage III trainer")
    return pack.make_stage3_trainer(use_stub=use_stub)


def make_validator(pack: DomainPack) -> Any:
    """Build the reward validator with the pack's smoke states / sandbox config."""
    smoke = pack.smoke_states_fn() if pack.smoke_states_fn is not None else ()
    if pack.make_validator is not None:
        return pack.make_validator(config=pack.sandbox_config, smoke_states=smoke)
    return RewardValidator(config=pack.sandbox_config, smoke_states=smoke)


__all__ = [
    "DomainPack",
    "EnvAdapter",
    "get_pack",
    "make_score_fn",
    "make_stage2_trainer",
    "make_stage3_trainer",
    "make_validator",
]
