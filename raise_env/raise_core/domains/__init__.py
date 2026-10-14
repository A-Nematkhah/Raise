"""
Domain pack registry for reward search.

Each environment family registers a module (``domains/<name>/pack.py``)
exposing ``get_pack() -> DomainPack``. The only registered domain is ``crowdnav``.
"""

from __future__ import annotations

import importlib
from typing import Any, List, Optional

from raise_core.domains.base import DomainPack, EnvAdapter
from raise_core.sandbox.validator import RewardValidator

DEFAULT_DOMAIN = "crowdnav"

_REGISTRY = {
    "crowdnav": "domains.crowdnav.pack",
}


def available_domains() -> List[str]:
    """Registered domain names (sorted)."""
    return sorted(_REGISTRY.keys())


def load_domain(name: str = DEFAULT_DOMAIN, *, with_adapter: bool = True) -> DomainPack:
    """
    Load a domain pack by name.

    Raises ``KeyError`` if ``name`` is not registered.
    """
    key = str(name).strip().lower()
    if key not in _REGISTRY:
        known = ", ".join(available_domains()) or "(none)"
        raise KeyError(f"Unknown domain {name!r}. Registered: {known}")
    module_path = _REGISTRY[key]
    mod = importlib.import_module(module_path)
    get_pack = getattr(mod, "get_pack", None)
    if get_pack is None:
        raise ImportError(f"Domain module {module_path!r} has no get_pack()")
    pack = get_pack(with_adapter=with_adapter)
    if not isinstance(pack, DomainPack):
        raise TypeError(f"{module_path}.get_pack() must return DomainPack")
    if pack.name != key:
        raise ValueError(
            f"Domain pack name mismatch: registry key={key!r}, pack.name={pack.name!r}"
        )
    return pack


def make_score_fn_for_domain(
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
        raise NotImplementedError(f"Domain {pack.name!r} has no make_score_fn")
    result = pack.make_score_fn(mode=mode, dataset_path=path)
    if isinstance(result, tuple):
        return result[0]
    return result


def make_stage2_trainer_for_domain(
    pack: DomainPack, *, use_stub: bool = False
) -> Any:
    """Resolve the Stage II trainer via the domain pack."""
    if pack.make_stage2_trainer is None:
        raise NotImplementedError(f"Domain {pack.name!r} has no Stage II trainer")
    return pack.make_stage2_trainer(use_stub=use_stub)


def make_stage3_trainer_for_domain(
    pack: DomainPack, *, use_stub: bool = False
) -> Any:
    """Resolve the Stage III trainer via the domain pack."""
    if pack.make_stage3_trainer is None:
        raise NotImplementedError(f"Domain {pack.name!r} has no Stage III trainer")
    return pack.make_stage3_trainer(use_stub=use_stub)


def make_validator_for_domain(pack: DomainPack) -> Any:
    """Build the reward validator with the pack's smoke states / sandbox config."""
    smoke = pack.smoke_states_fn() if pack.smoke_states_fn is not None else ()
    if pack.make_validator is not None:
        return pack.make_validator(config=pack.sandbox_config, smoke_states=smoke)
    return RewardValidator(config=pack.sandbox_config, smoke_states=smoke)


def register_domain(name: str, module_path: str) -> None:
    """Register an additional domain module path (tests / extensions)."""
    key = str(name).strip().lower()
    _REGISTRY[key] = module_path


__all__ = [
    "DEFAULT_DOMAIN",
    "DomainPack",
    "EnvAdapter",
    "available_domains",
    "load_domain",
    "make_score_fn_for_domain",
    "make_stage2_trainer_for_domain",
    "make_stage3_trainer_for_domain",
    "make_validator_for_domain",
    "register_domain",
]
