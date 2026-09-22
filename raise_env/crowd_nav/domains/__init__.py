"""
Pluggable domain packs for reward search.

Default domain is ``crowdnav`` (RAISE on CrowdNav++). Other
environments are added as sibling packages under this package, each exposing
``get_pack() -> DomainPack``.
"""

from __future__ import annotations

from typing import Any, List, Optional

from crowd_nav.domains.base import DomainPack, EnvAdapter

DEFAULT_DOMAIN = "crowdnav"

_REGISTRY = {
    "crowdnav": "crowd_nav.domains.crowdnav",
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
    import importlib

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
    """
    Resolve Stage I score_fn via the domain pack.

    Uses ``pack.make_score_fn`` when set; CrowdNav returns ``(score_fn, dataset)``
    — this helper returns only ``score_fn`` for the pipeline.
    """
    path = dataset_path
    if path is None or not str(path).strip():
        path = pack.stage1_dataset_default

    if pack.make_score_fn is not None:
        result = pack.make_score_fn(mode=mode, dataset_path=path)
        if isinstance(result, tuple):
            return result[0]
        return result

    if pack.name == "crowdnav":
        from crowd_nav.domains.crowdnav.explore_score import make_score_fn

        score_fn, _dataset = make_score_fn(mode=mode, dataset_path=path)
        return score_fn

    raise NotImplementedError(
        f"Domain {pack.name!r} has no Stage I make_score_fn yet"
    )


def make_stage2_trainer_for_domain(
    pack: DomainPack, *, use_stub: bool = False
) -> Any:
    """Resolve Stage II PolicyTrainer via the domain pack (CrowdNav today)."""
    if pack.name == "crowdnav":
        from crowd_nav.domains.crowdnav.adapter import make_stage2_trainer

        return make_stage2_trainer(use_stub=use_stub)
    raise NotImplementedError(
        f"Domain {pack.name!r} has no Stage II trainer factory yet"
    )


def make_stage3_trainer_for_domain(
    pack: DomainPack, *, use_stub: bool = False
) -> Any:
    """Resolve Stage III PolicyTrainer via the domain pack (CrowdNav today)."""
    if pack.name == "crowdnav":
        from crowd_nav.domains.crowdnav.adapter import make_stage3_trainer

        return make_stage3_trainer(use_stub=use_stub)
    raise NotImplementedError(
        f"Domain {pack.name!r} has no Stage III trainer factory yet"
    )


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
    "register_domain",
]
