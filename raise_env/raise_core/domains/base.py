"""
Domain pack contract for pluggable simulation / training backends.

A Domain Pack is a folder under ``raise_core.domains`` that supplies everything
the reward-search algorithm needs for one environment family. The CrowdNav
pack is the default and must preserve Algorithm 1 baseline behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class EnvAdapter(Protocol):
    """
    Thin training / eval surface used by Stage II/III.

    CrowdNav's concrete adapter currently delegates to the existing
    ``RealPolicyTrainer`` / ``StubPolicyTrainer`` implementations; other
    domains will implement this without importing CrowdSim.
    """

    def train_and_eval(
        self,
        candidate: Any,
        *,
        round_index: int,
        config: Any,
        stage: str = "stage2",
    ) -> Any:
        """Train under the candidate reward and return stage metrics / bundle."""
        ...


@dataclass(frozen=True)
class DomainPack:
    """
    Loaded environment pack.

    Required today: identity, prompts, seed reward, spec path, Stage I
    ``make_score_fn``, and EnvAdapter for Stage II/III.
    """

    name: str
    display_name: str
    prompts: ModuleType
    seed_reward_source: str
    spec_path: str
    description: str = ""
    # Optional hooks (filled as phases land; None = use legacy reward_search path)
    adapter: Optional[EnvAdapter] = None
    stage1_dataset_default: Optional[str] = None
    make_score_fn: Optional[Callable[..., Any]] = None
    # Optional factory: () -> Sequence[smoke state objects] for RewardValidator.
    smoke_states_fn: Optional[Callable[[], Sequence[Any]]] = None
    # Pack-local Stage II/III trainer factories (preferred over name branches).
    make_stage2_trainer: Optional[Callable[..., Any]] = None
    make_stage3_trainer: Optional[Callable[..., Any]] = None
    selection_scalar_name: str = "navigation_scalar"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def format_d1_initial(self, **kwargs: Any) -> str:
        return self.prompts.format_d1_initial(**kwargs)

    def format_d1_initial_batch(self, n: int, **kwargs: Any) -> str:
        return self.prompts.format_d1_initial_batch(n, **kwargs)

    def format_d2_crossover(self, *args: Any, **kwargs: Any) -> str:
        return self.prompts.format_d2_crossover(*args, **kwargs)

    def format_d2_mutation(self, *args: Any, **kwargs: Any) -> str:
        return self.prompts.format_d2_mutation(*args, **kwargs)

    def format_d3_refinement(self, *args: Any, **kwargs: Any) -> str:
        return self.prompts.format_d3_refinement(*args, **kwargs)

    def format_d3_repair(self, **kwargs: Any) -> str:
        return self.prompts.format_d3_repair(**kwargs)

    @property
    def d1_system_prompt(self) -> str:
        return self.prompts.D1_SYSTEM_PROMPT

    @property
    def d3_system_prompt(self) -> str:
        return self.prompts.D3_SYSTEM_PROMPT

    @property
    def d5_seed_function(self) -> str:
        return self.seed_reward_source
