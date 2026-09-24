"""Highway domain pack — LLM reward prompts for highway-fast-v0."""

from __future__ import annotations

from typing import Optional

D1_SYSTEM_PROMPT = (
    "You are an expert in reinforcement learning and autonomous highway driving. "
    "Your goal is to design reward functions that keep the ego vehicle safe, "
    "on-road, and making forward progress at a reasonable speed. "
    "Return **only** valid Python code enclosed within a fenced code block. "
    "The code must be fully executable and should not include comments or "
    "explanations outside the block."
)

D1_USER_PROMPT = """Please write a Python function named {func_name} for highway driving (highway-fast-v0).
Task Description:
- Output a scalar reward from the ego vehicle's current state so a policy learns to drive forward safely without collisions or leaving the road.
Function Interface (EXACT FIELD STRUCTURE):
- Inputs:
  - state: A HighwayRewardState snapshot with ONLY these fields:
    - state.ego.x, state.ego.y, state.ego.vx, state.ego.vy, state.ego.heading
    - state.ego.speed, state.ego.lane_index, state.ego.on_road
    - state.others: tuple of NearbyVehicle; iterate `for v in state.others:` then v.x, v.y, v.vx, v.vy, v.heading
    - state.collision, state.off_road, state.timeout: bool (environment-owned)
    - state.action, state.time_step, state.global_time, state.time_limit
    - state.progress: float (forward meters since previous frame)
    - state.speed: float (ego speed, same as state.ego.speed)
  - memory: plain mutable dict cleared at every episode reset(); use for shaping (e.g. memory['prev_speed']).
  - CRITICAL: There is NO state.robot, NO state.humans, NO state.gx/gy, NO state.history — only fields listed above.
  - Do NOT define classes. Persistent state must use the ``memory`` dict only.
- Output:
  - A single finite float reward for the current frame.
Design Principles:
- Progress / speed: reward forward motion and reasonable high speed.
- Safety: heavily penalize collision and off-road; shape clearance to nearby vehicles.
- Interpretability: clear local variables; no extra signature args.
Episode memory example:
```python
def compute_reward(state, memory):
    prev = memory.get("prev_x")
    progress = 0.0 if prev is None else float(state.ego.x) - float(prev)
    memory["prev_x"] = float(state.ego.x)
    return float(progress)
```
Math:
- Prefer `** 0.5` for square roots. Do not write `import math`.
- Do not use getattr, hasattr, or __import__.
Constraints (CRITICAL):
- Hyperparameters must be local variables inside the function body.
- Signature: exactly one function def {func_name}(state, memory): returning a finite float.
- Sandbox: no import/from, no classes, no while loops, no reflection builtins.
- Access fields only via dot notation. Iterate others with `for v in state.others:`.
- Output Format: only the Python function in a single code block.
{seed_block}
{reflection_block}
{external_knowledge_block}
"""

PROMPT_MEMORY_EXAMPLE = '''def compute_reward(state, memory):
    prev = memory.get("prev_x")
    progress = 0.0 if prev is None else float(state.ego.x) - float(prev)
    memory["prev_x"] = float(state.ego.x)
    return float(progress)
'''

_REWARD_STATE_ACCESS = """\
HighwayRewardState access (dot notation only — never getattr/hasattr/__import__):
- state.ego.x, state.ego.y, state.ego.vx, state.ego.vy, state.ego.heading, state.ego.speed, state.ego.lane_index, state.ego.on_road
- state.others — loop `for v in state.others:` then v.x, v.y, v.vx, v.vy, v.heading
- state.collision, state.off_road, state.timeout
- state.action, state.time_step, state.global_time, state.time_limit, state.progress, state.speed
- memory: plain dict for episode-local state; cleared on reset
- Math: no import math; use ** 0.5. No getattr/hasattr/__import__.
"""

_D2_SANDBOX_RULES = """
Sandbox rules (CRITICAL — invalid code is discarded):
- Define exactly ONE top-level function `{func_name}(state, memory)` returning a finite float.
- Do NOT use import/from, classes, while loops, print, lambda, or reflection builtins.
- Forbidden: getattr, hasattr, __import__, eval, exec, type, setattr, delattr, globals, locals, vars, open.
- Access HighwayRewardState only via dot notation (see below).
- Use ``memory`` (dict) for cross-timestep shaping; never invent state.history.
{reward_state_access}
- Hyperparameters as locals only; no extra args beyond (state, memory).
- Return ONLY one Python fenced code block with no text outside it.
"""

D2_CROSSOVER_PROMPT = """You are a reward function architect for highway driving. Synthesize a new function combining complementary strengths of two parents while addressing the reflection.
Parent A:
- {code_A}
Parent B:
- {code_B}
Reflection:
- {reflection}.
Synthesis Task:
- Write an improved `{func_name}` that merges safety and progress terms.
- Strip any getattr/hasattr patterns; use direct state.* / v.* access.
- Define exactly one function: def {func_name}(state, memory): ... returning a finite float.
{sandbox_rules}
- Return only a single Python fenced code block.
"""

D2_MUTATION_PROMPT = """You are a reward function optimizer for highway driving. Mutate the underperforming parent to address the weakness below with minimal edits.
Prior Reflection:
- {reflection}
Underperforming Parent Code to Mutate:
- {func_signature}
- {parent_code}
Mutation Task:
- Create a mutated `{func_name}` with a small precise change.
- Keep direct dot access; no getattr/hasattr.
- Define exactly one function: def {func_name}(state, memory): ... returning a finite float.
{sandbox_rules}
- Return only a single Python fenced code block.
"""

D3_SYSTEM_PROMPT = (
    "You are a senior researcher in autonomous driving and RL. "
    "Rewrite the reward to produce a smooth, dense, numerically stable per-frame "
    "signal that differentiates safe fast driving from collisions and off-road. "
    "IMPORTANT SCHEMA: def compute_reward(state, memory): — memory is a plain dict. "
    "state.ego has .x .y .vx .vy .heading .speed .lane_index .on_road. "
    "state.others is a tuple; iterate with 'for v in state.others:'. "
    "state.collision / state.off_road / state.timeout (bool). "
    "state.progress and state.speed are available. "
    "NO state.robot, NO state.humans, NO state.gx. "
    "SANDBOX: never getattr/hasattr/__import__/eval; use ** 0.5; always return a finite float. "
    "Output only valid Python code (no markdown fences or commentary)."
)

D3_USER_PROMPT = """Current score (best so far): {last_score:.4f} (higher is better)
Core components:
- Forward progress / speed shaping
- Collision and off-road penalties
- Clearance to nearby vehicles
- Stability (bounded magnitudes)
Focus note: {feedback}
{extra_context_if_any}
Revise the function below.
Maintain signature def compute_reward(state, memory): and return a finite float.
{current_code}
"""

D4_EXTERNAL_KNOWLEDGE = """# External Knowledge — Highway Fast
## Task
- Domain: multi-lane highway driving (highway-fast-v0)
- Ego must survive the episode without collision/off-road while making forward progress
## Metrics (mapped to RAISE ProxyMetrics)
- SR: fraction of episodes completed without collision
- CR: collision rate
- TR: off-road / other early failure rate (when distinguished); else 0
- Primary scalar: SR - CR - 0.5*TR
"""

D5_SEED_FUNCTION = '''def compute_reward(state, memory):
    """Highway seed: progress + speed with collision/off-road penalties."""
    collision_penalty = -20.0
    off_road_penalty = -10.0
    speed_coef = 0.05
    progress_coef = 1.0
    if state.collision:
        return float(collision_penalty)
    if state.off_road:
        return float(off_road_penalty)
    return float(progress_coef * state.progress + speed_coef * state.speed)
'''


def format_d1_initial(
    *,
    func_name: str = "compute_reward",
    include_seed: bool = True,
    include_external_knowledge: bool = True,
    reflection: str = "",
) -> str:
    seed_block = ""
    if include_seed:
        seed_block = (
            "Seed function (perturb / diversify; do not copy verbatim unless useful):\n"
            "```python\n"
            f"{D5_SEED_FUNCTION.rstrip()}\n"
            "```\n"
        )
    reflection_block = ""
    if reflection.strip():
        reflection_block = (
            f"Reflective guidance from prior generations:\n{reflection.strip()}\n"
        )
    external_block = ""
    if include_external_knowledge:
        external_block = f"External knowledge:\n{D4_EXTERNAL_KNOWLEDGE}\n"
    return D1_USER_PROMPT.format(
        func_name=func_name,
        seed_block=seed_block,
        reflection_block=reflection_block,
        external_knowledge_block=external_block,
    )


D1_BATCH_USER_PROMPT = """Please write **{n} diverse** Python reward functions for highway-fast-v0 driving.
Each function: def {func_name}_vK(state, memory) for K=1..{n}, returning a finite float.
Use HighwayRewardState fields only (state.ego.*, state.others, state.collision/off_road/timeout, state.progress, state.speed, memory).
No imports, classes, getattr/hasattr. Prefer ** 0.5 for roots.
{seed_block}
{reflection_block}
{external_knowledge_block}
Return only one fenced Python code block containing all {n} functions.
"""


def format_d1_initial_batch(
    n: int,
    *,
    func_name: str = "compute_reward",
    include_seed: bool = True,
    include_external_knowledge: bool = True,
    reflection: str = "",
) -> str:
    if n < 1:
        raise ValueError("batch size n must be >= 1")
    seed_block = ""
    if include_seed:
        seed_block = (
            "Seed function (perturb / diversify):\n```python\n"
            f"{D5_SEED_FUNCTION.rstrip()}\n```\n"
        )
    reflection_block = ""
    if reflection.strip():
        reflection_block = f"Reflection:\n{reflection.strip()}\n"
    external_block = ""
    if include_external_knowledge:
        external_block = f"External knowledge:\n{D4_EXTERNAL_KNOWLEDGE}\n"
    return D1_BATCH_USER_PROMPT.format(
        n=int(n),
        func_name=func_name,
        seed_block=seed_block,
        reflection_block=reflection_block,
        external_knowledge_block=external_block,
    )


def format_d2_crossover(
    code_a: str,
    code_b: str,
    reflection: str,
    *,
    func_name: str = "compute_reward",
) -> str:
    return D2_CROSSOVER_PROMPT.format(
        code_A=code_a.rstrip(),
        code_B=code_b.rstrip(),
        reflection=reflection.strip() or "(none)",
        func_name=func_name,
        sandbox_rules=_D2_SANDBOX_RULES.format(
            func_name=func_name,
            reward_state_access=_REWARD_STATE_ACCESS,
        ),
    )


def format_d2_mutation(
    parent_code: str,
    reflection: str,
    *,
    func_name: str = "compute_reward",
    func_signature: str = "def compute_reward(state, memory):",
    elitist_code: Optional[str] = None,
) -> str:
    code = parent_code if elitist_code is None else elitist_code
    return D2_MUTATION_PROMPT.format(
        reflection=reflection.strip() or "(none)",
        func_signature=func_signature,
        parent_code=code.rstrip(),
        func_name=func_name,
        sandbox_rules=_D2_SANDBOX_RULES.format(
            func_name=func_name,
            reward_state_access=_REWARD_STATE_ACCESS,
        ),
    )


def format_d3_refinement(
    current_code: str,
    *,
    last_score: float,
    feedback: str = "",
    extra_context_if_any: str = "",
) -> str:
    return D3_USER_PROMPT.format(
        last_score=float(last_score),
        feedback=feedback.strip() or "(none)",
        extra_context_if_any=extra_context_if_any,
        current_code=current_code.rstrip(),
    )


def format_d3_repair(
    *,
    bad_code: str,
    validation_error: str,
) -> str:
    return (
        "The following highway reward failed validation:\n\n"
        f"ERROR: {validation_error}\n\n"
        f"ORIGINAL CODE:\n{bad_code}\n\n"
        "Fix it. Keep signature def compute_reward(state, memory):. "
        "Use only HighwayRewardState fields (state.ego.*, state.others, "
        "state.collision/off_road/timeout, state.progress, state.speed, memory). "
        "No imports, getattr, hasattr. Always return a finite float. "
        "Output only the fixed function (no markdown)."
    )
