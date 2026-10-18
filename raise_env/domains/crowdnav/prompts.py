"""CrowdNav domain pack — LLM reward prompts for crowd-aware robot navigation."""

from __future__ import annotations

from domains.crowdnav.settings import action_space_prompt_block
from raise_core.sandbox.config import (
    SAFE_MATH_ATTRIBUTES,
    SAFE_MATH_CONSTANTS,
    SandboxConfig,
)

DOMAIN_NAME = "crowdnav"

_MAGNITUDE_RULE = (
    f"Keep every per-step total reward within +/-{SandboxConfig().max_abs_reward:g} "
    "(larger values are rejected)."
)

_MATH_RULE = (
    "Math: `math` is pre-injected - never write `import math`. Allowed: "
    + ", ".join(f"math.{n}" for n in SAFE_MATH_ATTRIBUTES if n not in SAFE_MATH_CONSTANTS)
    + "; constants "
    + ", ".join(f"math.{n}" for n in SAFE_MATH_CONSTANTS)
    + ". Any other math.* is rejected."
)


def _configured_env():
    from domains.crowdnav.env_wrapper import load_env_config
    from domains.crowdnav.settings import get_settings

    settings = get_settings()
    return load_env_config(settings.env_config_source), settings


def _task_description() -> str:
    config, settings = _configured_env()
    human_num = (
        int(settings.human_num_override)
        if settings.human_num_override is not None
        else int(config.sim.human_num)
    )
    human_num_range = (
        0
        if settings.human_num_override is not None
        else int(config.sim.human_num_range)
    )
    if settings.kinematics == "unicycle":
        human_num_range = max(human_num_range, human_num - 1)
    minimum = max(1, human_num - human_num_range)
    maximum = human_num + human_num_range
    fov_pi = float(config.robot.FOV)
    population_change = (
        "and may change during an episode"
        if human_num_range > 0
        else "and remains fixed during an episode"
    )
    return (
        "CrowdNav++ variable-human CrowdSim: a robot "
        f"(radius {float(config.robot.radius):g} m, preferred speed "
        f"{float(config.robot.v_pref):g} m/s, {settings.kinematics} kinematics) "
        "must reach the goal in the configured scenario. Human population starts "
        f"between {minimum} and {maximum} {population_change}; "
        f"humans use {config.humans.policy}. Human placement uses the configured "
        f"{float(config.sim.circle_radius):g} m circle radius. The robot's sensor "
        f"range is {float(config.robot.sensor_range):g} m and its field of view is "
        f"{fov_pi:g}π radians. The robot is "
        f"{'visible' if config.robot.visible else 'invisible'} to humans. Each step "
        f"lasts {float(config.env.time_step):g} s; the episode time limit is "
        f"{float(config.env.time_limit):g} s."
    )


def _selection_objective() -> str:
    config, _ = _configured_env()
    return (
        "Your reward function is judged by training a PPO policy with it and "
        "evaluating it on held-out CrowdNav++ validation scenarios. Candidates "
        "by mean per-episode outcome score: +1 for success, -1 for collision, "
        "and -0.5 for timeout (higher is better). Native CrowdNav++ reward and "
        "the reward function you write affect PPO training, but native task "
        "return is not the selection fitness. Prefer policies that reach the "
        "goal without collision rather than timing out."
    )


def _system_prompt() -> str:
    return (
        "You are an expert in reinforcement learning and crowd-aware robot navigation. "
        "Your goal is to design reward functions for the CrowdNav++ simulator. "
        + _selection_objective()
        + " "
        + _RETURN_CONTRACT
        + " "
        + "Put the code in exactly one ```python fenced block; any explanation must "
        "come before the fence. The code must be fully executable."
    )


def _external_knowledge() -> str:
    return (
        "# External Knowledge — CrowdNav++\n"
        "## Task\n- "
        + _task_description()
        + "\n## Selection objective (not the reward itself)\n"
        + _selection_objective()
        + "\n## Metrics (as in the CrowdNav++ evaluation)\n"
        "- SR: success rate (goal reached); CR: collision rate; TR: timeout rate\n"
        "- nav_time: mean navigation time of successful episodes (s); "
        "path_length: mean path length of successful episodes (m)\n"
        "- danger_freq: fraction of steps in danger; "
        "min_dist_in_danger: upstream mean distance during those steps (m)"
    )


def __getattr__(name: str) -> str:
    if name == "D1_SYSTEM_PROMPT":
        return _system_prompt()
    if name == "D4_EXTERNAL_KNOWLEDGE":
        return _external_knowledge()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_DIAGNOSIS_BEFORE_CODE = (
    "Before writing code:\n"
    "1) In 2-4 sentences, based ONLY on the evidence given above (not on any "
    "assumption about what 'usually' goes wrong), diagnose what is currently "
    "limiting the holdout fitness - e.g. collisions, a freezing robot that "
    "times out, slow or detouring arrivals, discomfort-zone intrusions, or "
    "something else you notice.\n"
    "2) Then revise the reward function to address your own diagnosis.\n"
    "- If the success rate is always near zero, a larger rewrite of the "
    "reward function is appropriate.\n"
    "Output your diagnosis as plain text BEFORE the code fence. The code fence "
    "must contain ONLY the function, no diagnosis text inside it.\n"
    "A targeted change can still mean removing or replacing a constant "
    "entirely if the evidence suggests it no longer reflects reality - don't "
    "assume an inherited number is correct just because it came from the "
    "parent."
)

# Token-budget fallback for ``_DIAGNOSIS_BEFORE_CODE`` (same contract, fewer words).
_DIAGNOSIS_BEFORE_CODE_COMPACT = (
    "Before the code fence, write a 2-4 sentence diagnosis grounded only in the "
    "evidence above, then revise the function to address it (a larger rewrite "
    "is fine if the success rate is near zero). The fence must contain ONLY "
    "the function."
)

_RETURN_CONTRACT = (
    "Return exactly ``(total_reward, reward_components)`` where "
    "``total_reward`` is a finite float used for RL, and "
    "``reward_components`` is a flat dict[str, float] of named terms for "
    "this step (e.g. {\"progress_term\": 0.2, \"goal_reward\": 0.0, "
    "\"collision_penalty\": 0.0}). At most 12 keys; string keys only; "
    "finite float values only; no nested containers. The dict is "
    "informational (diagnostics/reflection) and must not change what "
    "``total_reward`` already encodes."
)

_STATE_FIELDS = """\
  - state: a CrowdNavRewardState snapshot taken AFTER the step, with ONLY these fields:
    - state.robot.px, state.robot.py, state.robot.vx, state.robot.vy, state.robot.gx, state.robot.gy, state.robot.theta, state.robot.radius, state.robot.v_pref, state.robot.kinematics
    - state.humans: tuple of all humans; iterate `for h in state.humans:` then h.px, h.py, h.vx, h.vy, h.radius, h.visible
    - FRAMES: every position and velocity is in the WORLD frame (metres, m/s); (state.robot.gx, state.robot.gy) is the goal. Distances to humans must be computed, e.g. ((h.px - state.robot.px) ** 2 + (h.py - state.robot.py) ** 2) ** 0.5 is centre-to-centre; subtract h.radius + state.robot.radius for the gap between bodies.
    - state.collision, state.reached_goal, state.timeout, state.danger: bool flags from upstream CrowdNav++.
    - state.min_separation: float, the smallest current robot-human body gap; state.discomfort_dist is the configured comfort threshold.
    - state.goal_distance: float, robot-centre distance to the goal after the step; state.progress: float, decrease of goal_distance during this step (negative when moving away).
    - state.action (see the action space below), state.time_step (s), state.global_time (s since reset), state.time_limit (s)
    - state.success_reward, state.collision_penalty, state.discomfort_penalty_factor: configured upstream reward parameters.
    - state.potential_delta, state.potential_factor: upstream potential-shaping delta and its kinematics-dependent factor for this step.
  - memory: plain mutable dict cleared at every episode reset(); use for shaping (e.g. memory['prev_speed'])."""

D1_USER_PROMPT = """Please write a Python function named {func_name} for crowd-aware robot navigation (CrowdNav++).
Task Description:
- {task_description}
- Output ``(total_reward, reward_components)`` for the robot's current step so a policy learns to reach the goal quickly without collisions or intrusions under the selection objective above (total_reward trains RL; reward_components are for reflection only).
Function Interface (EXACT FIELD STRUCTURE):
- Inputs:
""" + _STATE_FIELDS + """
  - CRITICAL: There is NO state.self_state, NO state.human_states, NO .position / .velocity tuples, NO state.dmin, NO state.history - only the fields listed above.
  - Do NOT define classes. Persistent state must use the ``memory`` dict only.
- Output:
  - Return ``(total_reward, reward_components)`` - total_reward is the finite float used for RL; reward_components is a flat dict[str, float] naming the terms that sum (or otherwise compose) into that total for this step.
- Design Principles:
- Prefer dense, finite shaping from documented state fields.
- Interpretability: clear local variables; named components in the dict; no extra signature args.
- Reason about multi-objective trade-offs yourself from the selection objective; do not assume a named failure mode.
Episode memory example:
```python
def compute_reward(state, memory):
    prev = memory.get("prev_goal_distance")
    progress = 0.0 if prev is None else float(prev) - float(state.goal_distance)
    memory["prev_goal_distance"] = float(state.goal_distance)
    total = float(progress)
    return total, {{"progress_term": total}}
```
- """ + _MATH_RULE + """
- Do not use getattr, hasattr, or __import__.
Constraints (CRITICAL):
- Hyperparameters must be local variables inside the function body.
- No module-level lists/dicts/sets (they would survive episode resets); keep cross-step state in memory.
- """ + _MAGNITUDE_RULE + """
- Signature: exactly one function def {func_name}(state, memory): returning (finite float, dict[str, float]).
- Sandbox: no import/from, no classes, no while loops, no list/dict/set comprehensions or generator expressions, no reflection builtins.
- Access fields only via dot notation. Iterate humans with `for h in state.humans:`.
- AST allowlist: ONLY the fields listed above (+ memory.get / memory['k']). Any other
  attribute (self_state, human_states, position, velocity, dmin, distance, speed, history, …) is REJECTED.
Negative examples (DO NOT write these — they fail validation):
```python
# BAD — hallucinated fields
state.self_state.px   # use state.robot.px
state.human_states    # use state.humans
state.robot.position  # use (state.robot.px, state.robot.py)
state.dmin            # use state.min_separation
state.robot.speed     # compute (state.robot.vx ** 2 + state.robot.vy ** 2) ** 0.5
```
- Output Format: only the Python function in a single code block.
{action_space_block}
{seed_block}
{reflection_block}
{external_knowledge_block}
"""

PROMPT_MEMORY_EXAMPLE = '''def compute_reward(state, memory):
    prev = memory.get("prev_goal_distance")
    progress = 0.0 if prev is None else float(prev) - float(state.goal_distance)
    memory["prev_goal_distance"] = float(state.goal_distance)
    total = float(progress)
    return total, {"progress_term": total}
'''

_REWARD_STATE_ACCESS = """\
CrowdNavRewardState access (dot notation only — never getattr/hasattr/__import__); snapshot AFTER the step:
- state.robot.px, .py, .vx, .vy, .gx, .gy, .theta, .radius, .v_pref, .kinematics (world frame; (gx, gy) is the goal)
- state.humans — loop `for h in state.humans:` then h.px, h.py, h.vx, h.vy, h.radius, h.visible (world frame; includes all humans)
- state.collision, state.reached_goal, state.timeout (mutually exclusive outcome flags), state.danger (upstream danger classification)
- state.min_separation (smallest current robot-human body gap; negative on collision), state.discomfort_dist (configured)
- state.goal_distance (after the step), state.progress (decrease of goal_distance this step)
- state.action, state.time_step, state.global_time, state.time_limit
- state.success_reward, state.collision_penalty, state.discomfort_penalty_factor
- state.potential_delta, state.potential_factor (upstream potential-based shaping)
- memory: plain dict for episode-local state; cleared on reset
- FORBIDDEN (rejected by AST): self_state, human_states, position, velocity, dmin, distance, speed, history
- """ + _MATH_RULE + """
- No getattr/hasattr/__import__.
"""

_D2_SANDBOX_RULES = """
Sandbox rules (CRITICAL — invalid code is discarded):
- Define exactly ONE top-level function `{func_name}(state, memory)` returning
  ``(total_reward: float, reward_components: dict[str, float])``.
- ``total_reward`` is the only value used for RL; ``reward_components`` is a flat
  dict of named terms (≤12 str keys, finite float values, no nesting) for reflection.
- Do NOT use import/from, classes, while loops, print, comprehensions,
  generator expressions, or reflection builtins.
- Forbidden: getattr, hasattr, __import__, eval, exec, type, setattr, delattr, globals, locals, vars, open.
- Access CrowdNavRewardState only via dot notation (see below).
- Use ``memory`` (dict) for cross-timestep shaping; never invent state.history.
{reward_state_access}
- Hyperparameters as locals only; no extra args beyond (state, memory).
- No module-level lists/dicts/sets (they would survive episode resets).
- """ + _MAGNITUDE_RULE + """
- Put diagnosis as plain text BEFORE the fence; the fenced block must contain ONLY the function.
"""

D2_CROSSOVER_PROMPT = """You are a reward function architect for crowd-aware robot navigation (CrowdNav). Synthesize a new function combining complementary strengths of two parents while addressing the reflection.
Parent A:
- {code_A}
Parent B:
- {code_B}
Reflection / evidence:
- {reflection}.
Synthesis Task:
- Write an improved `{func_name}` that merges complementary terms from the parents.
- Strip any getattr/hasattr patterns; use direct state.* / h.* access.
- Define exactly one function: def {func_name}(state, memory): ... returning (float, dict[str, float]).
{sandbox_rules}
{diagnosis}
"""

D2_MUTATION_PROMPT = """You are a reward function optimizer for crowd-aware robot navigation (CrowdNav). Mutate the parent using the evidence below with a targeted change.
Prior Reflection / evidence:
- {reflection}
Parent Code to Mutate:
- {parent_block}
Mutation Task:
- Create a mutated `{func_name}` with a targeted change that addresses your diagnosis.
- Keep direct dot access; no getattr/hasattr.
- Define exactly one function: def {func_name}(state, memory): ... returning (float, dict[str, float]).
{sandbox_rules}
{diagnosis}
"""

D5_SEED_FUNCTION = '''def compute_reward(state, memory):
    """Reproduce the configured CrowdNav++ CrowdSim reward."""
    goal_reward = 0.0
    collision_term = 0.0
    discomfort_term = 0.0
    potential_term = 0.0
    spin_term = 0.0
    backwards_term = 0.0
    if not state.timeout:
        if state.collision:
            collision_term = state.collision_penalty
        elif state.reached_goal:
            goal_reward = state.success_reward
        elif state.danger:
            gap = state.min_separation - state.discomfort_dist
            discomfort_term = gap * state.discomfort_penalty_factor * state.time_step
        else:
            potential_term = state.potential_factor * state.potential_delta
    if state.robot.kinematics == "unicycle":
        spin_term = -4.5 * state.action[1] ** 2
        if state.action[0] < 0.0:
            backwards_term = -2.0 * abs(state.action[0])
    total = float(
        goal_reward
        + collision_term
        + discomfort_term
        + potential_term
        + spin_term
        + backwards_term
    )
    return total, {
        "goal_reward": float(goal_reward),
        "collision_penalty": float(collision_term),
        "discomfort_penalty": float(discomfort_term),
        "potential_shaping": float(potential_term),
        "unicycle_spin_penalty": float(spin_term),
        "unicycle_backwards_penalty": float(backwards_term),
    }
'''


def format_d1_initial(
    *,
    func_name: str = "compute_reward",
    include_seed: bool = True,
    include_external_knowledge: bool = True,
    reflection: str = "",
    compact: bool = False,
) -> str:
    """D.1 single-function prompt; ``compact`` drops the external-knowledge block
    (it repeats the system prompt's selection objective)."""
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
            f"Reflective guidance / evidence from prior generations:\n{reflection.strip()}\n"
        )
    external_block = ""
    if include_external_knowledge and not compact:
        external_block = f"External knowledge:\n{_external_knowledge()}\n"
    return D1_USER_PROMPT.format(
        func_name=func_name,
        task_description=_task_description(),
        action_space_block=action_space_prompt_block(),
        seed_block=seed_block,
        reflection_block=reflection_block,
        external_knowledge_block=external_block,
    )


_D1_BATCH_SANDBOX_RULES = """
Sandbox rules (CRITICAL — invalid code is discarded):
- Define exactly {n} top-level functions `{func_name}_v1` … `{func_name}_v{n}`, each
  returning ``(total_reward: float, reward_components: dict[str, float])``.
- Do NOT define helper functions — each reward must be a single self-contained def.
- ``total_reward`` is the only value used for RL; ``reward_components`` is a flat
  dict of named terms (≤12 str keys, finite float values, no nesting) for reflection.
- Do NOT use import/from, classes, while loops, print, comprehensions,
  generator expressions, or reflection builtins.
- Forbidden: getattr, hasattr, __import__, eval, exec, type, setattr, delattr, globals, locals, vars, open.
- Access CrowdNavRewardState only via dot notation (see below).
- Use ``memory`` (dict) for cross-timestep shaping; never invent state.history.
{reward_state_access}
- Hyperparameters as locals only; no extra args beyond (state, memory).
- No module-level lists/dicts/sets (they would survive episode resets).
- """ + _MAGNITUDE_RULE + """
"""

D1_BATCH_USER_PROMPT = """Please write **{n} diverse** Python reward functions for crowd-aware robot navigation (CrowdNav++).
Task: {task_description}
Each function: def {func_name}_vK(state, memory) for K=1..{n}, returning (finite float, dict[str, float]).
{sandbox_rules}
{action_space_block}
{seed_block}
{reflection_block}
{external_knowledge_block}
Return exactly one fenced Python code block containing all {n} functions.
"""


def format_d1_initial_batch(
    n: int,
    *,
    func_name: str = "compute_reward",
    include_seed: bool = True,
    include_external_knowledge: bool = True,
    reflection: str = "",
    compact: bool = False,
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
        reflection_block = f"Reflection / evidence:\n{reflection.strip()}\n"
    external_block = ""
    if include_external_knowledge and not compact:
        external_block = f"External knowledge:\n{_external_knowledge()}\n"
    return D1_BATCH_USER_PROMPT.format(
        n=int(n),
        func_name=func_name,
        task_description=_task_description(),
        sandbox_rules=_D1_BATCH_SANDBOX_RULES.format(
            n=int(n),
            func_name=func_name,
            reward_state_access=_REWARD_STATE_ACCESS,
        ),
        action_space_block=action_space_prompt_block(),
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
    compact: bool = False,
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
        diagnosis=_DIAGNOSIS_BEFORE_CODE_COMPACT if compact else _DIAGNOSIS_BEFORE_CODE,
    )


def format_d2_mutation(
    parent_code: str,
    reflection: str,
    *,
    func_name: str = "compute_reward",
    func_signature: str = "def compute_reward(state, memory):",
    compact: bool = False,
) -> str:
    # ``parent_code`` normally already starts with ``def ...``; only prepend the
    # signature when a caller passes a body fragment (keeps the kwarg for API
    # compatibility without duplicating the def line in the usual case).
    body = str(parent_code or "").rstrip()
    if "def " not in body.split("\n", 1)[0]:
        parent_block = f"{func_signature}\n{body}".rstrip()
    else:
        parent_block = body
    return D2_MUTATION_PROMPT.format(
        reflection=reflection.strip() or "(none)",
        parent_block=parent_block,
        func_name=func_name,
        sandbox_rules=_D2_SANDBOX_RULES.format(
            func_name=func_name,
            reward_state_access=_REWARD_STATE_ACCESS,
        ),
        diagnosis=_DIAGNOSIS_BEFORE_CODE_COMPACT if compact else _DIAGNOSIS_BEFORE_CODE,
    )


def seed_variant(n: int) -> str:
    """Valid ``compute_reward`` variant #n for the no-API ``seed`` LLM client.

    Structural diversity: upstream reward / human clearance / goal potential /
    progress shaping / time penalty, with coefficients varying in ``n``.
    """
    kind = int(n) % 5
    if kind == 0:
        goal = 1.0 + 0.05 * ((n % 10) - 5)
        coll = -0.25 - 0.02 * ((n % 6) - 3)
        body = (
            f"    goal_reward = {goal:.4f}\n"
            f"    collision_penalty = {coll:.4f}\n"
            "    if state.timeout:\n"
            "        return float(0.0)\n"
            "    if state.collision:\n"
            "        return float(collision_penalty)\n"
            "    if state.reached_goal:\n"
            "        return float(goal_reward)\n"
            "    if state.min_separation < state.discomfort_dist:\n"
            "        return float((state.min_separation - state.discomfort_dist) * 0.5 * state.time_step)\n"
            "    return float(0.0)\n"
        )
    elif kind == 1:
        w = 0.05 + 0.01 * ((n % 8) - 4)
        body = (
            f"    clear_w = {w:.4f}\n"
            "    if state.collision:\n"
            "        return float(-0.25)\n"
            "    if state.reached_goal:\n"
            "        return float(1.0)\n"
            "    nearest = 10.0\n"
            "    for h in state.humans:\n"
            "        d = ((h.px - state.robot.px) ** 2 + (h.py - state.robot.py) ** 2) ** 0.5\n"
            "        d = d - h.radius - state.robot.radius\n"
            "        if d < nearest:\n"
            "            nearest = d\n"
            "    clear = min(nearest, 1.0)\n"
            "    return float(clear_w * (clear - 1.0) * state.time_step)\n"
        )
    elif kind == 2:
        k = 0.1 + 0.02 * ((n % 6) - 3)
        body = (
            f"    potential_w = {k:.4f}\n"
            "    if state.collision:\n"
            "        return float(-0.25)\n"
            "    if state.reached_goal:\n"
            "        return float(1.0)\n"
            "    return float(-potential_w * state.goal_distance * state.time_step)\n"
        )
    elif kind == 3:
        k = 0.1 + 0.02 * ((n % 7) - 3)
        body = (
            f"    progress_w = {k:.4f}\n"
            "    if state.collision:\n"
            "        return float(-0.25)\n"
            "    if state.reached_goal:\n"
            "        return float(1.0)\n"
            "    reward = progress_w * state.progress\n"
            "    if state.min_separation < state.discomfort_dist:\n"
            "        reward = reward + (state.min_separation - state.discomfort_dist) * 0.5 * state.time_step\n"
            "    return float(reward)\n"
        )
    else:
        pen = 0.01 + 0.002 * ((n % 5) - 2)
        body = (
            f"    time_pen = {pen:.4f}\n"
            "    if state.collision:\n"
            "        return float(-0.25)\n"
            "    if state.reached_goal:\n"
            "        return float(1.0)\n"
            "    return float(-time_pen * state.time_step / state.time_limit * 100.0)\n"
        )
    return (
        "```python\n"
        "def compute_reward(state, memory):\n"
        f"{body}"
        "```\n"
    )
