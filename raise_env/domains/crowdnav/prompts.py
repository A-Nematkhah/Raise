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

_TASK = (
    "CrowdNav (vita-epfl/CrowdNav simulator): a robot (radius 0.3 m, preferred "
    "speed v_pref = 1 m/s) starts at (0, -4) and must reach its goal at (0, 4) "
    "through 5 humans that cross a 4 m circle under ORCA. The robot is "
    "invisible to the humans: they do not yield to it. Each step lasts 0.25 s; "
    "an episode ends on reaching the goal, on a collision, or at the time limit "
    "(about 24 s)."
)

_SELECTION_OBJECTIVE = (
    "Your reward function is judged by training a PPO policy with it and "
    "evaluating that policy on held-out CrowdNav validation scenarios. "
    "Candidates are ranked by one holdout fitness (higher is better): the mean "
    "discounted task return of the original CrowdNav evaluation - reaching the "
    "goal earns +1 discounted by 0.9 per second of navigation time, a collision "
    "costs -0.25, reaching the time limit earns 0, and every step closer than "
    "the 0.2 m discomfort distance to a human costs a small penalty. Fast, "
    "collision-free arrivals score highest; freezing until the time limit "
    "scores about zero; collisions score below that. Reason about these "
    "trade-offs directly from the evidence given to you."
)

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

D1_SYSTEM_PROMPT = (
    "You are an expert in reinforcement learning and crowd-aware robot navigation. "
    "Your goal is to design reward functions for the CrowdNav simulator. "
    + _SELECTION_OBJECTIVE
    + " "
    + _RETURN_CONTRACT
    + " "
    + "Put the code in exactly one ```python fenced block; any explanation must "
    "come before the fence. The code must be fully executable."
)

_STATE_FIELDS = """\
  - state: a CrowdNavRewardState snapshot taken AFTER the step, with ONLY these fields:
    - state.robot.px, state.robot.py, state.robot.vx, state.robot.vy, state.robot.gx, state.robot.gy, state.robot.theta, state.robot.radius, state.robot.v_pref
    - state.humans: tuple of humans; iterate `for h in state.humans:` then h.px, h.py, h.vx, h.vy, h.radius
    - FRAMES: every position and velocity is in the WORLD frame (metres, m/s); (state.robot.gx, state.robot.gy) is the goal. Distances to humans must be computed, e.g. ((h.px - state.robot.px) ** 2 + (h.py - state.robot.py) ** 2) ** 0.5 is centre-to-centre; subtract h.radius + state.robot.radius for the gap between bodies.
    - state.collision, state.reached_goal, state.timeout: bool (environment-owned outcome of THIS step, mutually exclusive; each ends the episode). state.timeout marks a failed episode that ran out of time.
    - state.min_separation: float, the smallest gap between the robot's and any human's body during this step (negative on collision); state.discomfort_dist: float (0.2 m), the comfort threshold for that gap.
    - state.goal_distance: float, robot-centre distance to the goal after the step; state.progress: float, decrease of goal_distance during this step (negative when moving away).
    - state.action (see the action space below), state.time_step (s), state.global_time (s since reset), state.time_limit (s)
  - memory: plain mutable dict cleared at every episode reset(); use for shaping (e.g. memory['prev_speed'])."""

D1_USER_PROMPT = """Please write a Python function named {func_name} for crowd-aware robot navigation (CrowdNav).
Task Description:
- """ + _TASK + """
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
- state.robot.px, .py, .vx, .vy, .gx, .gy, .theta, .radius, .v_pref (world frame; (gx, gy) is the goal)
- state.humans — loop `for h in state.humans:` then h.px, h.py, h.vx, h.vy, h.radius (world frame)
- state.collision, state.reached_goal, state.timeout (outcome of this step, mutually exclusive; timeout = ran out of time = failure)
- state.min_separation (smallest robot-human body gap during this step; negative on collision), state.discomfort_dist (0.2 m)
- state.goal_distance (after the step), state.progress (decrease of goal_distance this step)
- state.action, state.time_step, state.global_time, state.time_limit
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

D4_EXTERNAL_KNOWLEDGE = f"""# External Knowledge — CrowdNav
## Task
- {_TASK}
## Selection objective (not the reward itself)
{_SELECTION_OBJECTIVE}
## Metrics (as in the CrowdNav evaluation)
- SR: success rate (goal reached); CR: collision rate; TR: timeout rate
- nav_time: mean navigation time of successful episodes (s); path_length: mean path length of successful episodes (m)
- danger_freq: fraction of steps inside the discomfort distance; min_dist_in_danger: mean gap during those steps (m)
"""

D5_SEED_FUNCTION = '''def compute_reward(state, memory):
    """CrowdNav seed: the upstream CrowdSim reward (crowd_nav/configs/env.config)."""
    success_reward = 1.0
    collision_penalty = -0.25
    discomfort_penalty_factor = 0.5
    goal_reward = 0.0
    collision_term = 0.0
    discomfort_term = 0.0
    # Upstream priority: timeout (reward 0) > collision > goal > discomfort.
    if not state.timeout:
        if state.collision:
            collision_term = collision_penalty
        elif state.reached_goal:
            goal_reward = success_reward
        elif state.min_separation < state.discomfort_dist:
            gap = state.min_separation - state.discomfort_dist
            discomfort_term = gap * discomfort_penalty_factor * state.time_step
    total = float(goal_reward + collision_term + discomfort_term)
    return total, {
        "goal_reward": float(goal_reward),
        "collision_penalty": float(collision_term),
        "discomfort_penalty": float(discomfort_term),
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
        external_block = f"External knowledge:\n{D4_EXTERNAL_KNOWLEDGE}\n"
    return D1_USER_PROMPT.format(
        func_name=func_name,
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

D1_BATCH_USER_PROMPT = """Please write **{n} diverse** Python reward functions for crowd-aware robot navigation (CrowdNav).
Task: """ + _TASK + """
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
        external_block = f"External knowledge:\n{D4_EXTERNAL_KNOWLEDGE}\n"
    return D1_BATCH_USER_PROMPT.format(
        n=int(n),
        func_name=func_name,
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
