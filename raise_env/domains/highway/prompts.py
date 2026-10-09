"""Highway domain pack — LLM reward prompts for highway-fast-v0."""

from __future__ import annotations

from typing import Optional

from domains.highway.action_config import action_space_prompt_block
from raise_core.sandbox.config import (
    SAFE_MATH_ATTRIBUTES,
    SAFE_MATH_CONSTANTS,
    SandboxConfig,
)

DOMAIN_NAME = "highway"

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

_SELECTION_OBJECTIVE = (
    "Your reward function is judged by training a PPO policy and evaluating it "
    "on held-out rollouts. Candidates are ranked by one holdout fitness "
    "score (higher is better) that combines survival (survival rate minus "
    "twice the collision rate minus the off-road rate, scaled by a speed gate "
    "that grows as the policy drives closer to free-flowing traffic speed) "
    "with forward progress and mean speed, which only count in proportion to "
    "positive survival, minus penalties for lagging, degenerate or very slow "
    "driving. A collision costs more than a survived episode earns, so "
    "progress made before crashing earns nothing. Pure low-speed "
    "lane-keeping is not preferred either. Reason about these "
    "trade-offs directly from the evidence given to you."
)

_DIAGNOSIS_BEFORE_CODE = (
    "Before writing code:\n"
    "1) In 2-4 sentences, based ONLY on the evidence given above (not on any "
    "assumption about what 'usually' goes wrong), diagnose what is currently "
    "limiting the holdout fitness — e.g. an unexploited safety/speed "
    "trade-off, a degenerate strategy visible in the numbers, or something "
    "else you notice.\n"
    "2) Then revise the reward function to address your own diagnosis.\n"
    "- If the survival/success rate is always near zero, rewrite the entire "
    "reward function.\n"
    "Output your diagnosis as plain text BEFORE the code fence. The code fence "
    "must contain ONLY the function, no diagnosis text inside it.\n"
    "A 'small precise change' can still mean removing or replacing a constant "
    "entirely if the evidence suggests it no longer reflects reality — don't "
    "assume an inherited number is correct just because it came from the "
    "parent."
)

_RETURN_CONTRACT = (
    "Return exactly ``(total_reward, reward_components)`` where "
    "``total_reward`` is a finite float used for RL, and "
    "``reward_components`` is a flat dict[str, float] of named terms for "
    "this step (e.g. {\"progress_term\": 0.5, \"speed_term\": 0.1, "
    "\"collision_penalty\": 0.0}). At most 12 keys; string keys only; "
    "finite float values only; no nested containers. The dict is "
    "informational (diagnostics/reflection) and must not change what "
    "``total_reward`` already encodes."
)

_BASE_MODEL_SPEED_PRIOR = (
    "Do not assume a real-world 'typical highway speed' is the right target "
    "here — this environment's actual safe/achievable speed range must be "
    "inferred from the evidence given to you (population trade-offs, feasible "
    "range), not from general driving knowledge, since it depends on this "
    "specific simulator's traffic density and dynamics. "
)

D1_SYSTEM_PROMPT = (
    "You are an expert in reinforcement learning and autonomous highway driving. "
    "Your goal is to design reward functions for highway-fast-v0. "
    + _SELECTION_OBJECTIVE
    + " "
    + _BASE_MODEL_SPEED_PRIOR
    + _RETURN_CONTRACT
    + " "
    + "Return **only** valid Python code enclosed within a fenced code block "
    "for this initial generation (no commentary outside the block). "
    "The code must be fully executable."
)

D1_USER_PROMPT = """Please write a Python function named {func_name} for highway driving (highway-fast-v0).
Task Description:
- Output ``(total_reward, reward_components)`` from the ego vehicle's current state so a policy can learn safe forward driving under the selection objective above (total_reward trains RL; reward_components are for reflection only).
Function Interface (EXACT FIELD STRUCTURE):
- Inputs:
  - state: A HighwayRewardState snapshot with ONLY these fields:
    - state.ego.x, state.ego.y, state.ego.vx, state.ego.vy, state.ego.heading
    - state.ego.speed, state.ego.lane_index, state.ego.on_road
    - state.others: tuple of NearbyVehicle; iterate `for v in state.others:` then v.x, v.y, v.vx, v.vy, v.heading
    - FRAMES: state.ego.x / state.ego.y are ABSOLUTE road coordinates; v.x, v.y, v.vx, v.vy are EGO-RELATIVE (other minus ego: v.x > 0 ahead, v.x < 0 behind, |v.y| < 2 same lane). Never compare v.x with state.ego.x or memory['prev_x']; compare v.x with 0 (e.g. an overtake is a vehicle whose v.x flipped from > 0 to <= 0 between steps). Headings (state.ego.heading, v.heading) are absolute radians, ~0 along the road.
    - LANES: state.ego.lane_index 0 is the leftmost lane; larger index = further right (y grows to the right, 4 m per lane). LANE_LEFT lowers lane_index, LANE_RIGHT raises it.
    - state.collision, state.off_road, state.timeout: bool (environment-owned). state.timeout is True only on the final step of an episode that reached the time limit WITHOUT collision/off-road — it marks a survived (successful) episode, not a failure.
    - state.action (type depends on the action space below), state.time_step, state.global_time, state.time_limit
    - state.progress: float (forward meters since previous frame)
    - state.speed: float (ego speed, same as state.ego.speed)
  - memory: plain mutable dict cleared at every episode reset(); use for shaping (e.g. memory['prev_speed']).
  - CRITICAL: There is NO state.robot, NO state.humans, NO state.gx/gy, NO state.history — only fields listed above.
  - Do NOT define classes. Persistent state must use the ``memory`` dict only.
- Output:
  - Return ``(total_reward, reward_components)`` — total_reward is the finite float used for RL; reward_components is a flat dict[str, float] naming the terms that sum (or otherwise compose) into that total for this step.
- Design Principles:
- Prefer dense, finite shaping from documented state fields.
- Interpretability: clear local variables; named components in the dict; no extra signature args.
- Reason about multi-objective trade-offs yourself from the selection objective; do not assume a named failure mode.
Episode memory example:
```python
def compute_reward(state, memory):
    prev = memory.get("prev_x")
    progress = 0.0 if prev is None else float(state.ego.x) - float(prev)
    memory["prev_x"] = float(state.ego.x)
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
- Access fields only via dot notation. Iterate others with `for v in state.others:`.
- AST allowlist: ONLY the fields listed above (+ memory.get / memory['k']). Any other
  attribute (lane_position, distance, robot, humans, gx, history, …) is REJECTED.
Negative examples (DO NOT write these — they fail validation):
```python
# BAD — hallucinated fields
state.lane_position   # use state.ego.lane_index
state.distance        # compute ((v.x)**2+(v.y)**2)**0.5 for v in state.others
state.robot.px        # does not exist; use state.ego.x
state.humans          # does not exist; use state.others
```
- Output Format: only the Python function in a single code block.
{action_space_block}
{seed_block}
{reflection_block}
{external_knowledge_block}
"""

PROMPT_MEMORY_EXAMPLE = '''def compute_reward(state, memory):
    prev = memory.get("prev_x")
    progress = 0.0 if prev is None else float(state.ego.x) - float(prev)
    memory["prev_x"] = float(state.ego.x)
    total = float(progress)
    return total, {"progress_term": total}
'''

_REWARD_STATE_ACCESS = """\
HighwayRewardState access (dot notation only — never getattr/hasattr/__import__):
- state.ego.x, state.ego.y, state.ego.vx, state.ego.vy, state.ego.heading, state.ego.speed, state.ego.lane_index, state.ego.on_road
- state.others — loop `for v in state.others:` then v.x, v.y, v.vx, v.vy, v.heading
- FRAMES: state.ego.x/y are absolute; v.x, v.y, v.vx, v.vy are ego-relative (v.x > 0 ahead, v.x < 0 behind). Compare v.x with 0, never with state.ego.x or memory['prev_x']. Headings are absolute radians (~0 along the road).
- LANES: lane_index 0 is the leftmost lane; larger index = further right (y grows to the right). LANE_LEFT lowers lane_index.
- state.collision, state.off_road, state.timeout (timeout = survived to the time limit, i.e. success)
- state.action, state.time_step, state.global_time, state.time_limit, state.progress, state.speed
- memory: plain dict for episode-local state; cleared on reset
- FORBIDDEN (rejected by AST): lane_position, distance, robot, humans, gx/gy, history, px/py
- """ + _MATH_RULE + """
- No getattr/hasattr/__import__.
"""

_D2_SANDBOX_RULES = """
Sandbox rules (CRITICAL — invalid code is discarded):
- Define exactly ONE top-level function `{func_name}(state, memory)` returning
  ``(total_reward: float, reward_components: dict[str, float])``.
- ``total_reward`` is the only value used for RL; ``reward_components`` is a flat
  dict of named terms (≤12 str keys, finite float values, no nesting) for reflection.
- Do NOT use import/from, classes, while loops, print, lambda, comprehensions,
  generator expressions, or reflection builtins.
- Forbidden: getattr, hasattr, __import__, eval, exec, type, setattr, delattr, globals, locals, vars, open.
- Access HighwayRewardState only via dot notation (see below).
- Use ``memory`` (dict) for cross-timestep shaping; never invent state.history.
{reward_state_access}
- Hyperparameters as locals only; no extra args beyond (state, memory).
- No module-level lists/dicts/sets (they would survive episode resets).
- """ + _MAGNITUDE_RULE + """
- Put diagnosis as plain text BEFORE the fence; the fenced block must contain ONLY the function.
"""

D2_CROSSOVER_PROMPT = """You are a reward function architect for highway driving. Synthesize a new function combining complementary strengths of two parents while addressing the reflection.
Parent A:
- {code_A}
Parent B:
- {code_B}
Reflection / evidence:
- {reflection}.
Synthesis Task:
- Write an improved `{func_name}` that merges complementary terms from the parents.
- Strip any getattr/hasattr patterns; use direct state.* / v.* access.
- Define exactly one function: def {func_name}(state, memory): ... returning (float, dict[str, float]).
{sandbox_rules}
""" + _DIAGNOSIS_BEFORE_CODE + """
"""

D2_MUTATION_PROMPT = """You are a reward function optimizer for highway driving. Mutate the underperforming parent using the evidence below with minimal edits.
Prior Reflection / evidence:
- {reflection}
Underperforming Parent Code to Mutate:
- {func_signature}
- {parent_code}
Mutation Task:
- Create a mutated `{func_name}` with a small precise change that addresses your diagnosis.
- Keep direct dot access; no getattr/hasattr.
- Define exactly one function: def {func_name}(state, memory): ... returning (float, dict[str, float]).
{sandbox_rules}
""" + _DIAGNOSIS_BEFORE_CODE + """
"""

D4_EXTERNAL_KNOWLEDGE = f"""# External Knowledge — Highway Fast
## Task
- Domain: multi-lane highway driving (highway-fast-v0)
## Selection objective (not the reward itself)
{_SELECTION_OBJECTIVE}
## Metrics (mapped to RAISE ProxyMetrics)
- SR: fraction of episodes survived without collision/off-road
- CR: collision rate; TR: off-road rate
- mean_speed / mean_progress / lane_change_rate / overtakes_per_km as logged on holdout
- soft_success: small fitness bonus only (not a primary objective —
  do not design the reward specifically to hit this indicator)
"""

D5_SEED_FUNCTION = '''def compute_reward(state, memory):
    """Highway seed: dense progress/speed shaping with collision/off-road costs."""
    collision_penalty = -20.0
    off_road_penalty = -10.0
    speed_coef = 0.08
    progress_coef = 1.0
    if state.collision:
        total = float(collision_penalty)
        return total, {
            "collision_penalty": total,
            "off_road_penalty": 0.0,
            "progress_term": 0.0,
            "speed_term": 0.0,
        }
    if state.off_road:
        total = float(off_road_penalty)
        return total, {
            "collision_penalty": 0.0,
            "off_road_penalty": total,
            "progress_term": 0.0,
            "speed_term": 0.0,
        }
    progress_term = float(progress_coef * state.progress)
    speed_term = float(speed_coef * state.speed)
    total = float(progress_term + speed_term)
    return total, {
        "collision_penalty": 0.0,
        "off_road_penalty": 0.0,
        "progress_term": progress_term,
        "speed_term": speed_term,
    }
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
            f"Reflective guidance / evidence from prior generations:\n{reflection.strip()}\n"
        )
    external_block = ""
    if include_external_knowledge:
        external_block = f"External knowledge:\n{D4_EXTERNAL_KNOWLEDGE}\n"
    return D1_USER_PROMPT.format(
        func_name=func_name,
        action_space_block=action_space_prompt_block(),
        seed_block=seed_block,
        reflection_block=reflection_block,
        external_knowledge_block=external_block,
    )


D1_BATCH_USER_PROMPT = """Please write **{n} diverse** Python reward functions for highway-fast-v0 driving.
Each function: def {func_name}_vK(state, memory) for K=1..{n}, returning (finite float, dict[str, float]).
Use HighwayRewardState fields only (state.ego.*, state.others, state.collision/off_road/timeout, state.progress, state.speed, memory).
FORBIDDEN fields (will be rejected): lane_position, distance, robot, humans, gx, gy, history.
No imports, classes, getattr/hasattr. """ + _MATH_RULE + """
{action_space_block}
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
        reflection_block = f"Reflection / evidence:\n{reflection.strip()}\n"
    external_block = ""
    if include_external_knowledge:
        external_block = f"External knowledge:\n{D4_EXTERNAL_KNOWLEDGE}\n"
    return D1_BATCH_USER_PROMPT.format(
        n=int(n),
        func_name=func_name,
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
