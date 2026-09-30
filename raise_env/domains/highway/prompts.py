"""Highway domain pack — LLM reward prompts for highway-fast-v0."""

from __future__ import annotations

from typing import Optional

DOMAIN_NAME = "highway"

_SELECTION_OBJECTIVE = (
    "Your reward function is judged by training a PPO policy and evaluating it "
    "on held-out rollouts. Among policies that survive at least one holdout "
    "episode (SR > 0), candidates are ranked by Pareto dominance across: "
    "collision rate, off-road rate, forward progress, mean speed, "
    "lane_change_rate, and overtakes_per_km (successful passes of a vehicle "
    "that was ahead). Survival rate is used only as the feasibility gate "
    "(SR > 0), not as a separate ranked objective, because with exclusive "
    "outcomes it is redundant with collision/off-road rates. A candidate is "
    "only worse than another if it is equal-or-worse on every ranked "
    "objective and strictly worse on at least one (differences smaller than "
    "eval sampling noise do not count) — there is no fixed weighting between "
    "safety, speed, and overtaking, and there is no preferred target speed. "
    "Pure low-speed lane-keeping that never passes slower traffic is not "
    "automatically preferred. Reason about these trade-offs directly from "
    "the evidence given to you."
)

_DIAGNOSIS_BEFORE_CODE = (
    "Before writing code:\n"
    "1) In 2-4 sentences, based ONLY on the evidence given above (not on any "
    "assumption about what 'usually' goes wrong), diagnose what is currently "
    "limiting the holdout objectives / Pareto standing — e.g. an unexploited "
    "safety/speed trade-off, a degenerate strategy visible in the numbers, or "
    "something else you notice. When reward-component trends are present, use "
    "them together with the task metrics.\n"
    "2) Then revise the reward function to address your own diagnosis.\n"
    "Interpretation principles for reward-component trends (EUREKA):\n"
    "- If the survival/success rate is always near zero, rewrite the entire "
    "reward function.\n"
    "- If a component's value is nearly identical across all checkpoints, "
    "RL was not able to optimize it as written — consider changing its "
    "scale/temperature, rewriting it, or discarding it.\n"
    "- If one component's magnitude is significantly larger than the others, "
    "rescale it to a comparable range.\n"
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
    - state.collision, state.off_road, state.timeout: bool (environment-owned)
    - state.action, state.time_step, state.global_time, state.time_limit
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
Math:
- Prefer `** 0.5` for square roots. Do not write `import math`.
- Do not use getattr, hasattr, or __import__.
Constraints (CRITICAL):
- Hyperparameters must be local variables inside the function body.
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
state.robot.px        # CrowdNav-only; use state.ego.x
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
- state.collision, state.off_road, state.timeout
- state.action, state.time_step, state.global_time, state.time_limit, state.progress, state.speed
- memory: plain dict for episode-local state; cleared on reset
- FORBIDDEN (rejected by AST): lane_position, distance, robot, humans, gx/gy, history, px/py
- Math: no import math; use ** 0.5. No getattr/hasattr/__import__.
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

D3_SYSTEM_PROMPT = (
    "You are a senior researcher in autonomous driving and RL. "
    + _SELECTION_OBJECTIVE
    + " "
    + _BASE_MODEL_SPEED_PRIOR
    + "IMPORTANT SCHEMA: def compute_reward(state, memory): — memory is a plain dict. "
    "state.ego has .x .y .vx .vy .heading .speed .lane_index .on_road. "
    "state.others is a tuple; iterate with 'for v in state.others:'. "
    "state.collision / state.off_road / state.timeout (bool). "
    "state.progress and state.speed are available. "
    "NO state.robot, NO state.humans, NO state.gx, NO state.lane_position, NO state.distance. "
    "SANDBOX: never getattr/hasattr/__import__/eval; use ** 0.5; "
    "return (total_reward: float, reward_components: dict[str, float]) — "
    "flat component dict ≤12 str keys, finite floats, no nesting. "
    "Write a short diagnosis as plain text, then a single Python fenced code block "
    "containing ONLY the revised function."
)

D3_USER_PROMPT = """Current score (best so far): {last_score:.4f} (higher is better)
Evidence / focus note:
{feedback}
{extra_context_if_any}
""" + _DIAGNOSIS_BEFORE_CODE + """
Revise the function below.
Maintain signature def compute_reward(state, memory): and return (total_reward, reward_components).
{current_code}
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
- soft_success: logged diagnostic only (not a Pareto / selection objective —
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
    from domains.highway.action_config import action_space_prompt_block

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
No imports, classes, getattr/hasattr. Prefer ** 0.5 for roots.
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
    from domains.highway.action_config import action_space_prompt_block

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
    from domains.highway.sandbox_fields import HIGHWAY_FORBIDDEN_HALLUCINATIONS

    blob = f"{bad_code}\n{validation_error}".lower()
    seen = [name for name in HIGHWAY_FORBIDDEN_HALLUCINATIONS if name in blob]
    remap = {
        "lane_position": "state.ego.lane_index",
        "distance": "((v.x)**2 + (v.y)**2) ** 0.5 for v in state.others",
        "robot": "state.ego (e.g. state.ego.x, state.ego.speed)",
        "humans": "state.others",
        "gx": "(no goal x — use state.progress / state.ego.x)",
        "gy": "(no goal y — use state.ego.y)",
        "px": "state.ego.x",
        "py": "state.ego.y",
        "history": "memory dict only",
        "reaching_goal": "(no goal flag — use progress/speed shaping)",
        "position": "state.ego.x / state.ego.y",
        "velocity": "state.ego.vx / state.ego.vy or state.speed",
        "vehicles": "state.others",
        "nearby": "state.others",
        "lane_id": "state.ego.lane_index",
        "ego_vehicle": "state.ego",
        "vx_ego": "state.ego.vx",
    }
    fix_lines = []
    for name in seen:
        alt = remap.get(name, "a documented HighwayRewardState field")
        fix_lines.append(f"- Replace `{name}` → {alt}")
    if not fix_lines:
        fix_lines.append(
            "- If ERROR mentions an attribute, delete it and use only "
            "state.ego.* / state.others / state.collision|off_road|timeout / "
            "state.progress / state.speed / memory."
        )
    fix_block = "\n".join(fix_lines)
    return (
        "The following highway reward failed validation. Repair it in ONE shot.\n\n"
        f"ERROR: {validation_error}\n\n"
        f"ORIGINAL CODE:\n{bad_code}\n\n"
        "MANDATORY FIELD FIXES (hallucinated names are rejected by AST allowlist):\n"
        f"{fix_block}\n\n"
        "Allowed fields only:\n"
        "  state.ego.{x,y,vx,vy,heading,speed,lane_index,on_road}\n"
        "  state.others → for v in state.others: v.{x,y,vx,vy,heading}\n"
        "  state.collision, state.off_road, state.timeout\n"
        "  state.progress, state.speed, state.action, state.time_step, "
        "state.global_time, state.time_limit\n"
        "  memory.get / memory['key']\n\n"
        "Keep signature def compute_reward(state, memory):. "
        "No imports, getattr, hasattr, comprehensions. "
        "Return (total_reward: float, reward_components: dict[str, float]). "
        "Output only the fixed function (no markdown)."
    )
