"""
CrowdNav domain pack — RAISE Appendix D prompt templates (arXiv:2605.11859).

Canonical home for CrowdNav LLM prompts. ``crowd_nav.reward_search.prompts``
re-exports this module for backward compatibility.

Ported from Appendix D.1–D.5, with intentional adaptations for this fork:
- Signature is ``compute_reward(state, memory)`` over our ``RewardState``
  (Phase 1 sandbox contract) instead of the paper's ``cal_reward`` /
  ``compute_reward(inst, traj)``.
- Stateful shaping uses a sandbox-owned ``memory`` dict (cleared on episode
  ``reset()``), **not** classes. Classes are forbidden by the AST policy;
  teaching class-based rewards caused LLM hallucinations (state.history /
  state.prev_state) that could never pass validation.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# D.1 Initial Population
# ---------------------------------------------------------------------------

D1_SYSTEM_PROMPT = (
    "You are an expert in reinforcement learning and robot navigation. "
    "Your goal is to design reward functions that effectively guide a robot "
    "toward safe, efficient, and socially compliant navigation behaviors. "
    "Return **only** valid Python code enclosed within a fenced code block. "
    "The code must be fully executable and should not include comments or "
    "explanations outside the block."
)

# {func_name} defaults to compute_reward; interface adapted to RewardState + memory.
D1_USER_PROMPT = """Please write a Python function named {func_name} for a robot navigation task in a crowded environment.
Task Description:
- The function's goal is to output a scalar reward value based on the robot's current state, guiding it to its goal while avoiding collisions with dynamic human agents.
Function Interface (EXACT FIELD STRUCTURE):
- Inputs:
  - state: A RewardState snapshot for the current frame with ONLY these fields:
    - state.robot.px, state.robot.py (position)
    - state.robot.vx, state.robot.vy (velocity)
    - state.robot.radius, state.robot.gx, state.robot.gy, state.robot.v_pref
    - state.humans: tuple of HumanObservable objects; iterate via `for human in state.humans:` then access human.px, human.py, human.vx, human.vy, human.radius
    - state.dmin: float (closest human distance minus radii, precomputed)
    - state.discomfort_dist: float (TOP-LEVEL on state, not under robot)
    - state.collision, state.reaching_goal, state.timeout: bool (environment-computed flags)
    - state.action, state.time_step, state.global_time, state.time_limit: float or tuple
  - memory: A plain mutable dict owned by the runner. Cleared at every episode reset();
    the SAME dict object is passed on every compute call within that episode. Use it for
    progress shaping (e.g. store previous distance under memory['prev_dist']).
  - CRITICAL: There is NO state.history, NO state.prev_state, NO state.obstacle_dist, NO state.obstacle_distance, NO state.safety_dist — these do not exist. Only access fields listed above.
  - Do NOT define classes. Persistent state must use the ``memory`` dict only.
- Output:
  - A single scalar (float) representing the reward for the current state or action.
Design Principles:
- Goal-Progress: The function should reward progress toward the robot goal.
- Collision Avoidance: The function must penalize states that are too close to humans or result in a collision.
- Interpretability: The code should be well-commented, and variable names should be clear to facilitate human understanding.
Episode memory (if you need to track progress across timesteps):
- Use the injected ``memory`` dict — never classes, never attributes on state:
```python
def compute_reward(state, memory):
    dist = ((state.robot.px - state.robot.gx)**2 + (state.robot.py - state.robot.gy)**2) ** 0.5
    prev = memory.get("prev_dist")
    progress = 0.0 if prev is None else float(prev) - dist
    memory["prev_dist"] = dist
    return float(progress)
```
- The framework clears ``memory`` once per episode start and passes it into every call.
Math Functions:
- The math module is not importable (no `import math`, no `__import__`). Prefer `** 0.5` for square roots.
- Example: dist = ((dx**2 + dy**2) ** 0.5). If you must use math helpers, call the pre-injected name `math.sqrt(...)` with NO import statement.
- Do not use getattr, hasattr, or __import__.
Constraints (CRITICAL):
- Hyperparameters: All tuning parameters (e.g., weights, constants) must be defined as local variables inside the function body. Do not add them as function arguments beyond (state, memory).
- Signature: Define exactly one function: def {func_name}(state, memory): ... that returns a finite float.
- Sandbox: No import statements, no classes, while loops, or reflection builtins (getattr, hasattr, __import__, eval, type, ...). Access state fields only via dot notation (state.robot.px, state.dmin, state.humans, ...). Iterate humans with `for human in state.humans:` — do not index HumanObservable like a sequence (no human[0]).
- Scalars are plain floats/bools: never write ``state.robot.px[0]``, ``state.dmin[0]``, ``human.vx[0]``, etc. Use ``state.robot.px`` directly.
- Output Format: Your response must contain only the Python function within a single code block. Do not include any explanatory text or print statements.
{seed_block}
{reflection_block}
{external_knowledge_block}
"""

# Canonical prompt example — must remain sandbox-valid (regression-tested).
PROMPT_MEMORY_EXAMPLE = '''def compute_reward(state, memory):
    dist = ((state.robot.px - state.robot.gx)**2 + (state.robot.py - state.robot.gy)**2) ** 0.5
    prev = memory.get("prev_dist")
    progress = 0.0 if prev is None else float(prev) - dist
    memory["prev_dist"] = dist
    return float(progress)
'''

# ---------------------------------------------------------------------------
# D.2 Evolutionary Operations
# ---------------------------------------------------------------------------

_REWARD_STATE_ACCESS = """\
RewardState access (dot notation only — never getattr/hasattr/__import__):
- state.robot.px, state.robot.py, state.robot.vx, state.robot.vy, state.robot.radius, state.robot.gx, state.robot.gy, state.robot.v_pref
- state.humans — loop with `for human in state.humans:` then human.px, human.py, human.vx, human.vy, human.radius (never human[0] / sequence indexing)
- state.dmin, state.discomfort_dist, state.collision, state.reaching_goal, state.timeout
- state.action, state.time_step, state.global_time, state.time_limit
- Scalars are floats/bools — never index them (no state.robot.px[0], state.dmin[0], human.px[0])
- memory: plain dict for episode-local state (e.g. memory['prev_dist']); cleared on episode reset
- Math: math module is not importable; use ** 0.5 for square roots. Do not use getattr, hasattr, or __import__.
"""

_D2_SANDBOX_RULES = """
Sandbox rules (CRITICAL — invalid code is discarded):
- Define exactly ONE top-level function `{func_name}(state, memory)` returning a finite float.
- Do NOT use import/from (math is not importable — prefer ** 0.5), classes, while loops, print, lambda, or reflection builtins.
- Forbidden names (instant rejection): getattr, hasattr, __import__, eval, exec, type, setattr, delattr, globals, locals, vars, open.
- Access RewardState only via dot notation (see below). If a parent uses getattr/hasattr or dynamic field lookup, rewrite those lines to explicit attribute access before returning code.
- Use ``memory`` (dict) for cross-timestep shaping; never invent state.history / state.prev_state.
{reward_state_access}
- Use only local variables for hyperparameters; no extra function arguments beyond (state, memory).
- Return ONLY one Python fenced code block with no text outside it.
"""

D2_CROSSOVER_PROMPT = """You are a reward function architect. Your task is to synthesize a new function by combining the complementary strengths of two parent functions. Below are two candidate reward functions and a textual reflection. Use them to synthesize an improved hybrid design that combines the strengths of both functions while addressing weaknesses noted in the reflection.
Parent A:
- {code_A}
Parent B:
- {code_B}
Reflection:
- {reflection}.
Synthesis Task:
- Based on the provided reflection, write a new, improved function `{func_name}` that merges the superior safety features from Parent A with the efficiency-promoting logic from Parent B.
- When copying logic from parents, strip any getattr/hasattr/dynamic-access patterns and use direct `state.*` / `human.*` attribute access instead.
- Define exactly one function: def {func_name}(state, memory): ... returning a finite float.
{sandbox_rules}
- Return only a single Python fenced code block.
"""

D2_MUTATION_PROMPT = """You are a reward function optimizer. Your task is to perform a targeted mutation on an underperforming parent reward to address a specific weakness identified in the reflection (RAISE §4.2: mutation addresses individual weaknesses). A prior reflection summarizing Score1 feedback is provided below. Use it to locally improve this weaker parent by small, targeted edits while preserving any useful structure.
Prior Reflection:
- {reflection}
Underperforming Parent Code to Mutate:
- {func_signature}
- {parent_code}
Mutation Task:
- Based on the reflection, create a mutated function `{func_name}`. Make a minimal, precise change to the code to address the identified weakness without needlessly destroying working terms.
- Keep direct dot access to RewardState fields; do not introduce getattr, hasattr, or other forbidden reflection builtins.
- Define exactly one function: def {func_name}(state, memory): ... returning a finite float.
{sandbox_rules}
- Return only a single Python fenced code block.
"""

# ---------------------------------------------------------------------------
# D.3 Reward Function Refinement (Stages II/III; kept for completeness)
# ---------------------------------------------------------------------------

D3_SYSTEM_PROMPT = (
    "You are a senior researcher in robot motion planning and reinforcement learning. "
    "Rewrite the provided reward function to produce a smooth, dense, and numerically "
    "stable per-frame signal that differentiates between successful, safe, and "
    "inefficient navigation behaviors. "
    "IMPORTANT SCHEMA REMINDER: "
    "Signature must be def compute_reward(state, memory): — memory is a plain dict "
    "cleared each episode; use it for progress shaping (never classes). "
    "state.robot has .px, .py, .vx, .vy, .radius, .gx, .gy, .v_pref. "
    "state.humans is a tuple; iterate with 'for human in state.humans:' then access human.px, human.py, human.vx, human.vy, human.radius. "
    "state.dmin (float), state.discomfort_dist (float, TOP-LEVEL, not under robot), state.collision/reaching_goal/timeout (bool). "
    "NO state.history, NO state.obstacle_dist, NO state.safety_dist — these do not exist. "
    "SANDBOX HARD RULES (instant reject if violated): "
    "- Never use getattr, hasattr, setattr, eval, exec, type, print, __import__. "
    "- Access fields only with dot notation (state.dmin, state.robot.px). "
    "- math module is not importable; use ** 0.5 for square roots "
    "(do not write import math or __import__). "
    "- Always return a finite float (never None). "
    "Requirements: "
    "- Keep the original function signature compute_reward(state, memory). "
    "- Maintain O(T) computational complexity. "
    "- Ensure bounded reward magnitudes and graceful handling of incomplete trajectories. "
    "- Output only valid Python code (no markdown fences or commentary)."
)

D3_USER_PROMPT = """Current score (best so far): {last_score:.4f} (higher is better, approx. range -1 to 1)
Core components to ensure (abstract, unordered):
- Progressive advancement signal
- Safety differentiation
- Terminal / completion handling
- Efficiency (avoid needless detours)
- Stability (avoid erratic frame jumps)
Guidelines:
Strengthen discrimination between clearly successful, safe, efficient progress and poor / unsafe / stagnant behavior. Maintain:
- Dense per-frame shaping (not a single terminal spike)
- Bounded, numerically stable magnitudes (avoid runaway growth)
- Graceful handling of incomplete trajectories
Discourage unsafe or aimless motion and excessive oscillation without over-penalizing reasonable detours. Keep the logic straightforward and linear-time.
Keep all existing effective terms (progress-to-goal, safety distance shaping, smoothness, efficiency) unless there is a concrete numerical reason to adjust them.
Make only minimal, localized changes that strictly improve discriminative sharpness without removing previously working logic.
Avoid producing nearly constant rewards across different frames or trajectories; variance should reflect qualitative behavioral differences.
Focus note: {feedback}
{extra_context_if_any}
Revise the function below.
Maintain the original signature def compute_reward(state, memory): and return a meaningful per-frame shaping signal aggregated appropriately.
Sandbox reminder: math is not importable (use ** 0.5); never getattr/hasattr/__import__; always return a finite float.
Return **only** the updated function definition (no additional text).
{current_code}
"""

# ---------------------------------------------------------------------------
# D.4 External Knowledge
# ---------------------------------------------------------------------------

D4_EXTERNAL_KNOWLEDGE = """# External Knowledge for Reward Function Design
## Task Definition
- Domain: Crowd-robot navigation in continuous 2D environments
- Dataset: Synthetic environments derived from popular benchmarks, where human agents follow realistic social trajectories
- Input:
  - Current robot position (x_t, y_t)
  - Goal position g = (x_g, y_g)
  - Human positions {p_h(t)} within the robot's observation field
- Output:
  - A **reward function** r that maps each navigation state to a scalar reward signal used to train reinforcement learning policies (e.g., PPO)
- Objective:
  - Encourage goal-directed progress
  - Penalize collisions and unsafe proximity
  - Promote smooth, efficient, and socially compliant motion
## Evaluation Metrics
- SR (Success Rate): Percentage of episodes where the robot successfully reaches the goal within the time limit
- CR (Collision Rate): Percentage of episodes involving collisions with humans or obstacles
- TR (Timeout Rate): Percentage of episodes where the robot fails to reach the goal within the time limit
- NT (Navigation Time): Average time taken to reach the goal in successful episodes
- PL (Path Length): Average total distance traveled, including during collisions and timeouts
- ITR (Intrusion Time Ratio): Fraction of time the robot intrudes into humans' predicted positions, triggering danger events
- SD (Social Distance): Average minimum distance between the robot and nearby humans during navigation
"""

# ---------------------------------------------------------------------------
# D.5 Seed Function (adapted to compute_reward(state, memory) / RewardState)
# ---------------------------------------------------------------------------

D5_SEED_FUNCTION = '''def compute_reward(state, memory):
    """CrowdNav++-style seed (Appendix D.5), adapted to RewardState + memory."""
    success_reward = 10.0
    collision_penalty = -20.0
    pot_factor = 2.0
    if state.reaching_goal:
        return float(success_reward)
    if state.collision:
        return float(collision_penalty)
    # Potential-style progress toward goal (dense shaping via memory).
    dist = ((state.robot.px - state.robot.gx) ** 2 + (state.robot.py - state.robot.gy) ** 2) ** 0.5
    prev = memory.get("prev_dist")
    if prev is None:
        memory["prev_dist"] = dist
        return float(0.0)
    progress = float(prev) - dist
    memory["prev_dist"] = dist
    return float(pot_factor * progress)
'''


def format_d1_initial(
    *,
    func_name: str = "compute_reward",
    include_seed: bool = True,
    include_external_knowledge: bool = True,
    reflection: str = "",
) -> str:
    """Build the D.1 user prompt (system prompt is separate)."""
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
        reflection_block = f"Reflective guidance from prior generations:\n{reflection.strip()}\n"
    external_block = ""
    if include_external_knowledge:
        external_block = f"External knowledge:\n{D4_EXTERNAL_KNOWLEDGE}\n"
    return D1_USER_PROMPT.format(
        func_name=func_name,
        seed_block=seed_block,
        reflection_block=reflection_block,
        external_knowledge_block=external_block,
    )


D1_BATCH_USER_PROMPT = """Please write **{n} diverse** Python reward functions for a robot navigation task in a crowded environment.
Task Description:
- Each function's goal is to output a scalar reward value based on the robot's current state, guiding it to its goal while avoiding collisions with dynamic human agents.
Function Interface (same for every function):
- Inputs:
  - state: A RewardState snapshot for the current frame with ONLY these fields:
    - state.robot.px, state.robot.py, state.robot.vx, state.robot.vy, state.robot.radius, state.robot.gx, state.robot.gy, state.robot.v_pref
    - state.humans: tuple of HumanObservable; iterate `for human in state.humans:` then human.px/py/vx/vy/radius (never index HumanObservable)
    - state.dmin, state.discomfort_dist (TOP-LEVEL), state.collision, state.reaching_goal, state.timeout
    - state.action, state.time_step, state.global_time, state.time_limit
    - NO state.history / state.prev_state
  - memory: plain mutable dict cleared each episode; use for progress shaping (no classes)
- Math: math module is not importable; use ** 0.5 for square roots. Do not use getattr, hasattr, or __import__.
- Output:
  - A single scalar (float) representing the reward for the current state or action.
Design Principles:
- Goal-Progress: reward progress toward the robot goal.
- Collision Avoidance: penalize unsafe proximity and collisions.
- Diversity: the {n} functions must differ in structure and hyperparameters (not trivial renames).
Constraints (CRITICAL):
- Hyperparameters must be local variables inside each function body (no extra arguments beyond state, memory).
- Define exactly {n} top-level functions. Name them ``{func_name}_v1``, ``{func_name}_v2``, ... ``{func_name}_v{n}`` (each takes (state, memory) and returns a finite float).
- Do **not** use import statements (math is not importable — prefer ** 0.5), classes, while loops, or reflection builtins (getattr, hasattr, __import__, eval, type, ...).
- Access RewardState only via dot notation (state.robot.px, state.dmin, state.humans, ...); use memory for cross-step state.
- Scalars are plain floats/bools: never write ``state.robot.px[0]`` / ``state.dmin[0]`` / ``human.vx[0]``.
- Output Format: return **only** Python code in a single fenced code block. No prose outside the block.
{seed_block}
{reflection_block}
{external_knowledge_block}
"""


def format_d1_initial_batch(
    n: int,
    *,
    func_name: str = "compute_reward",
    include_seed: bool = True,
    include_external_knowledge: bool = True,
    reflection: str = "",
) -> str:
    """Appendix D.1 batch variant: one LLM call proposes ``n`` diverse functions."""
    if n < 1:
        raise ValueError("batch size n must be >= 1")
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
        reflection_block = f"Reflective guidance from prior generations:\n{reflection.strip()}\n"
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
    """
    Build D.2 mutation prompt for an underperforming parent (§4.2).

    ``elitist_code`` is accepted as a deprecated alias of ``parent_code`` for
    older call sites / tests.
    """
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
    """
    Stage II/III sandbox-failure repair prompt.

    Must stay aligned with AST policy: no import statements (math is
    pre-injected), no getattr/hasattr/__import__, always return a finite float.
    """
    return (
        "The following reward function failed validation with this error:\n\n"
        f"ERROR: {validation_error}\n\n"
        f"ORIGINAL CODE:\n{bad_code}\n\n"
        "Please fix the code to pass validation. Remember:\n"
        "- state.robot.px, state.robot.py, state.robot.vx, state.robot.vy, "
        "state.robot.radius, state.robot.gx, state.robot.gy, state.robot.v_pref\n"
        "- state.humans: iterate with `for human in state.humans:` then "
        "human.px/py/vx/vy/radius (never index HumanObservable like human[0])\n"
        "- state.dmin, state.discomfort_dist (TOP-LEVEL), "
        "state.collision, state.reaching_goal, state.timeout\n"
        "- Scalars are floats/bools: never index them "
        "(no state.robot.px[0], state.dmin[0], human.vx[0])\n"
        "- NO state.history, NO state.prev_state, NO state.obstacle_dist\n"
        "- math module is not importable: do NOT write `import math` or "
        "`__import__`; use ** 0.5 for square roots "
        "(or call pre-injected math.sqrt with no import)\n"
        "- Signature must be: def compute_reward(state, memory): "
        "(memory is a plain dict cleared each episode)\n"
        "- Always return a finite float (never None)\n"
        "- Forbidden: getattr, hasattr, __import__, eval, exec, type, classes, "
        "import/from statements\n\n"
        "Return only the corrected function in a Python code block."
    )
