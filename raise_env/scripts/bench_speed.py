#!/usr/bin/env python
"""
Speed benchmark for the RAISE CrowdNav pipeline (no source changes).

Every table row runs in a fresh worker subprocess with its own environment
(thread env vars, RAISE_FORCE_CUDA, progress mode). A worker executes one
discarded warmup and ``--repeats`` timed repetitions and appends one JSON line
per repetition to ``results/bench/raw.jsonl`` (append-only, fsync per line).

Instrumentation (rollout/update timers, env step counters, no-callback /
no-Monitor variants) is applied at runtime inside the worker process only.

Usage (from ``raise_env/``)::

    python scripts/bench_speed.py --phase info 0 1 2 3 4 5 report
    python scripts/bench_speed.py --phase 1 --only p1.threads.t4      # rerun one row
    python scripts/bench_speed.py --phase all --resume                # skip finished rows
    python scripts/bench_speed.py --phase 0 1 --quick                 # smoke-size budgets
    python scripts/bench_speed.py --list                              # print all rows
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import argparse  # noqa: E402
import base64  # noqa: E402
import copy  # noqa: E402
import csv  # noqa: E402
import ctypes  # noqa: E402
import fnmatch  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import uuid  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple  # noqa: E402

BENCH_DIR = os.path.join("results", "bench")
RAW_PATH = os.path.join(BENCH_DIR, "raw.jsonl")
WORK_DIR = os.path.join(BENCH_DIR, "_work")
MODEL_DIR = os.path.join(BENCH_DIR, "models")
LOG_DIR = os.path.join(BENCH_DIR, "logs")
PROFILE_DIR = os.path.join(BENCH_DIR, "profiles")
DATASET = "domains/crowdnav/data/stage1_dataset"
REF_RUN = os.path.join("results", "crowdnav_16h")
RUN_RAISE = os.path.join(_ROOT, "scripts", "run_raise.py")
SEED = 425
PHYSICAL_CORES_FALLBACK = max(1, (os.cpu_count() or 2) // 2)

# Heavier reward than the seed: a human loop with a closest-approach
# estimate, a short look-ahead loop and episode memory (passes the sandbox).
HEAVY_REWARD = '''def compute_reward(state, memory):
    r = state.robot
    total = 0.0
    nearest = 10.0
    ttc_pen = 0.0
    look_pen = 0.0
    for h in state.humans:
        dx = h.px - r.px
        dy = h.py - r.py
        dist = math.sqrt(dx * dx + dy * dy) - h.radius - r.radius
        if dist < nearest:
            nearest = dist
        dvx = h.vx - r.vx
        dvy = h.vy - r.vy
        a = dvx * dvx + dvy * dvy
        b = 2.0 * (dx * dvx + dy * dvy)
        if a > 1e-9 and b < 0.0:
            t = -b / (2.0 * a)
            cx = dx + dvx * t
            cy = dy + dvy * t
            miss = math.sqrt(cx * cx + cy * cy) - h.radius - r.radius
            if miss < 0.2 and t < 3.0:
                ttc_pen = ttc_pen + (0.2 - miss) * math.exp(-t)
        for k in range(4):
            fx = dx + dvx * 0.25 * k
            fy = dy + dvy * 0.25 * k
            fd = math.hypot(fx, fy)
            if fd < 0.8:
                look_pen = look_pen + 0.01 * (0.8 - fd)
    prev = memory.get("prev_goal", state.goal_distance)
    memory["prev_goal"] = state.goal_distance
    progress = prev - state.goal_distance
    speed = math.hypot(r.vx, r.vy)
    last_speed = memory.get("prev_speed", speed)
    memory["prev_speed"] = speed
    smooth = -0.01 * abs(speed - last_speed)
    terminal = 0.0
    if state.collision:
        terminal = -0.25
    elif state.reached_goal:
        terminal = 1.0
    total = terminal + 0.1 * progress - 0.05 * ttc_pen - look_pen + smooth
    if nearest < state.discomfort_dist:
        total = total - 0.05 * (state.discomfort_dist - nearest)
    return float(total), {
        "terminal": float(terminal),
        "progress": float(0.1 * progress),
        "ttc": float(-0.05 * ttc_pen),
        "lookahead": float(-look_pen),
        "smooth": float(smooth),
    }
'''

P5_COMMON = [
    "--llm", "seed", "--allow-seed-llm",
    "--population", "4", "--generations", "2",
    "--stage2-train-steps", "2000", "--stage3-train-steps", "4000",
    "--stage2-eval-episodes", "10", "--stage3-eval-episodes", "10",
    "--final-eval-episodes", "10", "--device", "cpu", "--no-resume",
]
P5_QUICK = [
    "--llm", "seed", "--allow-seed-llm",
    "--population", "2", "--generations", "1",
    "--stage2-train-steps", "256", "--stage3-train-steps", "256",
    "--stage2-eval-episodes", "2", "--stage3-eval-episodes", "2",
    "--final-eval-episodes", "2", "--device", "cpu", "--no-resume",
]
# Best class-A/D combination (chosen from phases 1-3; override with --p5-best-*).
# Phase 1 (p1.threads.*) and phase 3 (p3.thread.*) found no class-A/D setting faster
# than the defaults (torch_threads=1, label_workers=1), so p5.best is an A/A control.
P5_BEST_FLAGS: List[str] = ["--label-workers", "1", "--torch-threads", "1"]
P5_BEST_ENV: Dict[str, str] = {}
# Class-B comparison only (changes PPO n_steps/batch_size): not a free speedup.
P5_CLASSB_FLAGS: List[str] = ["--n-envs", "4"]


# ======================================================================= utils


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_raw(record: Dict[str, Any]) -> None:
    os.makedirs(BENCH_DIR, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str, allow_nan=True)
    with open(RAW_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _read_raw() -> List[Dict[str, Any]]:
    if not os.path.isfile(RAW_PATH):
        return []
    rows = []
    with open(RAW_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _proc_cpu_s() -> float:
    t = os.times()
    return float(t.user + t.system)


class _SPPI(ctypes.Structure):
    _fields_ = [
        ("IdleTime", ctypes.c_longlong),
        ("KernelTime", ctypes.c_longlong),
        ("UserTime", ctypes.c_longlong),
        ("DpcTime", ctypes.c_longlong),
        ("InterruptTime", ctypes.c_longlong),
        ("InterruptCount", ctypes.c_ulong),
    ]


def _cpu_snapshot() -> Optional[List[Tuple[int, int, int]]]:
    """Per-logical-core (idle, kernel, user) 100 ns ticks (Windows only)."""
    if os.name != "nt":
        return None
    try:
        n = int(os.cpu_count() or 1)
        arr = (_SPPI * n)()
        ret = ctypes.c_ulong(0)
        status = ctypes.windll.ntdll.NtQuerySystemInformation(
            8, ctypes.byref(arr), ctypes.sizeof(arr), ctypes.byref(ret)
        )
        if status != 0:
            return None
        return [(int(a.IdleTime), int(a.KernelTime), int(a.UserTime)) for a in arr]
    except Exception:  # noqa: BLE001
        return None


def _cpu_util(a: Any, b: Any) -> Optional[List[float]]:
    """Busy fraction per logical core between two snapshots (kernel time includes idle)."""
    if not a or not b:
        return None
    out = []
    for (i0, k0, u0), (i1, k1, u1) in zip(a, b):
        tot = (k1 - k0) + (u1 - u0)
        idle = i1 - i0
        out.append(0.0 if tot <= 0 else max(0.0, min(1.0, 1.0 - idle / tot)))
    return out


class _Usage:
    """Wall, process CPU and per-core system utilization over a code block."""

    def __enter__(self) -> "_Usage":
        self.snap0 = _cpu_snapshot()
        self.cpu0 = _proc_cpu_s()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.wall = time.perf_counter() - self.t0
        self.cpu = _proc_cpu_s() - self.cpu0
        self.per_core = _cpu_util(self.snap0, _cpu_snapshot())

    def as_dict(self, prefix: str = "") -> Dict[str, Any]:
        d: Dict[str, Any] = {
            f"{prefix}proc_cpu_s": round(self.cpu, 4),
            f"{prefix}proc_cpu_cores": round(self.cpu / max(self.wall, 1e-9), 3),
        }
        if self.per_core:
            d[f"{prefix}sys_util_mean"] = round(statistics.fmean(self.per_core), 4)
            d[f"{prefix}sys_busy_cores"] = round(sum(self.per_core), 3)
            d[f"{prefix}sys_util_per_core"] = [round(x, 3) for x in self.per_core]
        return d


def _physical_cores() -> int:
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum"],
                capture_output=True, text=True, timeout=60,
            ).stdout.strip()
            return int(out)
        except Exception:  # noqa: BLE001
            pass
    return PHYSICAL_CORES_FALLBACK


def _parse_dur(text: str) -> Optional[float]:
    """Inverse of ``raise_core.console.format_seconds``."""
    s = text.strip()
    m = re.fullmatch(r"(\d+)h(\d+)m", s)
    if m:
        return int(m.group(1)) * 3600.0 + int(m.group(2)) * 60.0
    m = re.fullmatch(r"(\d+)m([\d.]+)s", s)
    if m:
        return int(m.group(1)) * 60.0 + float(m.group(2))
    m = re.fullmatch(r"([\d.]+)s", s)
    if m:
        return float(m.group(1))
    return None


@contextmanager
def _main_module_as(path: Optional[str]):
    """Make multiprocessing spawn children re-import ``path`` as ``__mp_main__``.

    ``scripts/run_raise.py`` is the real ``__main__`` of a pipeline run, so its
    top-level imports are paid by every spawned smoke-test / Score1 child.
    """
    if not path:
        yield
        return
    main = sys.modules["__main__"]
    old_file = getattr(main, "__file__", None)
    old_spec = getattr(main, "__spec__", None)
    main.__file__ = os.path.abspath(path)
    main.__spec__ = None
    try:
        yield
    finally:
        main.__file__ = old_file
        main.__spec__ = old_spec


def _setup_logging(verbose: bool = False) -> None:
    from raise_core import console

    console.configure_run_logging(verbose=bool(verbose), output_dir=None)


def _apply_threads(threads: int) -> int:
    import torch

    if int(threads) > 0:
        torch.set_num_threads(int(threads))
    return int(torch.get_num_threads())


# ================================================================ candidates


def _reward_source(which: str) -> str:
    if which in ("seed", "seed_raw"):
        from domains.crowdnav.prompts import D5_SEED_FUNCTION

        return D5_SEED_FUNCTION
    if which in ("heavy", "heavy_raw"):
        return HEAVY_REWARD
    if which.startswith("variant"):
        from domains.crowdnav.prompts import seed_variant
        from raise_core.llm import extract_python_code, normalize_to_compute_reward

        n = int(which[len("variant"):])
        return normalize_to_compute_reward(extract_python_code(seed_variant(n)))
    raise ValueError(which)


def _validated_reward(st: Dict[str, Any], which: str) -> Any:
    """Validated SandboxedReward (cached per worker; validation spawns a child)."""
    key = f"rf_{which}"
    if key not in st:
        if "validator" not in st:
            from domains.crowdnav.reward_checks import make_crowdnav_validator

            st["validator"] = make_crowdnav_validator()
        st[key] = st["validator"].validate_code(_reward_source(which))
    return st[key]


class _RawReward:
    """Unsandboxed call of the compiled function (no timing / bounds bookkeeping)."""

    def __init__(self, sandboxed: Any) -> None:
        self._fn = sandboxed._compute_fn
        self._mem: Dict[str, Any] = {}

    def reset(self) -> None:
        self._mem.clear()

    def compute(self, state: Any) -> float:
        v = self._fn(state, self._mem)
        return float(v[0]) if isinstance(v, tuple) else float(v)


def _candidate(st: Dict[str, Any], which: str = "seed", cid: str = "bench_seed") -> Any:
    from raise_core.candidate import RewardCandidate

    rf = copy.deepcopy(_validated_reward(st, which))
    return RewardCandidate(
        candidate_id=cid, code=_reward_source(which), reward_fn=rf, valid=True, origin="bench"
    )


def _stage2_cfg(p: Dict[str, Any], out_root: str) -> Any:
    from raise_core.refine import Stage2Config

    instr = bool(p.get("instr", True))
    return Stage2Config(
        train_env_steps=int(p.get("steps", 3000)),
        eval_episodes=int(p.get("eval_episodes", 30)),
        seed=SEED,
        device=str(p.get("device", "cpu")),
        output_root=out_root,
        n_envs=int(p.get("n_envs", 1)),
        eval_mode=str(p.get("eval_mode", "holdout_only")),
        eval_split="val",
        action_mode=str(p.get("action_mode", "discrete")),
        robot_kinematics=str(p.get("kinematics", "holonomic")),
        eval_deterministic=bool(p.get("eval_deterministic", False)),
        eval_seed_base=SEED,
        record_training_dynamics=instr,
        record_reward_components=instr,
    )


def _policy_fingerprint(ckpt: str) -> Dict[str, Any]:
    """Hash + sum of the saved policy parameters (``policy.pth`` inside the SB3 zip)."""
    import zipfile

    import torch

    with zipfile.ZipFile(ckpt) as z:
        data = z.read("policy.pth")
    sd = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    h = hashlib.sha256()
    total = 0.0
    for k in sorted(sd):
        arr = sd[k].detach().cpu().double().numpy()
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
        total += float(arr.sum())
    return {"policy_hash": h.hexdigest()[:16], "param_sum": total}


def _cand_metrics(c: Any) -> Dict[str, Any]:
    md = c.metadata or {}
    lm = md.get("last_metrics") or {}
    eps = lm.get("holdout_episodes") or []
    return {
        "fitness": md.get("fitness"),
        "fitness_se": lm.get("fitness_se"),
        "SR": lm.get("SR"),
        "CR": lm.get("CR"),
        "TR": lm.get("TR"),
        "task_return": lm.get("task_return"),
        "train_wall_s": md.get("train_wall_seconds"),
        "eval_wall_s": md.get("eval_wall_seconds"),
        "actual_train_steps": md.get("actual_train_steps"),
        "ppo_n_steps": md.get("ppo_n_steps"),
        "eval_steps": int(sum(int(e.get("steps", 0)) for e in eps)),
        "eval_episodes_done": len(eps),
        "checkpoint_path": md.get("checkpoint_path"),
    }


# ======================================================= runtime patches


_PHASE_ACC: Dict[str, float] = {"rollout_s": 0.0, "update_s": 0.0, "n_rollouts": 0, "n_updates": 0}


def _install_phase_timers() -> None:
    """Wrap ``collect_rollouts`` / ``PPO.train`` with perf_counter accumulators."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

    if getattr(OnPolicyAlgorithm, "_bench_wrapped", False):
        return
    orig_c = OnPolicyAlgorithm.collect_rollouts
    orig_t = PPO.train

    def collect(self, *a, **k):
        t = time.perf_counter()
        try:
            return orig_c(self, *a, **k)
        finally:
            _PHASE_ACC["rollout_s"] += time.perf_counter() - t
            _PHASE_ACC["n_rollouts"] += 1

    def train(self, *a, **k):
        t = time.perf_counter()
        try:
            return orig_t(self, *a, **k)
        finally:
            _PHASE_ACC["update_s"] += time.perf_counter() - t
            _PHASE_ACC["n_updates"] += 1

    OnPolicyAlgorithm.collect_rollouts = collect
    PPO.train = train
    OnPolicyAlgorithm._bench_wrapped = True


_STEP_COUNT = {"n": 0}


def _install_step_counter() -> None:
    from domains.crowdnav.env_wrapper import RewardInjectedCrowdNavEnv

    if getattr(RewardInjectedCrowdNavEnv, "_bench_counted", False):
        return
    orig = RewardInjectedCrowdNavEnv.step_upstream

    def step_upstream(self, up_action):
        _STEP_COUNT["n"] += 1
        return orig(self, up_action)

    RewardInjectedCrowdNavEnv.step_upstream = step_upstream
    RewardInjectedCrowdNavEnv._bench_counted = True


def _install_variant_patches(p: Dict[str, Any]) -> None:
    """Bench-only variants that the CLI cannot express (proposal validation)."""
    import domains.crowdnav.adapter as adapter

    if p.get("patch_nocb"):
        from stable_baselines3.common.callbacks import BaseCallback

        def _cb(total_timesteps, desc, dynamics=None):
            if dynamics is None:
                return None

            class _Obs(BaseCallback):
                def _on_step(self) -> bool:
                    dynamics.observe(
                        self.locals.get("infos") or (), self.locals.get("dones"), int(self.num_timesteps)
                    )
                    return True

            return _Obs()

        adapter._ppo_progress_callback = _cb
    if p.get("patch_nomonitor"):
        adapter.Monitor = lambda env: env


# ============================================================ worker kinds


def k_calib(p: Dict[str, Any], st: Dict[str, Any], rep: int) -> Dict[str, Any]:
    """Fixed CPU workload (thermal / frequency drift detector) + idle background load."""
    import numpy as np

    pc = time.perf_counter
    t = pc()
    s = 0
    for i in range(3_000_000):
        s += i * i
    py_s = pc() - t
    a = np.random.default_rng(0).standard_normal((256, 256))
    t = pc()
    for _ in range(200):
        a = np.tanh(a @ a * 1e-3)
    np_s = pc() - t
    snap0 = _cpu_snapshot()
    time.sleep(1.0)
    per = _cpu_util(snap0, _cpu_snapshot()) or []
    return {
        "py_loop_s": round(py_s, 4),
        "numpy_matmul_s": round(np_s, 4),
        "idle_bg_util_mean": round(statistics.fmean(per), 4) if per else None,
        "idle_bg_busy_cores": round(sum(per), 3) if per else None,
    }


def k_machine(p: Dict[str, Any], st: Dict[str, Any], rep: int) -> Dict[str, Any]:
    import gymnasium
    import numpy
    import stable_baselines3
    import torch

    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "logical_cores": os.cpu_count(),
        "physical_cores": _physical_cores(),
        "torch": torch.__version__,
        "torch_default_threads": torch.get_num_threads(),
        "torch_default_interop_threads": torch.get_num_interop_threads(),
        "torch_config_parallel": torch.__config__.parallel_info().splitlines()[:6],
        "sb3": stable_baselines3.__version__,
        "numpy": numpy.__version__,
        "gymnasium": gymnasium.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_cuda": torch.version.cuda,
        "env_OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "env_MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    }
    for mod in ("tiktoken", "tqdm", "matplotlib"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    if os.name == "nt":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            info["cpu_model"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            info["cpu_mhz_nominal"] = winreg.QueryValueEx(key, "~MHz")[0]
        except Exception:  # noqa: BLE001
            info["cpu_model"] = platform.processor()

        class _MEMSTAT(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        ms = _MEMSTAT()
        ms.dwLength = ctypes.sizeof(_MEMSTAT)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        info["ram_gib"] = round(ms.ullTotalPhys / 2**30, 1)
        info["ram_avail_gib"] = round(ms.ullAvailPhys / 2**30, 1)

        class _PWR(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong),
            ]

        pw = _PWR()
        ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(pw))
        info["ac_power"] = {0: False, 1: True}.get(int(pw.ACLineStatus))
        try:
            info["power_scheme"] = subprocess.run(
                ["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=30
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            info["power_scheme"] = None
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True
        ).stdout.strip()
        info["git_rev"] = rev
        info["git_dirty_tracked"] = bool(dirty)
    except Exception:  # noqa: BLE001
        pass
    os.makedirs(BENCH_DIR, exist_ok=True)
    with open(os.path.join(BENCH_DIR, "machine.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    return info


# ---------------------------------------------------------------- phase 0


def k_env_throughput(p: Dict[str, Any], st: Dict[str, Any], rep: int) -> Dict[str, Any]:
    """Random-policy steps/s of RewardInjectedCrowdNavEnv for one reward variant."""
    import numpy as np

    from domains.crowdnav.env_wrapper import RewardInjectedCrowdNavEnv

    which = str(p["reward"])
    if which == "none":
        rf = None
    elif which.endswith("_raw"):
        rf = _RawReward(_validated_reward(st, which))
    else:
        rf = copy.deepcopy(_validated_reward(st, which))
    t_ctor = time.perf_counter()
    env = RewardInjectedCrowdNavEnv(rf, phase="train", seed=1)
    ctor_s = time.perf_counter() - t_ctor
    rng = np.random.default_rng(0)
    n_act = int(env.action_space.n)
    steps = int(p["steps"])
    env.reset()
    episodes = 0
    rsum = 0.0
    with _Usage() as u:
        for _ in range(steps):
            _o, r, te, tr, _i = env.step(int(rng.integers(n_act)))
            rsum += r
            if te or tr:
                episodes += 1
                env.reset()
    env.close()
    return {
        "steps": steps,
        "wall_s": round(u.wall, 4),
        "steps_per_s": round(steps / u.wall, 1),
        "us_per_step": round(1e6 * u.wall / steps, 2),
        "episodes": episodes,
        "reward_sum": round(rsum, 6),
        "env_ctor_s": round(ctor_s, 4),
        **u.as_dict(),
    }


def k_step_decomp(p: Dict[str, Any], st: Dict[str, Any], rep: int) -> Dict[str, Any]:
    """Per-segment cost of one env step (replica of ``step_upstream``, same calls)."""
    import numpy as np

    from domains.crowdnav.env_wrapper import (
        RewardInjectedCrowdNavEnv,
        outcome_of,
        robot_centric_observation,
        step_min_separation,
    )
    from crowd_sim.envs.utils.info import Collision, Danger, ReachGoal, Timeout  # after raise_paths
    from domains.crowdnav.state import CrowdNavRewardState, HumanState, RobotState

    rf = copy.deepcopy(_validated_reward(st, "seed"))
    env = RewardInjectedCrowdNavEnv(rf, phase="train", seed=1)
    rng = np.random.default_rng(0)
    n_act = int(env.action_space.n)
    steps = int(p["steps"])
    keys = ["to_upstream", "min_separation", "crowdsim_step", "bookkeeping",
            "state_construction", "reward_compute", "info_dict", "observation", "reset"]
    acc = dict.fromkeys(keys, 0.0)
    pc = time.perf_counter
    env.reset()
    rsum = 0.0
    t_all = pc()
    for _ in range(steps):
        a = int(rng.integers(n_act))
        t0 = pc()
        up = env.to_upstream_action(a)
        t1 = pc()
        sim = env.sim
        dmin = step_min_separation(sim, up)
        t2 = pc()
        prev_dist = env._goal_distance
        _ob, native_reward, _done, info_obj = sim.step(up)
        t3 = pc()
        timeout = isinstance(info_obj, Timeout)
        collision = isinstance(info_obj, Collision)
        reached = isinstance(info_obj, ReachGoal)
        danger = isinstance(info_obj, Danger)
        robot = sim.robot
        goal_dist = math.hypot(robot.gx - robot.px, robot.gy - robot.py)
        env._goal_distance = goal_dist
        env._trail.append((float(robot.px), float(robot.py)))
        t4 = pc()
        state = CrowdNavRewardState(
            robot=RobotState(
                px=float(robot.px), py=float(robot.py), vx=float(robot.vx), vy=float(robot.vy),
                gx=float(robot.gx), gy=float(robot.gy), theta=float(robot.theta),
                radius=float(robot.radius), v_pref=float(robot.v_pref),
            ),
            humans=tuple(
                HumanState(px=float(h.px), py=float(h.py), vx=float(h.vx), vy=float(h.vy),
                           radius=float(h.radius))
                for h in sim.humans
            ),
            action=(float(up[0]), float(up[1])),
            collision=collision, reached_goal=reached, timeout=timeout,
            min_separation=float(dmin), discomfort_dist=env.discomfort_dist,
            goal_distance=float(goal_dist), progress=float(prev_dist - goal_dist),
            time_step=env.time_step, global_time=float(sim.global_time), time_limit=env.time_limit,
        )
        env.last_state = state
        t5 = pc()
        reward = float(rf.compute(state))
        raw = rf.last_reward_components()
        comps = {str(k): float(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
        t6 = pc()
        info = {
            "raise_outcome": outcome_of(info_obj), "raise_collision": collision,
            "raise_reached_goal": reached, "raise_timeout": timeout, "raise_danger": danger,
            "raise_danger_min_dist": float(info_obj.min_dist) if danger else None,
            "raise_min_separation": float(dmin), "raise_native_reward": float(native_reward),
            "raise_progress": float(state.progress),
            "raise_speed": math.hypot(state.robot.vx, state.robot.vy),
            "raise_goal_distance": float(goal_dist), "raise_global_time": float(sim.global_time),
            "raise_phase": env.current_phase, "raise_case": env.case,
            "raise_reward_components": comps,
        }
        t7 = pc()
        robot_centric_observation(sim)
        t8 = pc()
        acc["to_upstream"] += t1 - t0
        acc["min_separation"] += t2 - t1
        acc["crowdsim_step"] += t3 - t2
        acc["bookkeeping"] += t4 - t3
        acc["state_construction"] += t5 - t4
        acc["reward_compute"] += t6 - t5
        acc["info_dict"] += t7 - t6
        acc["observation"] += t8 - t7
        rsum += reward
        if collision or reached or timeout:
            t9 = pc()
            env.reset()
            acc["reset"] += pc() - t9
    total = pc() - t_all
    # Cost of the 9 perf_counter calls per step (subtracted from nothing; reported).
    t = pc()
    for _ in range(100_000):
        pc()
    timer_us = 1e6 * (pc() - t) / 100_000
    out = {f"{k}_us_per_step": round(1e6 * v / steps, 3) for k, v in acc.items()}
    out.update(
        steps=steps,
        total_us_per_step=round(1e6 * total / steps, 3),
        timer_overhead_us_per_step=round(9 * timer_us, 3),
        reward_sum=round(rsum, 6),
    )
    return out


def k_reward_micro(p: Dict[str, Any], st: Dict[str, Any], rep: int) -> Dict[str, Any]:
    """Per-call cost of SandboxedReward.compute vs the raw compiled function."""
    import numpy as np

    from domains.crowdnav.env_wrapper import RewardInjectedCrowdNavEnv

    if "states" not in st:
        env = RewardInjectedCrowdNavEnv(None, phase="train", seed=1)
        rng = np.random.default_rng(0)
        env.reset()
        states = []
        while len(states) < 3000:
            _o, _r, te, tr, _i = env.step(int(rng.integers(int(env.action_space.n))))
            states.append(env.last_state)
            if te or tr:
                env.reset()
        st["states"] = states
    states = st["states"]
    loops = int(p.get("loops", 10))
    out: Dict[str, Any] = {"n_calls": loops * len(states)}
    pc = time.perf_counter
    for which in ("seed", "heavy"):
        sbx = copy.deepcopy(_validated_reward(st, which))
        raw = _RawReward(sbx)
        for label, obj in (("sandboxed", sbx), ("raw", raw)):
            obj.reset()
            t = pc()
            for _ in range(loops):
                for s in states:
                    obj.compute(s)
            dt = pc() - t
            out[f"{which}_{label}_us_per_call"] = round(1e6 * dt / (loops * len(states)), 3)
        out[f"{which}_sandbox_overhead_us"] = round(
            out[f"{which}_sandboxed_us_per_call"] - out[f"{which}_raw_us_per_call"], 3
        )
        out[f"{which}_fraction_of_20ms_budget"] = round(
            out[f"{which}_sandboxed_us_per_call"] / 20000.0, 6
        )
    t = pc()
    for _ in range(100_000):
        pc()
        pc()
    out["perf_counter_pair_us"] = round(1e6 * (pc() - t) / 100_000, 4)
    return out


_PROFILE_KEYS = [
    ("learn_total", "on_policy_algorithm.py", "learn", "ct"),
    ("collect_rollouts", "on_policy_algorithm.py", "collect_rollouts", "ct"),
    ("ppo_train_update", "stable_baselines3/ppo/ppo.py", "train", "ct"),
    ("policy_forward_rollout", "stable_baselines3/common/policies.py", "forward", "ct"),
    ("vecenv_step_wait", "dummy_vec_env.py", "step_wait", "ct"),
    ("monitor_step_self", "stable_baselines3/common/monitor.py", "step", "tt"),
    ("monitor_step_cum", "stable_baselines3/common/monitor.py", "step", "ct"),
    ("env_step_upstream", "domains/crowdnav/env_wrapper.py", "step_upstream", "ct"),
    ("crowdsim_step", "crowd_sim/envs/crowd_sim.py", "step", "ct"),
    ("orca_predict", "crowd_sim/envs/policy/orca.py", "predict", "ct"),
    ("step_min_separation", "domains/crowdnav/env_wrapper.py", "step_min_separation", "ct"),
    ("robot_centric_observation", "domains/crowdnav/env_wrapper.py", "robot_centric_observation", "ct"),
    ("reward_sandbox_compute", "raise_core/sandbox/runtime.py", "compute", "ct"),
    ("reward_user_fn", "<reward_candidate>", "compute_reward", "ct"),
    ("progress_callback_on_step", "domains/crowdnav/adapter.py", "_on_step", "ct"),
    ("dynamics_observe", "domains/crowdnav/training_dynamics.py", "observe", "ct"),
    ("console_plain_update", "raise_core/console.py", "update", "ct"),
    ("tqdm_update", "tqdm/std.py", "update", "ct"),
    ("sb3_callback_on_step", "stable_baselines3/common/callbacks.py", "on_step", "ct"),
    ("eval_total", "domains/crowdnav/adapter.py", "_evaluate", "ct"),
    ("model_save", "stable_baselines3/common/base_class.py", "save", "ct"),
    ("ppo_init", "stable_baselines3/ppo/ppo.py", "__init__", "ct"),
    ("make_train_vec_env", "domains/crowdnav/adapter.py", "_make_train_vec_env", "ct"),
    ("torch_optimizer_step", "torch/optim/adam.py", "step", "ct"),
    ("torch_backward", "torch/_tensor.py", "backward", "ct"),
]


def _profile_extract(stats: Any) -> Dict[str, float]:
    out = {k: 0.0 for k, *_ in _PROFILE_KEYS}
    for (fn, _ln, name), (_cc, _nc, tt, ct, _callers) in stats.stats.items():
        path = fn.replace("\\", "/")
        for key, sub, func, field in _PROFILE_KEYS:
            if name == func and (path.endswith(sub) or path == sub):
                out[key] += tt if field == "tt" else ct
    out["profile_total"] = float(stats.total_tt)
    return out


def k_profile(p: Dict[str, Any], st: Dict[str, Any], rep: int) -> Dict[str, Any]:
    """cProfile of one Stage II train_and_eval (seed reward)."""
    import cProfile
    import pstats

    from domains.crowdnav.adapter import CrowdNavPPOTrainer

    out_root = os.path.join(WORK_DIR, "p0_profile")
    cand = _candidate(st, "seed")
    cfg = _stage2_cfg(p, out_root)
    trainer = CrowdNavPPOTrainer(stage="stage2")
    if rep < int(p.get("_warmup", 1)):
        t = time.perf_counter()
        trainer.train_and_eval(cand, round_index=0, config=cfg)
        return {"unprofiled_wall_s": round(time.perf_counter() - t, 4)}
    prof = cProfile.Profile()
    t = time.perf_counter()
    prof.enable()
    trainer.train_and_eval(cand, round_index=0, config=cfg)
    prof.disable()
    wall = time.perf_counter() - t
    os.makedirs(PROFILE_DIR, exist_ok=True)
    stem = os.path.join(PROFILE_DIR, f"{p['_row_id']}_rep{rep}")
    prof.dump_stats(stem + ".prof")
    stats = pstats.Stats(prof)
    with open(stem + "_top.txt", "w", encoding="utf-8") as fh:
        for key in ("cumulative", "tottime"):
            buf = io.StringIO()
            pstats.Stats(prof, stream=buf).sort_stats(key).print_stats(25)
            fh.write(f"===== sorted by {key} =====\n{buf.getvalue()}\n")
    vals = _profile_extract(stats)
    learn = vals["learn_total"] or 1e-9
    out: Dict[str, Any] = {"profiled_wall_s": round(wall, 4), "top_file": stem + "_top.txt"}
    for k, v in vals.items():
        out[f"{k}_s"] = round(v, 4)
        if k != "profile_total":
            out[f"{k}_share_of_learn"] = round(v / learn, 4)
    out.update(_cand_metrics(cand))
    return out


def _noop() -> None:
    return None


def k_cold_start(p: Dict[str, Any], st: Dict[str, Any], rep: int) -> Dict[str, Any]:
    """Cold-start costs: spawn, validation, Score1 worker, PPO.load, env construction."""
    import multiprocessing as mp

    from domains.crowdnav.env_wrapper import RewardInjectedCrowdNavEnv
    from domains.crowdnav.prompts import D5_SEED_FUNCTION
    from domains.crowdnav.reward_checks import make_crowdnav_validator
    from domains.crowdnav.score1_worker import make_score_fn
    from domains.crowdnav.stage1 import load_crowdnav_dataset, score_crowdnav_dataset

    out: Dict[str, Any] = {}
    pc = time.perf_counter
    ctx = mp.get_context("spawn")
    for label, main in (("light_main", None), ("run_raise_main", RUN_RAISE)):
        with _main_module_as(main):
            t = pc()
            proc = ctx.Process(target=_noop)
            proc.start()
            proc.join()
            out[f"spawn_noop_{label}_s"] = round(pc() - t, 4)
            validator = make_crowdnav_validator()
            t = pc()
            rf = validator.validate_code(D5_SEED_FUNCTION)
            out[f"validate_seed_{label}_s"] = round(pc() - t, 4)
    t = pc()
    for _ in range(5):
        RewardInjectedCrowdNavEnv(None, phase="train").close()
    out["env_ctor_s"] = round((pc() - t) / 5, 5)

    t = pc()
    ds = load_crowdnav_dataset(DATASET)
    out["dataset_load_inprocess_s"] = round(pc() - t, 4)
    out["dataset_frames"] = int(sum(len(x.states) for x in ds))
    t = pc()
    score_crowdnav_dataset(rf, ds, candidate_id="seed")
    out["score1_inprocess_s"] = round(pc() - t, 4)
    for label, main in (("light_main", None), ("run_raise_main", RUN_RAISE)):
        with _main_module_as(main):
            t = pc()
            worker, _ = make_score_fn(mode="dataset", dataset_path=DATASET)
            out[f"score1_make_fn_{label}_s"] = round(pc() - t, 4)
            t = pc()
            worker(rf, candidate_id="seed")
            out[f"score1_first_call_{label}_s"] = round(pc() - t, 4)
            t = pc()
            worker(rf, candidate_id="seed")
            out[f"score1_warm_call_{label}_s"] = round(pc() - t, 4)
            worker.close()

    from stable_baselines3 import PPO

    model_path = _ensure_model(st, steps=int(p.get("model_steps", 12000)))
    t = pc()
    PPO.load(model_path, device="cpu")
    out["ppo_load_s"] = round(pc() - t, 4)

    code = (
        "import time,json,runpy,sys;t0=time.perf_counter();import torch;t1=time.perf_counter();"
        "import stable_baselines3;t2=time.perf_counter();import domains.crowdnav.adapter;"
        "t3=time.perf_counter();sys.argv=['run_raise.py'];runpy.run_path(r'%s',run_name='__mp_main__');"
        "t4=time.perf_counter();print(json.dumps({'torch':t1-t0,'sb3':t2-t1,'adapter':t3-t2,"
        "'run_raise_main_after_adapter':t4-t3}))" % RUN_RAISE
    )
    t = pc()
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=_ROOT)
    out["fresh_import_process_s"] = round(pc() - t, 4)
    try:
        parts = json.loads(res.stdout.strip().splitlines()[-1])
        for k, v in parts.items():
            out[f"import_{k}_s"] = round(float(v), 4)
    except Exception:  # noqa: BLE001
        out["import_error"] = res.stderr[-500:]
    code2 = (
        "import time,json,runpy,sys;t0=time.perf_counter();sys.argv=['run_raise.py'];"
        "runpy.run_path(r'%s',run_name='__mp_main__');print(json.dumps(time.perf_counter()-t0))" % RUN_RAISE
    )
    res = subprocess.run([sys.executable, "-c", code2], capture_output=True, text=True, cwd=_ROOT)
    try:
        out["import_run_raise_main_alone_s"] = round(float(res.stdout.strip().splitlines()[-1]), 4)
    except Exception:  # noqa: BLE001
        out["import_run_raise_error"] = res.stderr[-500:]
    return out


def _ensure_model(st: Dict[str, Any], steps: int = 12000) -> str:
    """Seed-reward PPO policy trained once with K2 steps (threads=1, seed 425)."""
    path = os.path.join(MODEL_DIR, f"seed_k{steps}.zip")
    if os.path.isfile(path):
        return path
    from domains.crowdnav.adapter import CrowdNavPPOTrainer

    os.makedirs(MODEL_DIR, exist_ok=True)
    cand = _candidate(st, "seed")
    cfg = _stage2_cfg({"steps": steps, "eval_episodes": 1}, os.path.join(WORK_DIR, "model_build"))
    CrowdNavPPOTrainer(stage="stage2").train_and_eval(cand, round_index=0, config=cfg)
    shutil.copyfile(cand.metadata["checkpoint_path"], path)
    return path


# ---------------------------------------------------------------- phase 1


def k_ppo_train(p: Dict[str, Any], st: Dict[str, Any], rep: int) -> Dict[str, Any]:
    """One Stage II train_and_eval with rollout / update split."""
    from domains.crowdnav.adapter import CrowdNavPPOTrainer

    _install_phase_timers()
    _install_variant_patches(p)
    out_root = os.path.join(WORK_DIR, "p1", str(p["_row_id"]))
    cand = _candidate(st, str(p.get("reward", "seed")))
    cfg = _stage2_cfg(p, out_root)
    for k in list(_PHASE_ACC):
        _PHASE_ACC[k] = 0
    trainer = CrowdNavPPOTrainer(stage="stage2")
    with _Usage() as u:
        trainer.train_and_eval(cand, round_index=0, config=cfg)
    m = _cand_metrics(cand)
    train_wall = float(m["train_wall_s"] or 0.0)
    out: Dict[str, Any] = {
        "wall_s": round(u.wall, 4),
        "rollout_s": round(_PHASE_ACC["rollout_s"], 4),
        "update_s": round(_PHASE_ACC["update_s"], 4),
        "learn_other_s": round(train_wall - _PHASE_ACC["rollout_s"] - _PHASE_ACC["update_s"], 4),
        "n_rollouts": int(_PHASE_ACC["n_rollouts"]),
        "steps_per_s_actual": round(float(m["actual_train_steps"] or 0) / max(train_wall, 1e-9), 1),
        "rollout_steps_per_s": round(float(m["actual_train_steps"] or 0) / max(_PHASE_ACC["rollout_s"], 1e-9), 1),
        "eval_steps_per_s": round(m["eval_steps"] / max(float(m["eval_wall_s"] or 1e-9), 1e-9), 1),
        "outside_learn_s": round(u.wall - train_wall - float(m["eval_wall_s"] or 0.0), 4),
        **m,
        **u.as_dict(),
    }
    out.update(_policy_fingerprint(m["checkpoint_path"]))
    if not p.get("keep_ckpt"):
        shutil.rmtree(out_root, ignore_errors=True)
    return out


# ---------------------------------------------------------------- phase 2


def k_eval(p: Dict[str, Any], st: Dict[str, Any], rep: int) -> Dict[str, Any]:
    """Stage II evaluation (30 episodes) of a fixed K2 policy."""
    import numpy as np
    import torch as th
    from stable_baselines3 import PPO

    from domains.crowdnav.adapter import CrowdNavPPOTrainer
    from domains.crowdnav.env_wrapper import RewardInjectedCrowdNavEnv
    from domains.crowdnav.settings import set_settings, settings_from_config

    _install_step_counter()
    model_path = _ensure_model(st, steps=int(p.get("model_steps", 12000)))
    if "model" not in st:
        st["model"] = PPO.load(model_path, device="cpu")
    model = st["model"]
    mode = str(p["mode"])
    cand = _candidate(st, "seed")
    cfg = _stage2_cfg(
        {"eval_episodes": int(p.get("eval_episodes", 30)),
         "eval_mode": "both" if mode == "both_stoch" else "holdout_only",
         "eval_deterministic": mode == "holdout_det"},
        os.path.join(WORK_DIR, "p2"),
    )
    settings = set_settings(settings_from_config(cfg))
    trainer = CrowdNavPPOTrainer(stage="stage2")
    pc = time.perf_counter
    if mode == "predict_micro":
        env = RewardInjectedCrowdNavEnv(cand.reward_fn, phase="val", settings=settings)
        obs, _ = env.reset(options={"case": 0})
        obs_list = []
        t_env = 0.0
        while len(obs_list) < 3000:
            obs_list.append(obs)
            a, _ = model.predict(obs, deterministic=False)
            t = pc()
            obs, _r, te, tr, _i = env.step(a)
            t_env += pc() - t
            if te or tr:
                obs, _ = env.reset(options={"case": len(obs_list) % 100})
        n = len(obs_list)
        model.set_random_seed(SEED)
        t = pc()
        for o in obs_list:
            model.predict(o, deterministic=False)
        us_predict = 1e6 * (pc() - t) / n
        t = pc()
        with th.no_grad():
            for o in obs_list:
                model.policy._predict(th.as_tensor(o[None], dtype=th.float32), deterministic=False)
        us_policy = 1e6 * (pc() - t) / n
        batch = np.stack(obs_list)
        t = pc()
        for i in range(0, n, 30):
            model.predict(batch[i:i + 30], deterministic=False)
        us_batched = 1e6 * (pc() - t) / n
        return {
            "n_obs": n,
            "predict_us_per_call_batch1": round(us_predict, 2),
            "policy_predict_nograd_us_batch1": round(us_policy, 2),
            "predict_us_per_obs_batch30": round(us_batched, 2),
            "env_step_us": round(1e6 * t_env / n, 2),
            "predict_overhead_us": round(us_predict - us_policy, 2),
            "torch_threads": th.get_num_threads(),
        }
    _STEP_COUNT["n"] = 0
    with _Usage() as u:
        if mode == "reevaluate":
            cand.metadata = {"checkpoint_path": model_path}
            trainer.reevaluate(cand, round_index=0, config=cfg)
        else:
            trainer._evaluate(model, cand, config=cfg, seed=SEED, settings=settings, stage_tag="bench")
    steps = int(_STEP_COUNT["n"])
    eps = int(cfg.eval_episodes) * (2 if mode == "both_stoch" else 1)
    lm = (cand.metadata or {}).get("last_metrics") or {}
    return {
        "wall_s": round(u.wall, 4),
        "episodes": eps,
        "env_steps": steps,
        "episodes_per_s": round(eps / u.wall, 2),
        "steps_per_s": round(steps / u.wall, 1),
        "ms_per_episode": round(1e3 * u.wall / eps, 2),
        "us_per_step": round(1e6 * u.wall / max(steps, 1), 2),
        "fitness": lm.get("fitness"),
        "SR": lm.get("SR"),
        "CR": lm.get("CR"),
        "TR": lm.get("TR"),
        **u.as_dict(),
    }


# ---------------------------------------------------------------- phase 3


def _proc_init(threads: int) -> None:
    if int(threads) > 0:
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
    import torch

    if int(threads) > 0:
        torch.set_num_threads(int(threads))
    import domains.crowdnav.adapter  # noqa: F401
    import raise_core.refine  # noqa: F401
    _setup_logging(False)


def _proc_ping(seconds: float) -> int:
    time.sleep(float(seconds))
    return os.getpid()


def _proc_label(cand: Any, round_index: int, cfg: Any) -> Dict[str, Any]:
    """One candidate per process task: the same trainer.train_and_eval call as threads."""
    from domains.crowdnav.adapter import make_stage2_trainer

    c0 = _proc_cpu_s()
    t0 = time.perf_counter()
    make_stage2_trainer().train_and_eval(cand, round_index=int(round_index), config=cfg)
    return {
        "metadata": cand.metadata,
        "cpu_s": _proc_cpu_s() - c0,
        "wall_s": time.perf_counter() - t0,
        "pid": os.getpid(),
    }


def k_label(p: Dict[str, Any], st: Dict[str, Any], rep: int) -> Dict[str, Any]:
    """Label N candidates with label_candidates_parallel (threads) or a process pool."""
    from domains.crowdnav.adapter import make_stage2_trainer
    from raise_core.raise_loop.parallel_label import label_candidates_parallel

    workers = int(p["workers"])
    threads = int(p["threads"])
    executor = str(p["executor"])
    n_cand = int(p.get("n_cand", 6))
    out_root = os.path.join(WORK_DIR, "p3", str(p["_row_id"]), f"rep{rep}")
    shutil.rmtree(out_root, ignore_errors=True)
    cfg = _stage2_cfg(p, out_root)
    cands = [_candidate(st, f"variant{i + 1}", cid=f"v{i + 1}") for i in range(n_cand)]
    out: Dict[str, Any] = {"workers": workers, "threads": threads, "executor": executor}
    per_cand_cpu: List[float] = []
    if executor == "thread":
        trainer = make_stage2_trainer()

        def label_one(c: Any, ri: int) -> Dict[str, Any]:
            trainer.train_and_eval(c, round_index=ri, config=cfg)
            return {"status": "ok"}

        order: List[str] = []
        with _Usage() as u:
            n_ok, n_fail = label_candidates_parallel(
                cands, already_done=set(), workers=workers, label_one=label_one,
                on_done=lambda c, r, i: order.append(str(c.candidate_id)),
                round_index_fn=lambda i: i,
            )
        out.update(n_ok=n_ok, n_fail=n_fail)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        ctx = mp.get_context("spawn")
        pool = None
        if executor == "proc_warm":
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                       initializer=_proc_init, initargs=(threads,))
            pids = {f.result() for f in [pool.submit(_proc_ping, 3.0) for _ in range(workers)]}
            out["warm_pids"] = len(pids)
        with _Usage() as u:
            if pool is None:
                pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                           initializer=_proc_init, initargs=(threads,))
            futs = [pool.submit(_proc_label, c, i, cfg) for i, c in enumerate(cands)]
            results = [f.result() for f in futs]
            pool.shutdown(wait=True)
        n_fail = 0
        for c, r in zip(cands, results):
            c.metadata = r["metadata"]
            per_cand_cpu.append(float(r["cpu_s"]))
        out.update(n_ok=len(results), n_fail=n_fail, child_task_cpu_s=round(sum(per_cand_cpu), 3),
                   n_distinct_pids=len({r["pid"] for r in results}))
    out["wall_s"] = round(u.wall, 4)
    out.update(u.as_dict())
    if per_cand_cpu:
        out["total_cpu_cores_incl_children"] = round(
            (u.cpu + sum(per_cand_cpu)) / max(u.wall, 1e-9), 3
        )
    sum_train = 0.0
    sum_eval = 0.0
    per: Dict[str, Any] = {}
    for c in cands:
        m = _cand_metrics(c)
        sum_train += float(m["train_wall_s"] or 0.0)
        sum_eval += float(m["eval_wall_s"] or 0.0)
        fp = _policy_fingerprint(m["checkpoint_path"]) if m["checkpoint_path"] else {}
        per[str(c.candidate_id)] = {
            "fitness": m["fitness"], "SR": m["SR"], "CR": m["CR"],
            "train_wall_s": m["train_wall_s"], "eval_wall_s": m["eval_wall_s"], **fp,
        }
    out["sum_cand_train_s"] = round(sum_train, 3)
    out["sum_cand_eval_s"] = round(sum_eval, 3)
    out["per_candidate"] = per
    out["fitness_vector"] = [per[k]["fitness"] for k in sorted(per)]
    out["hash_vector"] = [per[k].get("policy_hash") for k in sorted(per)]
    shutil.rmtree(out_root, ignore_errors=True)
    return out


# ---------------------------------------------------------------- phase 4


def k_gen0(p: Dict[str, Any], st: Dict[str, Any], rep: int) -> Dict[str, Any]:
    """Gen0 proposal + validation of N seed-LLM candidates, then Score1 cold / warm."""
    from raise_core.domains import load_domain, make_validator_for_domain
    from raise_core.explore import StageIConfig, StageIEvolver
    from raise_core.feedback import FeedbackCardConfig
    from raise_core.llm import make_llm_client

    n = int(p.get("population", 6))
    out_dir = os.path.join(WORK_DIR, "p4_gen0", str(p["_row_id"]), f"rep{rep}")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    pack = load_domain("crowdnav")
    validator = make_validator_for_domain(pack)
    durations: List[float] = []
    orig = validator.try_validate

    def timed_validate(code: str):
        t = time.perf_counter()
        try:
            return orig(code)
        finally:
            durations.append(time.perf_counter() - t)

    validator.try_validate = timed_validate
    llm = make_llm_client("seed", base_code=pack.seed_reward_source,
                          variant_fn=getattr(pack.prompts, "seed_variant", None))
    out: Dict[str, Any] = {}
    main = RUN_RAISE if p.get("main") == "run_raise" else None
    with _main_module_as(main):
        t = time.perf_counter()
        score_fn, _ds = pack.make_score_fn(mode="dataset", dataset_path=DATASET)
        out["score_fn_make_s"] = round(time.perf_counter() - t, 4)
        evolver = StageIEvolver(
            llm, score_fn=score_fn, validator=validator,
            config=StageIConfig(population_size=n, generations=4, seed=SEED),
            prompts=pack.prompts,
            rejection_log_path=os.path.join(out_dir, "stage1_rejections.jsonl"),
            feedback=FeedbackCardConfig(), feedback_card=pack.feedback_card, llm_log_dir=out_dir,
        )
        with _Usage() as u_init:
            pop = evolver.initialize_population()
        with _Usage() as u_cold:
            evolver.score_population(pop)
        with _Usage() as u_warm:
            evolver.score_population(pop)
        close = getattr(score_fn, "close", None)
        if callable(close):
            close()
    out.update(
        population=len(pop),
        n_validations=len(durations),
        gen0_init_s=round(u_init.wall, 4),
        gen0_init_per_accepted_s=round(u_init.wall / max(1, len(pop)), 4),
        validate_median_s=round(statistics.median(durations), 4) if durations else None,
        validate_min_s=round(min(durations), 4) if durations else None,
        validate_max_s=round(max(durations), 4) if durations else None,
        validate_sum_s=round(sum(durations), 4),
        gen0_init_proc_cpu_cores=round(u_init.cpu / max(u_init.wall, 1e-9), 3),
        score1_cold_s=round(u_cold.wall, 4),
        score1_warm_s=round(u_warm.wall, 4),
        score1_warm_per_candidate_s=round(u_warm.wall / max(1, len(pop)), 4),
        score1_warm_proc_cpu_cores=round(u_warm.cpu / max(u_warm.wall, 1e-9), 3),
    )
    if u_init.per_core:
        out["gen0_init_sys_busy_cores"] = round(sum(u_init.per_core), 3)
    return out


def k_feedback(p: Dict[str, Any], st: Dict[str, Any], rep: int) -> Dict[str, Any]:
    """Feedback-card render + token counting on the archive of a real run."""
    from domains.crowdnav import feedback_card as fc
    from domains.crowdnav.prompts import D1_SYSTEM_PROMPT
    from raise_core.feedback import CandidateArchive, FeedbackCardConfig
    from raise_core.token_budget import (
        HEURISTIC_MARGIN,
        TokenCounter,
        fit_to_budget,
        make_token_counter,
    )

    pc = time.perf_counter
    t = pc()
    archive = CandidateArchive.from_epoch_files(os.path.join(REF_RUN, "closed_loop"))
    rebuild_s = pc() - t
    recs = [r for r in archive.records() if r.evaluated and r.metadata.get("fitness") is not None]
    last_gen = max(r.generation for r in recs)
    subject = max((r for r in recs if r.generation == last_gen), key=lambda r: float(r.metadata["fitness"]))
    cfg = FeedbackCardConfig()
    t = pc()
    card = fc.FeedbackCard(call_type="mutation", subjects=[subject], archive=archive, config=cfg)
    ctor_s = pc() - t
    ladder = fc.truncation_ladder(cfg)
    t = pc()
    rendered = card.render(ladder[0])
    render_s = pc() - t
    tk = make_token_counter("groq")
    heur = TokenCounter(method=f"heuristic x{HEURISTIC_MARGIN}", margin=HEURISTIC_MARGIN)
    msgs = [D1_SYSTEM_PROMPT, rendered.text]
    t = pc()
    n_tk = tk.count_request(msgs)
    tk_s = pc() - t
    t = pc()
    n_h = heur.count_request(msgs)
    h_s = pc() - t

    def render_level(level: int):
        if level >= len(ladder):
            return None
        r = card.render(ladder[level])
        return [D1_SYSTEM_PROMPT, r.text], ladder[level].name, {}

    t = pc()
    fit = fit_to_budget(render_level, tk, max_input_tokens=cfg.max_input_tokens,
                        target_input_tokens=cfg.target_input_tokens)
    fit_s = pc() - t
    return {
        "archive_records": len(archive.records()),
        "archive_rebuild_s": round(rebuild_s, 4),
        "card_ctor_s": round(ctor_s, 5),
        "card_render_full_s": round(render_s, 5),
        "card_chars": len(rendered.text),
        "tokens_tiktoken": n_tk,
        "tokens_heuristic": n_h,
        "count_tiktoken_s": round(tk_s, 5),
        "count_heuristic_s": round(h_s, 5),
        "token_method": tk.method,
        "fit_to_budget_s": round(fit_s, 5),
        "fit_level": fit.level,
        "fit_tokens": fit.tokens,
    }


def k_tiktoken_cold(p: Dict[str, Any], st: Dict[str, Any], rep: int) -> Dict[str, Any]:
    code = (
        "import time,json;t0=time.perf_counter();from raise_core.token_budget import make_token_counter;"
        "t1=time.perf_counter();c=make_token_counter('groq');c.count('hello world');t2=time.perf_counter();"
        "print(json.dumps({'import':t1-t0,'first_encoding_load_and_count':t2-t1,'method':c.method}))"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=_ROOT)
    d = json.loads(res.stdout.strip().splitlines()[-1])
    return {"token_budget_import_s": round(d["import"], 4),
            "tiktoken_first_load_s": round(d["first_encoding_load_and_count"], 4),
            "method": d["method"]}


def k_ckpt_io(p: Dict[str, Any], st: Dict[str, Any], rep: int) -> Dict[str, Any]:
    """Closed-loop checkpoint / epoch JSON writes with a real 8-generation payload."""
    import raise_core.json_io as jio
    from raise_core.feedback import CandidateArchive
    from raise_core.raise_loop.checkpoint import save_checkpoint
    from raise_core.raise_loop.logging_io import append_epoch_record, write_json

    with open(os.path.join(REF_RUN, "closed_loop", "checkpoint.json"), encoding="utf-8") as fh:
        payload = json.load(fh)
    with open(os.path.join(REF_RUN, "closed_loop", "pop_epoch_000.json"), encoding="utf-8") as fh:
        pop_payload = json.load(fh)
    with open(os.path.join(REF_RUN, "closed_loop", "epochs.jsonl"), encoding="utf-8") as fh:
        epoch_rows = [json.loads(x) for x in fh if x.strip()]
    root = os.path.join(WORK_DIR, "p4_io")
    shutil.rmtree(root, ignore_errors=True)
    dirs = [os.path.join(root, "closed_loop"), root]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    retries = {"n": 0}
    real_replace = jio.os.replace

    def counting_replace(a, b):
        try:
            return real_replace(a, b)
        except PermissionError:
            retries["n"] += 1
            raise

    jio.os.replace = counting_replace
    k = int(p.get("saves", 20))
    pc = time.perf_counter
    try:
        t = pc()
        for _ in range(k):
            for d in dirs:
                save_checkpoint(d, payload)
        save2_s = (pc() - t) / k
        t = pc()
        for _ in range(k):
            text = json.dumps(jio.sanitize_nonfinite(payload), indent=2, ensure_ascii=False, allow_nan=False)
        dumps_s = (pc() - t) / k
        tmp = os.path.join(root, "nofsync.json")
        t = pc()
        for _ in range(k):
            for _d in dirs:
                with open(tmp + ".tmp", "w", encoding="utf-8") as fh:
                    fh.write(text + "\n")
                real_replace(tmp + ".tmp", tmp)
        nofsync2_s = (pc() - t) / k
        t = pc()
        for _ in range(k):
            write_json(os.path.join(root, "closed_loop", "pop_epoch_000.json"), pop_payload)
        pop_s = (pc() - t) / k
        t = pc()
        for i, row in enumerate(epoch_rows[:4]):
            append_epoch_record(root, row)
        epochs_s = (pc() - t) / max(1, min(4, len(epoch_rows)))
        t = pc()
        CandidateArchive.from_epoch_files(os.path.join(REF_RUN, "closed_loop"))
        archive_s = pc() - t
    finally:
        jio.os.replace = real_replace
    shutil.rmtree(root, ignore_errors=True)
    return {
        "checkpoint_bytes": len(text.encode("utf-8")),
        "pop_epoch_bytes": len(json.dumps(pop_payload, indent=2, ensure_ascii=False, default=str).encode()),
        "save_ckpt_two_dirs_s": round(save2_s, 5),
        "json_dumps_only_s": round(dumps_s, 5),
        "write_two_dirs_no_fsync_s": round(nofsync2_s, 5),
        "fsync_and_resume_overhead_s": round(save2_s - dumps_s * 2 - nofsync2_s, 5),
        "pop_epoch_write_s": round(pop_s, 5),
        "append_epoch_record_s": round(epochs_s, 5),
        "archive_rebuild_8gen_s": round(archive_s, 4),
        "os_replace_permission_retries": retries["n"],
        "saves_per_rep": k,
    }


def k_llm_log(p: Dict[str, Any], st: Dict[str, Any], rep: int) -> Dict[str, Any]:
    """Latency / token statistics from a real Groq run's llm_calls.jsonl."""
    path = os.path.join(REF_RUN, "closed_loop", "llm_calls.jsonl")
    if not os.path.isfile(path):
        return {"available": False}
    req: Dict[str, Dict[str, Any]] = {}
    resp: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            if o.get("event") == "request":
                req[o["call_id"]] = o
            elif o.get("event") == "response":
                resp[o["call_id"]] = o
    with open(os.path.join(REF_RUN, "config.json"), encoding="utf-8") as fh:
        run_cfg = json.load(fh)
    lat = [float(r["latency_s"]) for r in resp.values() if r.get("latency_s") is not None]
    ptok = [int(r.get("prompt_tokens") or 0) for r in resp.values()]
    ctok = [int(r.get("completion_tokens") or 0) for r in resp.values()]
    by_type: Dict[str, int] = {}
    for r in req.values():
        by_type[r.get("call_type", "?")] = by_type.get(r.get("call_type", "?"), 0) + 1
    interval = float(os.environ.get("GROQ_MIN_REQUEST_INTERVAL", "3.0"))
    paced = [max(x, interval) for x in lat]
    gens = int(run_cfg.get("generations", 0))
    gen0_calls = by_type.get("initial_batch", 0) + by_type.get("initial", 0)
    breed_calls = len(req) - gen0_calls
    calls_per_breed_gen = breed_calls / max(1, gens - 1)
    breed_lat = [float(resp[c]["latency_s"]) for c, r in req.items()
                 if c in resp and r.get("call_type") not in ("initial_batch", "initial")]
    gen0_lat = [float(resp[c]["latency_s"]) for c, r in req.items()
                if c in resp and r.get("call_type") in ("initial_batch", "initial")]
    mean_paced_breed = statistics.fmean(max(x, interval) for x in breed_lat) if breed_lat else 0.0
    return {
        "available": True,
        "source": path.replace("\\", "/"),
        "provider": run_cfg.get("llm_provider"),
        "run_population": run_cfg.get("population"),
        "run_generations": gens,
        "n_requests": len(req),
        "n_responses": len(resp),
        "calls_by_type": by_type,
        "latency_median_s": round(statistics.median(lat), 3),
        "latency_mean_s": round(statistics.fmean(lat), 3),
        "latency_min_s": round(min(lat), 3),
        "latency_max_s": round(max(lat), 3),
        "prompt_tokens_median": statistics.median(ptok),
        "completion_tokens_median": statistics.median(ctok),
        "paced_total_s": round(sum(paced), 2),
        "groq_min_request_interval_s": interval,
        "gen0_paced_s": round(sum(max(x, interval) for x in gen0_lat), 3),
        "calls_per_breeding_generation": round(calls_per_breed_gen, 3),
        "mean_paced_breed_call_s": round(mean_paced_breed, 3),
        "est_llm_s_per_breeding_generation": round(calls_per_breed_gen * mean_paced_breed, 2),
    }


# ---------------------------------------------------------------- phase 5


_E2E_MARKERS = {
    "t_raise_start": "RAISE -> ",
    "t_closed_loop_start": "Closed-loop reward evolution",
    "t_closed_loop_done": "Closed loop complete",
    "t_stage3_done": "Stage III complete",
    "t_finished": "RAISE - finished",
    "t_post_run_start": "plots + animation",
    "t_post_run_done": "plots=",
}
_TRAIN_DONE_RE = re.compile(r"\[(Stage II|Stage III)\] PPO train done (\S+) in (\S+) \(")
_EVAL_RE = re.compile(r"\[([^\]]+)\] PPO metrics (\S+) \|.*\| eval (\S+)\s*$")


def _parse_e2e(lines: List[Tuple[float, str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, needle in _E2E_MARKERS.items():
        for t, line in lines:
            if needle in line:
                out[key] = round(t, 3)
                break
    sums = {"s2_train_sum_s": 0.0, "s3_train_sum_s": 0.0, "s2_eval_sum_s": 0.0,
            "s2_reeval_sum_s": 0.0, "s3_eval_sum_s": 0.0, "s3_reeval_sum_s": 0.0}
    counts = {"n_s2_train": 0, "n_s3_train": 0, "n_s2_reeval": 0, "n_s3_reeval": 0}
    score_start: Dict[int, float] = {}
    label_start: Dict[int, float] = {}
    epoch = 0
    scoring_seen = False
    for t, line in lines:
        m = _TRAIN_DONE_RE.search(line)
        if m:
            d = _parse_dur(m.group(3)) or 0.0
            key = "s2" if m.group(1) == "Stage II" else "s3"
            sums[f"{key}_train_sum_s"] += d
            counts[f"n_{key}_train"] += 1
        m = _EVAL_RE.search(line)
        if m:
            d = _parse_dur(m.group(3)) or 0.0
            tag = m.group(1)
            if tag == "Stage II":
                sums["s2_eval_sum_s"] += d
            elif tag == "Stage II re-eval":
                sums["s2_reeval_sum_s"] += d
                counts["n_s2_reeval"] += 1
            elif tag == "Stage III":
                sums["s3_eval_sum_s"] += d
            elif tag == "Stage III re-eval":
                sums["s3_reeval_sum_s"] += d
                counts["n_s3_reeval"] += 1
        if "scoring candidate" in line and not scoring_seen:
            score_start[epoch] = t
            scoring_seen = True
        m2 = re.search(r"epoch (\d+): Stage II labeling", line)
        if m2:
            epoch = int(m2.group(1))
            label_start[epoch] = t
            scoring_seen = False
            epoch += 1
    out.update({k: round(v, 3) for k, v in sums.items()})
    out.update(counts)
    s1 = 0.0
    for e, ts in score_start.items():
        if e in label_start:
            s1 += label_start[e] - ts
    out["score1_phases_sum_s"] = round(s1, 3)
    g = out
    if "t_closed_loop_start" in g and "t_closed_loop_done" in g:
        g["closed_loop_s"] = round(g["t_closed_loop_done"] - g["t_closed_loop_start"], 3)
        g["closed_loop_other_s"] = round(
            g["closed_loop_s"] - sums["s2_train_sum_s"] - sums["s2_eval_sum_s"] - sums["s2_reeval_sum_s"], 3
        )
        if 0 in score_start:
            g["gen0_proposal_validation_s"] = round(score_start[0] - g["t_closed_loop_start"], 3)
    if "t_closed_loop_done" in g and "t_stage3_done" in g:
        g["stage3_s"] = round(g["t_stage3_done"] - g["t_closed_loop_done"], 3)
        g["stage3_other_s"] = round(
            g["stage3_s"] - sums["s3_train_sum_s"] - sums["s3_eval_sum_s"] - sums["s3_reeval_sum_s"], 3
        )
    if "t_raise_start" in g:
        g["startup_s"] = g["t_raise_start"]
    if "t_finished" in g and "t_post_run_done" in g:
        g["post_run_s"] = round(g["t_post_run_done"] - g["t_finished"], 3)
    return out


def k_e2e(p: Dict[str, Any], st: Dict[str, Any], rep: int) -> Dict[str, Any]:
    """scripts/run_raise.py end to end; console lines timestamped by this process."""
    out_dir = os.path.join(WORK_DIR, "p5", str(p["_row_id"]), f"rep{rep}")
    shutil.rmtree(out_dir, ignore_errors=True)
    cmd = [sys.executable, RUN_RAISE, *p["flags"], "--output-dir", out_dir]
    env = os.environ.copy()
    env.update({k: str(v) for k, v in (p.get("run_env") or {}).items()})
    lines: List[Tuple[float, str]] = []
    os.makedirs(os.path.join(LOG_DIR, "p5"), exist_ok=True)
    log_path = os.path.join(LOG_DIR, "p5", f"{p['_row_id']}_rep{rep}.log")
    with _Usage() as u:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                cwd=_ROOT, text=True, encoding="utf-8", errors="replace")
        assert proc.stdout is not None
        t0 = u.t0
        with open(log_path, "w", encoding="utf-8") as lf:
            for line in proc.stdout:
                t = time.perf_counter() - t0
                lines.append((t, line.rstrip("\n")))
                lf.write(f"{t:9.3f} {line}")
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"run_raise exited with {rc}; see {log_path}")
    out = {"wall_s": round(u.wall, 3), "exit_code": rc, "log": log_path.replace("\\", "/"),
           "cmd": " ".join(cmd[1:]), **_parse_e2e(lines)}
    out.update({k: v for k, v in u.as_dict().items() if k != "sys_util_per_core"})
    ck = os.path.join(out_dir, "closed_loop", "checkpoint.json")
    if os.path.isfile(ck):
        out["checkpoint_bytes"] = os.path.getsize(ck)
    try:
        with open(os.path.join(out_dir, "final_candidate.json"), encoding="utf-8") as fh:
            fin = json.load(fh)
        md = fin.get("metadata") or {}
        out["final_candidate_id"] = fin.get("candidate_id")
        out["final_fitness_selection"] = md.get("fitness")
        out["final_eval_fitness"] = ((md.get("final_eval") or {}).get("metrics") or {}).get("fitness")
    except Exception:  # noqa: BLE001
        pass
    return out


KINDS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any], int], Dict[str, Any]]] = {
    "calib": k_calib,
    "machine": k_machine,
    "env_throughput": k_env_throughput,
    "step_decomp": k_step_decomp,
    "reward_micro": k_reward_micro,
    "profile": k_profile,
    "cold_start": k_cold_start,
    "ppo_train": k_ppo_train,
    "eval": k_eval,
    "label": k_label,
    "gen0": k_gen0,
    "feedback": k_feedback,
    "tiktoken_cold": k_tiktoken_cold,
    "ckpt_io": k_ckpt_io,
    "llm_log": k_llm_log,
    "e2e": k_e2e,
}
_LIGHT_KINDS = {"calib", "llm_log", "e2e", "tiktoken_cold"}


def worker_main(spec: Dict[str, Any]) -> int:
    kind = spec["kind"]
    params = dict(spec["params"])
    params["_row_id"] = spec["row_id"]
    params["_warmup"] = int(spec["warmup"])
    threads_eff = None
    if kind not in _LIGHT_KINDS:
        threads_eff = _apply_threads(int(spec["threads"]))
        _setup_logging(bool(params.get("verbose", False)))
    st: Dict[str, Any] = {}
    n_total = int(spec["warmup"]) + int(spec["repeats"])
    for i in range(n_total):
        warm = i < int(spec["warmup"])
        rec = {
            "ts": _utc(),
            "run_uid": spec["run_uid"],
            "row_id": spec["row_id"],
            "phase": spec["phase"],
            "kind": kind,
            "cls": spec.get("cls"),
            "desc": spec.get("desc"),
            "rep": i,
            "warmup": warm,
            "quick": bool(spec["quick"]),
            "spec_hash": spec["spec_hash"],
            "params": spec["params"],
            "threads_requested": int(spec["threads"]),
            "torch_threads_effective": threads_eff,
            "env": {k: os.environ.get(k) for k in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAISE_FORCE_CUDA",
                "RAISE_PLAIN_PROGRESS", "RAISE_FORCE_TQDM", "PYTHONHASHSEED")},
            "cmd": spec["cmd"],
        }
        t0 = time.perf_counter()
        try:
            metrics = KINDS[kind](params, st, i)
            rec["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            metrics = {}
            rec["status"] = "error"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["traceback"] = traceback.format_exc()[-4000:]
        rec["rep_wall_s"] = round(time.perf_counter() - t0, 4)
        rec["metrics"] = metrics
        _append_raw(rec)
        brief = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        print(f"[{spec['row_id']}] rep={i}{' (warmup)' if warm else ''} status={rec['status']} "
              f"{json.dumps(dict(list(brief.items())[:8]))}", flush=True)
        if rec["status"] != "ok":
            print(rec.get("traceback", ""), flush=True)
            return 1
    return 0


# ===================================================================== rows


class Row:
    def __init__(self, row_id: str, phase: str, kind: str, params: Optional[Dict[str, Any]] = None, *,
                 threads: int = 1, env: Optional[Dict[str, str]] = None, cls: str = "-",
                 desc: str = "", repeats: Optional[int] = None, warmup: Optional[int] = None) -> None:
        self.row_id = row_id
        self.phase = phase
        self.kind = kind
        self.params = params or {}
        self.threads = threads
        self.env = env or {}
        self.cls = cls
        self.desc = desc
        self.repeats = repeats
        self.warmup = warmup


def build_rows(quick: bool, args: Any) -> List[Row]:
    phys = 8
    try:
        with open(os.path.join(BENCH_DIR, "machine.json"), encoding="utf-8") as fh:
            phys = int(json.load(fh).get("physical_cores") or 8)
    except Exception:  # noqa: BLE001
        pass
    s3k = 1000 if quick else 3000
    simk = 5000 if quick else 20000
    eps = 10 if quick else 30
    lbl_steps = 500 if quick else 2000
    rows: List[Row] = []
    rows.append(Row("info.machine", "info", "machine", repeats=1, warmup=0, desc="machine fingerprint"))
    for ph in ("0", "1", "2", "3", "4", "5"):
        rows.append(Row(f"p{ph}.calib", ph, "calib", repeats=1, warmup=0,
                        desc="fixed CPU workload + idle background load before the phase"))

    # ---- phase 0
    for which, cls, desc in (
        ("none", "-", "reward_fn=None (native upstream reward)"),
        ("seed", "-", "seed reward via SandboxedReward"),
        ("seed_raw", "-", "seed reward, raw compiled fn (no sandbox bookkeeping)"),
        ("heavy", "-", "heavier reward (human loop + look-ahead) via SandboxedReward"),
        ("heavy_raw", "-", "heavier reward, raw compiled fn"),
    ):
        rows.append(Row(f"p0.sim.{which}", "0", "env_throughput", {"reward": which, "steps": simk},
                        cls=cls, desc=desc))
    rows.append(Row("p0.decomp.seed", "0", "step_decomp", {"steps": simk},
                    desc="per-segment step cost (replica of step_upstream)"))
    rows.append(Row("p0.rewardcall", "0", "reward_micro", {"loops": 3 if quick else 10},
                    desc="per-call reward cost, sandboxed vs raw"))
    rows.append(Row("p0.profile.ppo3000", "0", "profile", {"steps": s3k, "eval_episodes": eps},
                    desc="cProfile of Stage II train_and_eval (seed, instrumentation on)"))
    rows.append(Row("p0.cold", "0", "cold_start", {}, desc="cold-start costs"))

    # ---- phase 1
    base = {"steps": s3k, "eval_episodes": eps, "instr": True}
    for t in (1, 2, 4, phys):
        rows.append(Row(f"p1.threads.t{t}", "1", "ppo_train", dict(base), threads=t, cls="A",
                        desc=f"torch threads={t} (OMP/MKL env + torch.set_num_threads)"))
    rows.append(Row("p1.threads.tdefault", "1", "ppo_train", dict(base), threads=0, cls="A",
                    desc="no thread cap (= --torch-threads 0)"))
    for ne in (2, 4, 8):
        rows.append(Row(f"p1.nenvs.{ne}", "1", "ppo_train", {**base, "n_envs": ne}, cls="B",
                        desc=f"n_envs={ne} (changes n_steps / batch_size)"))
    rows.append(Row("p1.cuda", "1", "ppo_train", {**base, "device": "cuda"}, env={"RAISE_FORCE_CUDA": "1"},
                    cls="A*", desc="device=cuda with RAISE_FORCE_CUDA=1"))
    rows.append(Row("p1.action.continuous", "1", "ppo_train", {**base, "action_mode": "continuous"}, cls="B",
                    desc="continuous Box(2) actions"))
    rows.append(Row("p1.kin.unicycle", "1", "ppo_train", {**base, "kinematics": "unicycle"}, cls="B",
                    desc="unicycle kinematics (discrete)"))
    rows.append(Row("p1.action.continuous_unicycle", "1", "ppo_train",
                    {**base, "action_mode": "continuous", "kinematics": "unicycle"}, cls="B",
                    desc="continuous + unicycle"))
    rows.append(Row("p1.instr.off", "1", "ppo_train", {**base, "instr": False}, cls="A",
                    desc="record_training_dynamics/record_reward_components off"))
    rows.append(Row("p1.progress.tqdm", "1", "ppo_train", dict(base),
                    env={"RAISE_FORCE_TQDM": "1", "RAISE_PLAIN_PROGRESS": ""}, cls="A",
                    desc="tqdm bar forced on (stderr redirected to a file)"))
    rows.append(Row("p1.log.verbose", "1", "ppo_train", {**base, "verbose": True}, cls="A",
                    desc="--verbose logging"))
    rows.append(Row("p1.patch.nocb", "1", "ppo_train", {**base, "instr": False, "patch_nocb": True}, cls="A",
                    desc="[bench patch] instrumentation off + no progress callback"))
    rows.append(Row("p1.patch.lean", "1", "ppo_train",
                    {**base, "instr": False, "patch_nocb": True, "patch_nomonitor": True}, cls="A",
                    desc="[bench patch] no callback, no Monitor wrapper"))
    if not quick:
        rows.append(Row("p1.k2.t1", "1", "ppo_train", {**base, "steps": 12000}, threads=1, cls="A",
                        desc="full K2=12000 run, threads=1 (default)"))
        rows.append(Row("p1.k2.t2", "1", "ppo_train", {**base, "steps": 12000}, threads=2, cls="A",
                        desc="full K2=12000 run, threads=2"))
        rows.append(Row("p1.k2.nenvs4", "1", "ppo_train", {**base, "steps": 12000, "n_envs": 4}, cls="B",
                        desc="full K2=12000 run, n_envs=4"))

    # ---- phase 2
    for mode, desc in (
        ("holdout_stoch", "holdout_only, stochastic (default)"),
        ("holdout_det", "holdout_only, deterministic (class B)"),
        ("both_stoch", "eval_mode=both (train + holdout), stochastic"),
        ("reevaluate", "PPO.load + holdout eval (carried elite)"),
        ("predict_micro", "model.predict batch-1 overhead"),
    ):
        rows.append(Row(f"p2.eval.{mode}", "2", "eval",
                        {"mode": mode, "eval_episodes": eps, "model_steps": 12000},
                        cls={"holdout_det": "B", "both_stoch": "C"}.get(mode, "-"), desc=desc))

    # ---- phase 3
    lbl = {"steps": lbl_steps, "eval_episodes": eps, "instr": True, "n_cand": 6}
    for w in (1, 2, 3, 6):
        t = max(1, phys // w)
        rows.append(Row(f"p3.thread.w{w}t{t}", "3", "label", {**lbl, "workers": w, "threads": t,
                        "executor": "thread"}, threads=t, cls="D",
                        desc=f"label_workers={w} threads, torch threads={t}"))
    for w in (1, 2, 3, 6):
        if max(1, phys // w) == 1:
            continue
        rows.append(Row(f"p3.thread.w{w}t1", "3", "label", {**lbl, "workers": w, "threads": 1,
                        "executor": "thread"}, threads=1, cls="D",
                        desc=f"label_workers={w} threads, torch threads=1"))
    for ex in ("proc_cold", "proc_warm"):
        for w in (2, 3, 6):
            t = max(1, phys // w)
            rows.append(Row(f"p3.{ex}.w{w}t{t}", "3", "label", {**lbl, "workers": w, "threads": t,
                            "executor": ex}, threads=t, cls="D(proto)",
                            desc=f"[prototype] process pool ({ex}), {w} workers, torch threads={t}"))
    rows.append(Row("p3.proc_warm.w3t1", "3", "label", {**lbl, "workers": 3, "threads": 1,
                    "executor": "proc_warm"}, threads=1, cls="D(proto)",
                    desc="[prototype] process pool (proc_warm), 3 workers, torch threads=1"))

    # ---- phase 4
    rows.append(Row("p4.gen0.light", "4", "gen0", {"population": 6, "main": "light"},
                    desc="Gen0 (N=6) validation + Score1, light __main__"))
    rows.append(Row("p4.gen0.run_raise", "4", "gen0", {"population": 6, "main": "run_raise"},
                    desc="Gen0 (N=6) validation + Score1, spawn children re-import run_raise.py"))
    rows.append(Row("p4.feedback", "4", "feedback", {}, desc="feedback card render + token counting"))
    rows.append(Row("p4.tiktoken_cold", "4", "tiktoken_cold", {}, desc="fresh-process tiktoken load"))
    rows.append(Row("p4.ckpt_io", "4", "ckpt_io", {"saves": 5 if quick else 20},
                    desc="checkpoint / epoch JSON writes (real 300 KB payload)"))
    rows.append(Row("p4.llm_log", "4", "llm_log", {}, repeats=1, warmup=0,
                    desc="LLM latency/tokens from a real Groq run"))

    # ---- phase 5
    flags = P5_QUICK if quick else P5_COMMON
    best_flags = list(P5_BEST_FLAGS)
    best_env = dict(P5_BEST_ENV)
    if getattr(args, "p5_best_flags", None):
        best_flags = args.p5_best_flags.split()
    if getattr(args, "p5_best_env", None):
        best_env = dict(kv.split("=", 1) for kv in args.p5_best_env.split(",") if kv)
    rows.append(Row("p5.default", "5", "e2e", {"flags": list(flags), "run_env": {}}, threads=0,
                    desc="run_raise.py default flags (reduced budget)"))
    rows.append(Row("p5.best", "5", "e2e", {"flags": list(flags) + best_flags, "run_env": best_env},
                    threads=0, cls="A/D", desc="run_raise.py best class-A/D combination"))
    rows.append(Row("p5.classB_nenvs4", "5", "e2e",
                    {"flags": list(flags) + P5_CLASSB_FLAGS, "run_env": {}},
                    threads=0, cls="B", desc="run_raise.py --n-envs 4 (class B, changes training)"))
    return rows


def _spec_hash(row: Row, repeats: int, warmup: int, quick: bool) -> str:
    blob = json.dumps({"kind": row.kind, "params": row.params, "threads": row.threads, "env": row.env,
                       "repeats": repeats, "warmup": warmup, "quick": quick}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _row_cmd(row: Row, quick: bool) -> str:
    return f"python scripts/bench_speed.py --phase {row.phase} --only {row.row_id}" + (" --quick" if quick else "")


def _complete_rows(quick: bool) -> Dict[str, str]:
    """row_id -> spec_hash of rows with a complete, error-free set of repeats."""
    by_run: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in _read_raw():
        if bool(r.get("quick")) != quick:
            continue
        by_run.setdefault((r["row_id"], r["run_uid"]), []).append(r)
    done: Dict[str, str] = {}
    for (row_id, _uid), recs in by_run.items():
        if any(x.get("status") != "ok" for x in recs):
            continue
        reps = [x for x in recs if not x.get("warmup")]
        if not reps:
            continue
        expected = int((reps[0].get("params") or {}).get("_repeats", 0) or 0)
        n_rep = len(reps)
        if expected and n_rep < expected:
            continue
        done[row_id] = reps[0].get("spec_hash", "")
    return done


def run_rows(rows: List[Row], args: Any) -> None:
    quick = bool(args.quick)
    default_rep = 1 if quick else int(args.repeats or 3)
    default_warm = 0 if quick else 1
    done = _complete_rows(quick) if args.resume else {}
    os.makedirs(LOG_DIR, exist_ok=True)
    for row in rows:
        repeats = row.repeats if row.repeats is not None else default_rep
        warmup = row.warmup if row.warmup is not None else default_warm
        sh = _spec_hash(row, repeats, warmup, quick)
        if args.resume and done.get(row.row_id) == sh:
            print(f"[skip] {row.row_id} (complete in raw.jsonl)", flush=True)
            continue
        if row.kind == "ppo_train" and row.params.get("device") == "cuda":
            try:
                with open(os.path.join(BENCH_DIR, "machine.json"), encoding="utf-8") as fh:
                    if not json.load(fh).get("cuda_available"):
                        print(f"[skip] {row.row_id} (no GPU)", flush=True)
                        continue
            except Exception:  # noqa: BLE001
                pass
        spec = {
            "row_id": row.row_id, "phase": row.phase, "kind": row.kind,
            "params": {**row.params, "_repeats": repeats}, "threads": row.threads,
            "repeats": repeats, "warmup": warmup, "quick": quick, "run_uid": uuid.uuid4().hex[:12],
            "spec_hash": sh, "cls": row.cls, "desc": row.desc, "cmd": _row_cmd(row, quick),
        }
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0"
        env["RAISE_PLAIN_PROGRESS"] = "1"
        env.pop("RAISE_FORCE_TQDM", None)
        env.pop("RAISE_FORCE_CUDA", None)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            env.pop(var, None)
            if row.threads > 0:
                env[var] = str(row.threads)
        for k, v in row.env.items():
            if v == "":
                env.pop(k, None)
            else:
                env[k] = v
        if row.kind == "e2e":
            env.pop("RAISE_PLAIN_PROGRESS", None)
        payload = base64.urlsafe_b64encode(json.dumps(spec).encode()).decode()
        log_path = os.path.join(LOG_DIR, f"{row.row_id}.log")
        t0 = time.perf_counter()
        print(f"[run ] {row.row_id}: {row.desc} (warmup={warmup}, repeats={repeats})", flush=True)
        with open(log_path, "w", encoding="utf-8") as lf:
            rc = subprocess.call([sys.executable, os.path.abspath(__file__), "--worker", payload],
                                 env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=_ROOT)
        dt = time.perf_counter() - t0
        print(f"[done] {row.row_id}: exit={rc} in {dt:.1f}s -> {_brief(row.row_id, spec['run_uid'])}", flush=True)


_HEADLINE = ("steps_per_s_actual", "steps_per_s", "wall_s", "us_per_step", "total_us_per_step",
             "profiled_wall_s", "gen0_init_s", "save_ckpt_two_dirs_s", "card_render_full_s",
             "validate_seed_run_raise_main_s", "predict_us_per_call_batch1", "py_loop_s",
             "latency_median_s", "tiktoken_first_load_s", "seed_sandboxed_us_per_call")


def _brief(row_id: str, run_uid: str) -> str:
    recs = [r for r in _read_raw() if r.get("run_uid") == run_uid and not r.get("warmup")]
    if not recs:
        return "no records"
    if any(r.get("status") != "ok" for r in recs):
        return "ERROR: " + str(next(r.get("error") for r in recs if r.get("status") != "ok"))
    for key in _HEADLINE:
        vals = [r["metrics"].get(key) for r in recs if isinstance(r["metrics"].get(key), (int, float))]
        if vals:
            return f"{key} median={statistics.median(vals):.4g} (n={len(vals)})"
    return f"{len(recs)} ok"


# =================================================================== report


def _latest_records(quick: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """row_id -> non-warmup ok records of the latest complete run of that row."""
    by_run: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    order: Dict[Tuple[str, str], str] = {}
    for r in _read_raw():
        if bool(r.get("quick")) != quick:
            continue
        key = (r["row_id"], r["run_uid"])
        by_run.setdefault(key, []).append(r)
        order[key] = r["ts"]
    best: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {}
    for (row_id, uid), recs in by_run.items():
        if any(x.get("status") != "ok" for x in recs):
            continue
        reps = [x for x in recs if not x.get("warmup")]
        if not reps:
            continue
        expected = int((reps[0].get("params") or {}).get("_repeats", 0) or 0)
        if expected and len(reps) < expected:
            continue
        ts = order[(row_id, uid)]
        if row_id not in best or ts > best[row_id][0]:
            best[row_id] = (ts, reps)
    return {k: v[1] for k, v in best.items()}


def _stats(vals: Sequence[float]) -> Dict[str, float]:
    return {"median": statistics.median(vals), "min": min(vals), "max": max(vals), "n": len(vals)}


def _num_metrics(recs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    keys: List[str] = []
    for r in recs:
        for k, v in r["metrics"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in keys:
                keys.append(k)
    out = {}
    for k in keys:
        vals = [float(r["metrics"][k]) for r in recs
                if isinstance(r["metrics"].get(k), (int, float)) and not isinstance(r["metrics"].get(k), bool)]
        if vals:
            out[k] = _stats(vals)
    return out


def _m(latest: Dict[str, List[Dict[str, Any]]], row: str, key: str, stat: str = "median") -> Optional[float]:
    recs = latest.get(row)
    if not recs:
        return None
    s = _num_metrics(recs).get(key)
    return None if s is None else s[stat]


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.{nd}g}" if abs(v) < 1 else f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return str(v)


def _cell(latest: Dict[str, List[Dict[str, Any]]], row: str, key: str, nd: int = 3) -> str:
    recs = latest.get(row)
    if not recs:
        return "n/a"
    s = _num_metrics(recs).get(key)
    if s is None:
        return "n/a"
    if s["n"] == 1:
        return _fmt(s["median"], nd)
    return f"{_fmt(s['median'], nd)} [{_fmt(s['min'], nd)}–{_fmt(s['max'], nd)}]"


def _ident(latest: Dict[str, List[Dict[str, Any]]], row: str, base: str) -> str:
    """Equivalence of policy hash + fitness vs ``base`` (first repeat of each)."""
    a, b = latest.get(row), latest.get(base)
    if not a or not b:
        return "n/a"
    ha = {r["metrics"].get("policy_hash") for r in a}
    hb = {r["metrics"].get("policy_hash") for r in b}
    fa = {round(float(r["metrics"].get("fitness") or 0), 9) for r in a}
    fb = {round(float(r["metrics"].get("fitness") or 0), 9) for r in b}
    stable = "stable" if len(ha) == 1 else f"{len(ha)} distinct hashes across repeats"
    if ha == hb and len(ha) == 1:
        return f"identical (hash {next(iter(ha))}; {stable})"
    if fa == fb:
        return f"same fitness, different weights ({stable})"
    dfit = abs(statistics.median(fa) - statistics.median(fb))
    return f"differs (|Δfitness|={dfit:.3g}; {stable})"


def write_summary(quick: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    latest = _latest_records(quick)
    os.makedirs(BENCH_DIR, exist_ok=True)
    path = os.path.join(BENCH_DIR, "summary.csv" if not quick else "summary_quick.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "phase", "class", "metric", "median", "min", "max", "n", "run_uid", "cmd", "desc"])
        for row_id in sorted(latest):
            recs = latest[row_id]
            r0 = recs[0]
            for k, s in _num_metrics(recs).items():
                w.writerow([row_id, r0["phase"], r0.get("cls"), k, f"{s['median']:.6g}", f"{s['min']:.6g}",
                            f"{s['max']:.6g}", s["n"], r0["run_uid"], r0["cmd"], r0.get("desc")])
    return latest


def extrapolate(latest: Dict[str, List[Dict[str, Any]]], variant: str = "default",
                label_speedup: float = 1.0, train_rate_row: str = "p1.k2.t1") -> Dict[str, Any]:
    """Full default run (N=6, G=4, K2=12k, K3=70k, E=30, final 100) from measured unit costs."""
    N, G, K2, K3, E, E_final = 6, 4, 12000, 70000, 30, 100
    k2_train = _m(latest, train_rate_row, "train_wall_s")
    k2_eval = _m(latest, train_rate_row, "eval_wall_s")
    k3k_train = _m(latest, "p1.threads.t1", "train_wall_s")
    k2_steps = _m(latest, train_rate_row, "actual_train_steps") or K2
    if k2_train is None or k3k_train is None:
        return {"error": "missing p1 rows"}
    k3k_steps = _m(latest, "p1.threads.t1", "actual_train_steps") or 3072
    # Linear model train_wall = a + b * steps through the 3k and 12k medians.
    b = (k2_train - k3k_train) / max(1.0, k2_steps - k3k_steps)
    a = k2_train - b * k2_steps
    k3_train_linear = a + b * K3
    k3_train_rate = k2_train / k2_steps * K3
    eval_ep = (k2_eval or 0.0) / E
    reeval = (_m(latest, "p0.cold", "ppo_load_s") or 0.0) + eval_ep * E
    score1 = _m(latest, "p0.cold", "score1_warm_call_run_raise_main_s") or 0.0
    score1_start = (_m(latest, "p0.cold", "score1_first_call_run_raise_main_s") or 0.0) - score1
    validate = _m(latest, "p4.gen0.run_raise", "validate_median_s") or 0.0
    ckpt = _m(latest, "p4.ckpt_io", "save_ckpt_two_dirs_s") or 0.0
    pop_epoch = _m(latest, "p4.ckpt_io", "pop_epoch_write_s") or 0.0
    startup = _m(latest, "p5.default", "startup_s") or 0.0
    post_run = _m(latest, "p5.default", "post_run_s") or 0.0
    llm_breed = _m(latest, "p4.llm_log", "est_llm_s_per_breeding_generation") or 0.0
    llm_gen0 = _m(latest, "p4.llm_log", "gen0_paced_s") or 0.0

    s2_train = (N + (G - 1) * (N - 1)) * k2_train
    s2_eval = (N + (G - 1) * (N - 1)) * eval_ep * E
    s2_reeval = (G - 1) * reeval
    s2_par = (s2_train + s2_eval + s2_reeval) / max(label_speedup, 1e-9)
    n_valid = N + (G - 1) * (N - 1)
    validation = n_valid * validate
    score = G * N * score1 + score1_start
    ckpt_io = G * (N + 2) * ckpt + G * pop_epoch
    s3_train = N * k3_train_linear
    s3_eval = N * eval_ep * E
    final = eval_ep * E_final + (_m(latest, "p0.cold", "ppo_load_s") or 0.0)
    total_seed = startup + s2_par + validation + score + ckpt_io + s3_train + s3_eval + final + post_run
    llm_groq = llm_gen0 + (G - 1) * llm_breed
    return {
        "variant": variant,
        "inputs": {
            "N": N, "G": G, "K2": K2, "K3": K3, "E": E, "E_final": E_final,
            "k2_train_s": k2_train, "k2_eval_s(30 ep)": k2_eval, "eval_s_per_episode": eval_ep,
            "train_linear_a_s": a, "train_linear_b_s_per_step": b,
            "k3_train_linear_s": k3_train_linear, "k3_train_rate_s": k3_train_rate,
            "reeval_s": reeval, "score1_per_candidate_s": score1, "score1_worker_startup_s": score1_start,
            "validate_per_candidate_s": validate, "ckpt_save_s": ckpt, "pop_epoch_write_s": pop_epoch,
            "startup_s": startup, "post_run_s": post_run, "label_speedup": label_speedup,
            "llm_gen0_s": llm_gen0, "llm_per_breeding_gen_s": llm_breed,
        },
        "breakdown_s": {
            "stage2_env+update(train)": s2_train / max(label_speedup, 1e-9),
            "stage2_eval+reeval": (s2_eval + s2_reeval) / max(label_speedup, 1e-9),
            "validation(spawn smoke tests)": validation,
            "score1": score,
            "checkpoint_io": ckpt_io,
            "stage3_train": s3_train,
            "stage3_eval+final": s3_eval + final,
            "startup+post_run": startup + post_run,
        },
        "total_seed_llm_s": total_seed,
        "llm_groq_estimate_s": llm_groq,
        "total_groq_s": total_seed + llm_groq,
    }


def report_tables(quick: bool = False) -> None:
    latest = write_summary(quick)
    L: List[str] = []
    w = L.append
    w("# Auto-generated benchmark tables\n")
    w("Cells: median [min–max] over the timed repeats of the latest complete run of each row "
      "(warmup discarded). Rerun a row with the command in `summary.csv`.\n")
    w("\n## Phase 0 — simulator throughput\n")
    w("| row | steps/s | µs/step | reward_sum | n |\n|---|---|---|---|---|")
    for which in ("none", "seed", "seed_raw", "heavy", "heavy_raw"):
        r = f"p0.sim.{which}"
        n = len(latest.get(r, []))
        w(f"| `{r}` | {_cell(latest, r, 'steps_per_s')} | {_cell(latest, r, 'us_per_step')} | "
          f"{_cell(latest, r, 'reward_sum', 6)} | {n} |")
    w("\n### Step decomposition (`p0.decomp.seed`, µs per step)\n")
    w("| segment | µs/step |\n|---|---|")
    for seg in ("crowdsim_step", "min_separation", "observation", "state_construction", "reward_compute",
                "info_dict", "bookkeeping", "to_upstream", "reset", "total", "timer_overhead"):
        w(f"| {seg} | {_cell(latest, 'p0.decomp.seed', seg + '_us_per_step')} |")
    w("\n### Reward call cost (`p0.rewardcall`, µs per call)\n")
    w("| reward | sandboxed | raw | overhead | share of 20 ms budget |\n|---|---|---|---|---|")
    for which in ("seed", "heavy"):
        w(f"| {which} | {_cell(latest, 'p0.rewardcall', which + '_sandboxed_us_per_call')} | "
          f"{_cell(latest, 'p0.rewardcall', which + '_raw_us_per_call')} | "
          f"{_cell(latest, 'p0.rewardcall', which + '_sandbox_overhead_us')} | "
          f"{_cell(latest, 'p0.rewardcall', which + '_fraction_of_20ms_budget')} |")
    w("\n### Profile shares (`p0.profile.ppo3000`, share of `learn()` cumulative time)\n")
    w("| item | seconds | share of learn |\n|---|---|---|")
    for key, *_ in _PROFILE_KEYS:
        w(f"| {key} | {_cell(latest, 'p0.profile.ppo3000', key + '_s')} | "
          f"{_cell(latest, 'p0.profile.ppo3000', key + '_share_of_learn')} |")
    w("\n### Cold-start costs (`p0.cold`, seconds)\n")
    w("| item | seconds |\n|---|---|")
    for r in latest.get("p0.cold", [])[:1]:
        for k in r["metrics"]:
            if isinstance(r["metrics"][k], (int, float)):
                w(f"| {k} | {_cell(latest, 'p0.cold', k, 4)} |")

    w("\n## Phase 1 — PPO training throughput (3000 steps unless noted)\n")
    w("| row | class | torch thr | steps/s (actual) | train s | rollout s | update s | eval s (30 ep) "
      "| proc CPU cores | fitness | equivalence vs `p1.threads.t1` |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    p1 = sorted(k for k in latest if k.startswith("p1.") and k != "p1.calib")
    for r in p1:
        r0 = latest[r][0]
        w(f"| `{r}` | {r0.get('cls')} | {r0.get('torch_threads_effective')} | "
          f"{_cell(latest, r, 'steps_per_s_actual')} | {_cell(latest, r, 'train_wall_s')} | "
          f"{_cell(latest, r, 'rollout_s')} | {_cell(latest, r, 'update_s')} | {_cell(latest, r, 'eval_wall_s')} | "
          f"{_cell(latest, r, 'proc_cpu_cores')} | {_cell(latest, r, 'fitness')} | "
          f"{_ident(latest, r, 'p1.threads.t1') if not r.startswith('p1.k2') else _ident(latest, r, 'p1.k2.t1')} |")

    w("\n## Phase 2 — evaluation throughput\n")
    w("| row | class | wall s | episodes/s | env steps | steps/s | ms/episode | fitness |\n"
      "|---|---|---|---|---|---|---|---|")
    for mode in ("holdout_stoch", "holdout_det", "both_stoch", "reevaluate"):
        r = f"p2.eval.{mode}"
        cls = latest[r][0].get("cls") if r in latest else "?"
        w(f"| `{r}` | {cls} | {_cell(latest, r, 'wall_s')} | {_cell(latest, r, 'episodes_per_s')} | "
          f"{_cell(latest, r, 'env_steps')} | {_cell(latest, r, 'steps_per_s')} | "
          f"{_cell(latest, r, 'ms_per_episode')} | {_cell(latest, r, 'fitness')} |")
    r = "p2.eval.predict_micro"
    w("\n| predict metric (`p2.eval.predict_micro`) | µs |\n|---|---|")
    for k in ("predict_us_per_call_batch1", "policy_predict_nograd_us_batch1", "predict_us_per_obs_batch30",
              "predict_overhead_us", "env_step_us"):
        w(f"| {k} | {_cell(latest, r, k)} |")

    w("\n## Phase 3 — parallel labeling (6 candidates)\n")
    base_t8 = _m(latest, "p3.thread.w1t8", "wall_s")
    base_t1 = _m(latest, "p3.thread.w1t1", "wall_s")
    w("| row | class | wall s | speedup vs w1t8 | speedup vs w1t1 | proc CPU cores | sys busy cores "
      "| Σ cand train s | fitness vector = w1t1? | hashes = w1t1? |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    ref = latest.get("p3.thread.w1t1", [{}])[0].get("metrics", {}) if latest.get("p3.thread.w1t1") else {}
    for r in sorted(k for k in latest if k.startswith("p3.") and k != "p3.calib"):
        wall = _m(latest, r, "wall_s")
        recs = latest[r]
        fv_same = all(x["metrics"].get("fitness_vector") == ref.get("fitness_vector") for x in recs) if ref else None
        hv_same = all(x["metrics"].get("hash_vector") == ref.get("hash_vector") for x in recs) if ref else None
        cores = _cell(latest, r, "total_cpu_cores_incl_children") if "proc" in r else _cell(latest, r, "proc_cpu_cores")
        w(f"| `{r}` | {recs[0].get('cls')} | {_cell(latest, r, 'wall_s')} | "
          f"{_fmt(base_t8 / wall if base_t8 and wall else None)} | {_fmt(base_t1 / wall if base_t1 and wall else None)} | "
          f"{cores} | {_cell(latest, r, 'sys_busy_cores')} | {_cell(latest, r, 'sum_cand_train_s')} | "
          f"{fv_same} | {hv_same} |")

    w("\n## Phase 4 — non-PPO overhead\n")
    w("| row | metric | median [min–max] |\n|---|---|---|")
    for r in ("p4.gen0.light", "p4.gen0.run_raise", "p4.feedback", "p4.tiktoken_cold", "p4.ckpt_io", "p4.llm_log"):
        for k in _num_metrics(latest.get(r, [])):
            w(f"| `{r}` | {k} | {_cell(latest, r, k, 4)} |")

    w("\n## Phase 5 — end to end (reduced budget)\n")
    p5_rows = ("p5.default", "p5.best", "p5.classB_nenvs4")
    w("| metric | " + " | ".join(f"`{r}`" for r in p5_rows) + " |\n|---|---|---|---|")
    keys: List[str] = []
    for r in p5_rows:
        for k in _num_metrics(latest.get(r, [])):
            if k not in keys and not k.startswith("t_"):
                keys.append(k)
    for k in keys:
        w(f"| {k} | " + " | ".join(_cell(latest, r, k) for r in p5_rows) + " |")
    d = _m(latest, "p5.default", "wall_s")
    for r in p5_rows[1:]:
        b = _m(latest, r, "wall_s")
        if d and b:
            w(f"\nEnd-to-end speedup `{r}` vs `p5.default` (median wall): {d / b:.3f}x\n")

    extr = {"default": extrapolate(latest, "default")}
    best_w = None
    for cand in ("p3.thread.w2t1", "p3.thread.w3t1", "p3.thread.w6t1", "p3.thread.w2t4", "p3.thread.w3t2"):
        v = _m(latest, cand, "wall_s")
        if v and base_t1 and (best_w is None or v < best_w[1]):
            best_w = (cand, v)
    if best_w and base_t1:
        extr["best_threads_label"] = extrapolate(latest, f"label via {best_w[0]}", label_speedup=base_t1 / best_w[1])
    for ex in ("p3.proc_warm.w6t1", "p3.proc_warm.w3t1"):
        v = _m(latest, ex, "wall_s")
        if v and base_t1:
            extr[f"proposal_{ex}"] = extrapolate(latest, f"PROPOSAL process pool {ex}", label_speedup=base_t1 / v)
    with open(os.path.join(BENCH_DIR, "extrapolation.json"), "w", encoding="utf-8") as fh:
        json.dump(extr, fh, indent=2)
    w("\n## Full-run extrapolation (N=6, G=4, K2=12k, K3=70k, E=30, final 100 ep)\n")
    for name, e in extr.items():
        if "error" in e:
            w(f"- {name}: {e['error']}")
            continue
        w(f"\n### {name}: {e['variant']}\n")
        w("| component | seconds | share |\n|---|---|---|")
        tot = e["total_seed_llm_s"]
        for k, v in e["breakdown_s"].items():
            w(f"| {k} | {v:.1f} | {v / tot:.1%} |")
        w(f"| **total (seed LLM)** | **{tot:.1f}** ({tot / 60:.1f} min) | 100% |")
        w(f"| LLM (Groq estimate, added) | {e['llm_groq_estimate_s']:.1f} | — |")
        w(f"| **total with Groq** | **{e['total_groq_s']:.1f}** ({e['total_groq_s'] / 60:.1f} min) | — |")
    with open(os.path.join(BENCH_DIR, "tables.md" if not quick else "tables_quick.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {BENCH_DIR}/summary.csv, tables.md, extrapolation.json", flush=True)


# ===================================================================== main


_PHASE_ORDER = ["info", "0", "1", "2", "3", "4", "5"]


def main() -> int:
    ap = argparse.ArgumentParser(description="RAISE speed benchmark (see module docstring)")
    ap.add_argument("--phase", nargs="+", default=["all"],
                    choices=["info", "0", "1", "2", "3", "4", "5", "report", "all"])
    ap.add_argument("--quick", action="store_true", help="tiny budgets, 1 repeat, no warmup")
    ap.add_argument("--resume", action="store_true", help="skip rows complete in raw.jsonl")
    ap.add_argument("--only", default=None, help="comma-separated row_id glob patterns")
    ap.add_argument("--repeats", type=int, default=None, help="timed repeats per row (default 3)")
    ap.add_argument("--list", action="store_true", help="print rows and exit")
    ap.add_argument("--p5-best-flags", default=None, help="override run_raise flags of p5.best")
    ap.add_argument("--p5-best-env", default=None, help="override env of p5.best (K=V,K2=V2)")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        spec = json.loads(base64.urlsafe_b64decode(args.worker.encode()).decode())
        return worker_main(spec)
    phases = _PHASE_ORDER + ["report"] if "all" in args.phase else args.phase
    rows = build_rows(bool(args.quick), args)
    if args.list:
        for r in rows:
            print(f"{r.row_id:34s} phase={r.phase} kind={r.kind:14s} cls={r.cls:9s} thr={r.threads} {r.desc}")
        return 0
    patterns = [x.strip() for x in args.only.split(",")] if args.only else None
    for ph in phases:
        if ph == "report":
            continue
        sel = [r for r in rows if r.phase == ph]
        if patterns:
            sel = [r for r in sel if any(fnmatch.fnmatch(r.row_id, pat) for pat in patterns)]
        run_rows(sel, args)
    if "report" in phases:
        report_tables(bool(args.quick))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
