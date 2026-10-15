"""Run provenance snapshot (git, packages, argv) — never fails the run."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from raise_core.json_io import atomic_write_json, read_json_dict, utc_now_z


def _safe_version(mod_name: str) -> Optional[str]:
    try:
        mod = __import__(mod_name)
    except Exception:  # noqa: BLE001
        return None
    return getattr(mod, "__version__", None)


def _git_info(repo_root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"head": None, "dirty": None, "modified": []}
    try:
        head = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        out["head"] = head or None
        porcelain = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root}", "status", "--porcelain"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        lines = [ln for ln in porcelain.splitlines() if ln.strip()]
        out["dirty"] = bool(lines)
        out["modified"] = [ln[3:].strip() if len(ln) > 3 else ln for ln in lines[:50]]
    except Exception:  # noqa: BLE001
        pass
    return out


def collect_provenance(
    *,
    output_dir: str,
    resume_index: int = 0,
    torch_threads: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a provenance dict. Never raises."""
    here = Path(__file__).resolve()
    # raise_env/raise_core/provenance.py -> repo root is parents[2]
    repo_root = here.parents[2]
    try:
        import torch

        torch_ver = getattr(torch, "__version__", None)
        threads = int(torch.get_num_threads())
    except Exception:  # noqa: BLE001
        torch_ver, threads = None, None
    if torch_threads is not None:
        threads = int(torch_threads)
    return {
        "utc": utc_now_z(),
        "argv": list(sys.argv),
        "resume_index": int(resume_index),
        "output_dir": str(output_dir),
        "python": sys.version,
        "platform": platform.platform(),
        "git": _git_info(repo_root),
        "versions": {
            "torch": torch_ver,
            "stable_baselines3": _safe_version("stable_baselines3"),
            "gymnasium": _safe_version("gymnasium"),
            "gym": _safe_version("gym"),
            "numpy": _safe_version("numpy"),
            "groq": _safe_version("groq"),
            "tiktoken": _safe_version("tiktoken"),
            "rvo2": _safe_version("rvo2"),
        },
        "torch_num_threads": threads,
        "env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
    }


def write_provenance(output_dir: str, *, torch_threads: Optional[int] = None) -> str:
    """Append/write ``{output_dir}/provenance.json``. Returns the path."""
    path = os.path.join(output_dir, "provenance.json")
    prior = read_json_dict(path)
    resumes: List[Dict[str, Any]] = []
    if isinstance(prior, dict):
        prior_resumes = prior.get("resumes")
        if isinstance(prior_resumes, list):
            resumes = [dict(r) for r in prior_resumes if isinstance(r, dict)]
        elif prior.get("utc"):
            resumes = [{"utc": prior.get("utc"), "argv": prior.get("argv")}]
    entry = collect_provenance(
        output_dir=output_dir,
        resume_index=len(resumes),
        torch_threads=torch_threads,
    )
    resumes.append({"utc": entry["utc"], "argv": entry["argv"]})
    payload = {**entry, "resumes": resumes}
    if isinstance(prior, dict) and prior.get("utc") and not prior.get("resumes"):
        # Keep the first-process stamp as created_utc.
        payload["created_utc"] = prior.get("utc")
    elif isinstance(prior, dict) and prior.get("created_utc"):
        payload["created_utc"] = prior["created_utc"]
    else:
        payload["created_utc"] = entry["utc"]
    atomic_write_json(path, payload)
    return path


__all__ = ["collect_provenance", "write_provenance"]
