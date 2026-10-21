#!/usr/bin/env python
"""Probe Groq context limits for the configured RAISE LLM model.

Offline: estimates ranking-prompt size for N candidates under current card budget.
Live (needs GROQ_API_KEY or raise_env/groq_keys.json): binary-searches the largest
user prompt the API accepts for a tiny completion.

Working directory: raise_env/
  python scripts/dev/probe_groq_context.py
  python scripts/dev/probe_groq_context.py --live
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import raise_paths  # noqa: E402,F401
from raise_core.script_utils import bootstrap_raise_env  # noqa: E402


def _counter():
    from raise_core.token_budget import make_token_counter

    return make_token_counter("groq")


def _pad_prompt(n_tokens: int) -> str:
    """Build a prompt whose counted size is about ``n_tokens`` (plus short ask)."""
    counter = _counter()
    ask = "\nReply with exactly: OK"
    # Rough fill: ~1 token per 4 chars for letter runs under heuristic / o200k.
    unit = "abcd "
    fill = ""
    # Grow until count(fill+ask) >= n_tokens
    lo, hi = 0, max(100, n_tokens * 6)
    best = unit
    while lo < hi:
        mid = (lo + hi) // 2
        cand = unit * mid
        if counter.count(cand + ask) < n_tokens:
            lo = mid + 1
            best = cand
        else:
            hi = mid
            best = cand
    # Fine-tune
    while counter.count(best + ask) < n_tokens:
        best += unit
    while best and counter.count(best + ask) > n_tokens:
        best = best[: max(0, len(best) - len(unit))]
    return best + ask


def probe_live(model: str, *, low: int, high: int, max_tokens: int) -> dict:
    from raise_core.llm import GroqLLMClient

    client = GroqLLMClient(model=model, temperature=0.0, max_tokens=max_tokens)
    counter = _counter()
    accepted = 0
    rejected = None
    last_err = None
    lo, hi = low, high
    while lo <= hi:
        mid = (lo + hi) // 2
        prompt = _pad_prompt(mid)
        counted = counter.count_request(
            [str(client.system_prompt or ""), prompt]
        )
        try:
            client.complete(prompt, max_tokens=max_tokens)
            accepted = mid
            print(f"  OK  target~={mid} counted_request~={counted}")
            lo = mid + 1
        except Exception as exc:  # noqa: BLE001
            # Size accepted if the API ran but truncated the tiny completion
            # (reasoning models often burn max_tokens before emitting "OK").
            name = type(exc).__name__
            if name == "CompletionTruncatedError":
                accepted = mid
                print(f"  OK(trunc) target~={mid} counted_request~={counted}")
                lo = mid + 1
                continue
            rejected = mid
            last_err = f"{name}: {exc}"
            print(f"  FAIL target~={mid} counted_request~={counted} -> {last_err}")
            hi = mid - 1
    return {
        "model": model,
        "largest_accepted_user_target_tokens": accepted,
        "first_rejected_user_target_tokens": rejected,
        "last_error": last_err,
        "token_method": counter.method,
        "max_output_tokens_used": max_tokens,
    }


def probe_offline_ranking(*, population: int, max_input: int, target: int) -> dict:
    from raise_config import CONFIG
    from raise_core.evo_selection.llm_ranker import ranking_prompt, anonymous_labels
    from raise_core.token_budget import make_token_counter

    counter = make_token_counter("groq")
    # Compact synthetic card (~metrics + short code) vs fat card.
    compact = (
        "[A] mut_0001\n"
        "SR 0.80 | CR 0.10 | TR 0.10 | nav_time 12.00 s | path_length 9.00 m | "
        "danger_freq 0.050 | min_dist_in_danger 0.120 m\n"
        "(secondary) scalar fitness 0.650\n"
        "Behaviour: ok\n"
        "[D] vs parent: SR +0.05 CR -0.02\n"
    )
    fat = compact + ("history line with code summary " * 40 + "\n") * 8
    fat += ("episode representative " * 30 + "\n") * 3
    system = "You are a multi-objective candidate evaluator. Return strict JSON only."
    rows = []
    for label, body in (("compact", compact), ("fat", fat)):
        ids = [f"c{i}" for i in range(population)]
        labels = anonymous_labels(ids)
        cards = [(labels[i], body) for i in ids]
        prompt = ranking_prompt(cards)
        toks = counter.count_request([system, prompt])
        rows.append(
            {
                "card_style": label,
                "population": population,
                "request_tokens": toks,
                "fits_target": toks <= target,
                "fits_max": toks <= max_input,
            }
        )
    return {
        "token_method": counter.method,
        "config_max_input_tokens": max_input,
        "config_target_input_tokens": target,
        "yaml_model": CONFIG.llm.model,
        "rows": rows,
        "docs_context_window_gpt_oss_120b": 131_072,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live", action="store_true", help="Binary-search live Groq accept limit")
    p.add_argument("--model", default=None, help="Override model id")
    p.add_argument("--low", type=int, default=2000)
    p.add_argument("--high", type=int, default=20000, help="Live search upper bound (user tokens)")
    p.add_argument("--max-tokens", type=int, default=16, help="Tiny completion budget for live probe")
    p.add_argument("--population", type=int, default=8)
    return p


def main(argv: list[str] | None = None) -> int:
    bootstrap_raise_env()
    args = build_parser().parse_args(argv)
    from raise_config import CONFIG

    max_in = int(CONFIG.selection.max_input_tokens)
    target = int(CONFIG.selection.target_input_tokens)
    model = args.model or str(CONFIG.llm.model or "openai/gpt-oss-120b")

    print("=== Offline ranking size estimate ===")
    off = probe_offline_ranking(
        population=args.population, max_input=max_in, target=target
    )
    print(f"token_method={off['token_method']}")
    print(f"YAML budget max={max_in} target={target}")
    print(f"Groq docs context (gpt-oss-120b)={off['docs_context_window_gpt_oss_120b']}")
    for row in off["rows"]:
        print(
            f"  {row['card_style']}: N={row['population']} -> "
            f"{row['request_tokens']} tokens "
            f"(target_ok={row['fits_target']} max_ok={row['fits_max']})"
        )

    if not args.live:
        print(
            "\nNo --live flag: skipped API probe. "
            "Set GROQ_API_KEY or groq_keys.json then rerun with --live."
        )
        return 0

    print("\n=== Live Groq binary search ===")
    try:
        result = probe_live(
            model, low=args.low, high=args.high, max_tokens=args.max_tokens
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Live probe failed to start: {type(exc).__name__}: {exc}")
        return 2
    print("RESULT:", result)
    safe = int(0.85 * int(result["largest_accepted_user_target_tokens"] or 0))
    print(
        f"Suggested RAISE selection.max_input_tokens ~= {safe} "
        f"(85% of largest accepted ~={result['largest_accepted_user_target_tokens']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
