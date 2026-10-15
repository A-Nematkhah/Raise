"""Mutation prompt must not duplicate the ``def`` line (P0-4)."""

from __future__ import annotations

import re

from domains.crowdnav.prompts import format_d2_mutation

_PARENT = """def compute_reward(state, memory):
    return 0.0, {}
"""


def test_mutation_prompt_has_single_def_in_parent_section():
    text = format_d2_mutation(_PARENT, "reflection note")
    section = text.split("Parent Code to Mutate:", 1)[1].split("Mutation Task:", 1)[0]
    defs = re.findall(r"^\s*-?\s*def compute_reward\s*\(", section, flags=re.M)
    assert len(defs) == 1, section


def test_mutation_prompt_prepends_signature_when_body_lacks_def():
    text = format_d2_mutation(
        "    return 1.0, {}\n",
        "note",
        func_signature="def compute_reward(state, memory):",
    )
    section = text.split("Parent Code to Mutate:", 1)[1].split("Mutation Task:", 1)[0]
    assert "def compute_reward(state, memory):" in section
    assert "return 1.0" in section
