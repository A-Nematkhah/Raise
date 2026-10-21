"""Unicode punctuation cleanup must not rewrite string literals."""

from __future__ import annotations

from domains.crowdnav.reward_checks import normalize_unicode_punctuation


def test_unicode_dashes_outside_strings_become_ascii():
    code = "def compute_reward(state, memory):\n    x = 0.2\u202f + 1\u2011\n    return x\n"
    out = normalize_unicode_punctuation(code)
    assert "\u202f" not in out
    assert "\u2011" not in out
    assert "0.2 " in out or "0.2+" in out.replace(" ", "")


def test_unicode_inside_string_literal_is_preserved():
    code = (
        "def compute_reward(state, memory):\n"
        "    note = 'gap\u202fis\u20110.2'\n"
        "    return 0.0\n"
    )
    out = normalize_unicode_punctuation(code)
    assert "gap\u202fis\u20110.2" in out
