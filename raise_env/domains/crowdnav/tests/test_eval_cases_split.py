"""P1-4: eval_cases cycling and Stage II eval_split wiring."""

from __future__ import annotations

import logging

from domains.crowdnav.adapter import EVAL_SPLITS, _eval_split
from domains.crowdnav.env_wrapper import eval_cases
from raise_core.raise_loop.config import ClosedLoopConfig
from raise_core.raise_loop.runner import _RESUME_CONFIG_KEYS


def test_eval_cases_distinct_when_n_le_size():
    cases = eval_cases(425, 50, 100)
    assert len(cases) == 50
    assert len(set(cases)) == 50
    assert all(0 <= c < 100 for c in cases)


def test_eval_cases_cycles_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        cases = eval_cases(425, 150, 100)
    assert len(cases) == 150
    assert len(set(cases)) == 100
    assert any("cycle" in r.message for r in caplog.records)


def test_eval_split_accepts_train_val_test():
    assert EVAL_SPLITS == ("val", "test", "train")
    assert _eval_split(type("C", (), {"eval_split": "train"})()) == "train"
    assert _eval_split(type("C", (), {"eval_split": "val"})()) == "val"


def test_stage2_eval_split_in_resume_keys_and_defaults():
    assert "stage2_eval_split" in _RESUME_CONFIG_KEYS
    assert ClosedLoopConfig().stage2_eval_split == "val"
