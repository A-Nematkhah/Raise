# Configs

Edit **`raise.yaml`** — that is the single experiment config (env + budgets +
PPO + LLM + selection + safety). The comment block at the top of the file lists
run commands and CLI flags.

`raise_config.py` only loads/validates that YAML; do not put run numbers there.
