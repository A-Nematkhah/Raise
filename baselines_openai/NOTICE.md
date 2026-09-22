# Third-party notice: OpenAI Baselines

This directory contains a vendored copy of [OpenAI Baselines](https://github.com/openai/baselines)
(commit `ea25b9e8`, upstream `main` as of import into this repository).

- **Copyright:** OpenAI (2017)
- **License:** MIT — see `LICENSE` in this directory
- **Used by:** `raise_env/rl/networks/` and `raise_env/rl/vec_env/` for `VecEnv`,
  `VecNormalize`, logging, and related PPO training utilities.

Install into your virtual environment (from repo root; **TensorFlow must be
installed first** — `setup.py` asserts it at install time):

```bash
pip install tensorflow
pip install -e baselines_openai
```

Do not modify the MIT license text in `LICENSE`.
