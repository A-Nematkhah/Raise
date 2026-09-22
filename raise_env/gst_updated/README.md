# GST (Gumbel Social Transformer) — inference only

This subtree ships the **pretrained GST predictor** used by CrowdNav++ /
EvoNav when `sim.predict_method = 'inferred'`.

Training scripts, dataset builders, MGNN/PEC-Net loaders, and tuning shells
were removed from this fork. Obtain or keep checkpoints under:

```
gst_updated/results/100-gumbel_social_transformer-...-seed_1000/sj/checkpoint/epoch_100.pt
gst_updated/results/100-gumbel_social_transformer-...-seed_1000_rand/sj/checkpoint/epoch_100.pt
```

(`results/` is gitignored; download via `run/download_datasets_models.sh` or
copy from the [upstream CrowdNav++ release](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph).)

## Runtime entry used by EvoNav

- Model: `src/gumbel_social_transformer/st_model.py`
- Wrapper: `scripts/wrapper/crowd_nav_interface_parallel.py`
- Loaded by: `evonav_env/rl/vec_env/vec_pretext_normalize.py`

Set `pred.model_dir` in `crowd_nav/configs/config.py` (or the regime helpers in
`crowd_nav/reward_search/regime.py`) to the matching `.../sj` directory.
