# Experiment configs

One YAML file describes one run: `uv run shpe run configs/experiments/<file>.yaml`.
Copy `hello_model.yaml` to start a new one.

| Section | Meaning |
|---|---|
| `name` | Run directory suffix: `runs/<timestamp>-<name>/` |
| `seed` | Passed to the model |
| `data` | Where bars come from. `source: fixture` (synthetic; `symbols`, `start`, `end`, `seed`, `signal_strength`), `source: parquet` (`path` to a contract bars file), or `source: store` (`symbols`, `start`, `end`, optional `timeframe`, `feed`, `adjustment`) for bars downloaded with `shpe data bars`. The `snapshot` source arrives with D5. |
| `session` | Which bars to use: `regular`, `pre`, `post` or `all` |
| `features` | List of `{type: <feature_type>, ...params}` from the feature registry |
| `label` | `forward_return` with `horizon_minutes`, `target` (`sign` or `value`) and `sample_every_minutes` |
| `split` | Fractions of sessions for `train`/`validation`/`test`, plus `gap_days` |
| `model` | `type` from the model registry, plus `params` |

Every run directory contains the resolved `config.yaml`, so a run can be reproduced from its own folder.
