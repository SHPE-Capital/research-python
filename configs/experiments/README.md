# Experiment configs

One YAML file describes one run: `uv run shpe run configs/experiments/<file>.yaml`.
Copy `hello_model.yaml` to start a new one.

| Section | Meaning |
|---|---|
| `name` | Run directory suffix: `runs/<timestamp>-<name>/` |
| `seed` | Passed to the model |
| `data` | Where bars come from. `source: fixture` (synthetic; `symbols`, `start`, `end`, `seed`, `signal_strength`) or `source: parquet` (`path` to a contract bars file). The `store` and `snapshot` sources are added with D1 and D5. |
| `session` | Which bars to use: `regular`, `pre`, `post` or `all` |
| `features` | List of `{type: <feature_type>, ...params}` from the feature registry |
| `label` | `forward_return` with `horizon_minutes`, `target` (`sign` or `value`) and `sample_every_minutes` |
| `split` | Fractions of sessions for `train`/`validation`/`test`, plus `gap_days` |
| `model` | `type` from the model registry, plus `params` |

Every run directory contains the resolved `config.yaml`, so a run can be reproduced from its own folder.
