# Physics-Aware Control-SSM scripts

`water_level_etth1.sh` provides a runnable baseline command for the new `PhysicsAwareControlSSM` model.

For real water-level forecasting, replace:
- `--root_path` / `--data_path` with your water-level dataset path,
- `--data` with `custom`,
- `--enc_in`, `--c_out` and `rain/flow/wind` feature indices to match your schema.
