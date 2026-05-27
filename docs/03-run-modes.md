# Run Modes

`smoke` uses a fixed instruction. It checks that MiMo S2S can run.

```bash
python scripts/run_smoke.py
```

`baseline` asks Gemma4 to write one instruction and sends it to MiMo.

```bash
python scripts/run_baseline.py
```

`gepa` runs a small GEPA loop.

```bash
python scripts/run_gepa.py
```

For a shared server, keep `max_metric_calls` small in `configs/gepa_light.yaml`.

