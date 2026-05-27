# Run Modes

`baseline` asks Gemma4 to write one instruction, sends it to MiMo, then asks
Gemma4 to evaluate the run evidence.

```bash
python scripts/run_baseline.py
```

`gepa` runs a small GEPA loop. Gemma4 writes instructions, evaluates run
evidence, and reflects on feedback.

```bash
python scripts/run_gepa.py
```

For a shared server, keep `max_metric_calls` small in `configs/gepa_light.yaml`.
