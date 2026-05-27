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

`openclaw_task` creates an isolated OpenClaw agent, copies the workspace
`mimo-audio` skill, runs health plus Hank `s2s-smoke`, exports the trajectory,
and parses the generated audio path.

```bash
python scripts/run_openclaw_task.py
```

OpenClaw task output includes `task.json`, `openclaw_result.json`,
`parsed_result.json`, per-turn result snapshots, and a `trajectory/` bundle.

For a shared server, keep `max_metric_calls` small in `configs/gepa_light.yaml`.
The OpenClaw task runner uses a temporary agent and deletes it after trajectory
export; it does not overwrite the live OpenClaw skill.
