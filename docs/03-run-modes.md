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

`openclaw_rollout` creates isolated OpenClaw agents, copies the workspace
`mimo-audio` skill, runs task rows from `data/openclaw_rollout_tasks.jsonl`,
exports each trajectory, and parses generated audio paths.

```bash
python scripts/run_openclaw_rollout.py
```

OpenClaw rollout output includes a top-level `rollout_report.json` and one
`rollouts/<task_id>/` directory per collected task. Each task directory includes
`task.json`, `openclaw_result.json`, `parsed_result.json`, per-turn result
snapshots, and a `trajectory/` bundle.

For a shared server, keep `max_metric_calls` small in `configs/gepa_light.yaml`.
The OpenClaw rollout collector uses temporary agents and deletes them after
trajectory export; it does not overwrite the live OpenClaw skill.
