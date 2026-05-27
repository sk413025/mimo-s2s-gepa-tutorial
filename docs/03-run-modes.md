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

`trajectory_analysis` reads a rollout run and asks Gemma4 for structured
SkillOpt feedback.

```bash
python scripts/analyze_rollouts.py
```

Leave `rollout_run_dir` empty in `configs/analyze_rollouts.yaml` to analyze the
latest `outputs/*_openclaw_rollout` directory. The analysis stage writes
`trajectory_feedback.json` and does not modify `SKILL.md`.

`skillopt_gepa` uses DSPy GEPA to optimize the prompt behind the SkillOpt
candidate proposer.

```bash
python scripts/run_skillopt_gepa.py
```

Leave `trajectory_analysis_dir` empty in `configs/skillopt_gepa_light.yaml` to
use the latest `outputs/*_trajectory_analysis` directory. The metric writes a
candidate edit set and candidate skill for each GEPA metric call, validates it
with fresh OpenClaw rollouts, and returns the validation decision as GEPA
feedback. This mode does not overwrite the live OpenClaw skill. Metric
candidates and the final candidate are recorded in
`outputs/candidate_registry.jsonl`.

`skill_validation` runs fresh baseline and final-candidate OpenClaw rollouts,
then writes an accept/reject decision.

```bash
python scripts/validate_skill_candidate.py
```

Leave `candidate_dir` empty in `configs/validate_skill_candidate.yaml` to use
the latest `outputs/*_skillopt_gepa/final_candidate` directory. The validation
stage rejects ties and only accepts a candidate when it scores higher than the
live baseline. The default task file is
`data/openclaw_skillopt_validation_tasks.jsonl`, which includes both guided and
skill-driven OpenClaw tasks. It also appends a validation row to
`outputs/candidate_registry.jsonl`.

`skill_promotion` turns a validation result into a reviewable promotion package.

```bash
python scripts/promote_candidate.py
```

Leave `skill_validation_dir` empty in `configs/promote_candidate.yaml` to use
the latest `outputs/*_skill_validation` directory. The default config is
dry-run: it writes `promotion_report.json`, `diff.md`, `candidate/SKILL.md`,
and `live_before/SKILL.md`, but it does not overwrite the live OpenClaw skill.
Set `apply: true` only after reviewing the report. Rejected or tied candidates
are blocked unless `allow_rejected: true` is set intentionally. For manual use,
the script also accepts `--apply` and `--allow-rejected`.
