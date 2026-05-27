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

`skill_candidate` reads the latest trajectory analysis and asks Gemma4 for a
complete candidate `SKILL.md`.

```bash
python scripts/propose_skill_candidate.py
```

Leave `trajectory_analysis_dir` empty in `configs/propose_skill_candidate.yaml`
to use the latest `outputs/*_trajectory_analysis` directory. The candidate stage
writes `initial/SKILL.md`, `candidates/skill_v0001/SKILL.md`, `diff.md`, and
`proposal.json`; it does not overwrite the live OpenClaw skill.

`skill_validation` runs fresh baseline and candidate OpenClaw rollouts, then
writes an accept/reject decision.

```bash
python scripts/validate_skill_candidate.py
```

Leave `skill_candidate_dir` empty in `configs/validate_skill_candidate.yaml` to
use the latest `outputs/*_skill_candidate` directory. The validation stage
rejects ties and only accepts a candidate when it passes more rollout tasks than
the live baseline.
