# SkillOpt Implementation Plan

This project extends the original baseline and GEPA tutorial toward a real
SkillOpt-style loop for the OpenClaw `mimo-audio` skill. The earlier offline
candidate rewrite prototype has been removed so the project can focus on real
OpenClaw trajectories.

The live OpenClaw skill stays in:

```text
/home/sbplab/.openclaw/workspace/skills/mimo-audio/SKILL.md
```

The tutorial project owns isolated OpenClaw runs, trajectory export, trajectory
parsing, and later candidate validation. It must never overwrite the live skill
automatically.

SkillOpt-specific code lives under:

```text
src/mimo_s2s_gepa/skillopt/
```

## Tracking Checklist

- [x] Keep baseline and GEPA modes intact.
- [x] Remove the old offline candidate rewrite path.
- [x] Add `configs/openclaw_rollout.yaml`.
- [x] Add `scripts/run_openclaw_rollout.py`.
- [x] Create a temporary isolated OpenClaw agent and workspace.
- [x] Copy the live `mimo-audio` skill into that workspace.
- [x] Run health plus Hank `s2s-smoke` through OpenClaw.
- [x] Export and parse the OpenClaw trajectory bundle.
- [x] Require generated audio for the OpenClaw rollout acceptance gate.
- [x] Add a task dataset for repeated OpenClaw rollouts.
- [x] Save one rollout subdirectory and summary per task.
- [ ] Add a candidate `SKILL.md` patch proposer.
- [ ] Validate candidate skills through fresh OpenClaw rollouts.
- [ ] Add accept/reject promotion reports without touching the live skill.

## Acceptance Commands

```bash
python -m py_compile scripts/*.py src/mimo_s2s_gepa/*.py src/mimo_s2s_gepa/skillopt/*.py
python scripts/run_baseline.py
python scripts/run_gepa.py
python scripts/run_openclaw_rollout.py
```

## Expected OpenClaw Rollout Output

```text
outputs/<timestamp>_openclaw_rollout/
  rollout_report.json
  rollouts/<task_id>/
    task.json
    openclaw_result.json
    parsed_result.json
    rollout_summary.json
    trajectory/
      manifest.json
      events.jsonl
      artifacts.json
      prompts.json
      system-prompt.txt
      tools.json
```

## Acceptance Criteria

- The live OpenClaw skill is not overwritten.
- OpenClaw reads the isolated workspace `mimo-audio/SKILL.md`.
- OpenClaw executes the wrapper health command and `s2s-smoke`.
- The exported trajectory contains tool calls and tool results.
- `parsed_result.json` includes `final_status: success` and a generated
  `audio_path`.
- Temporary OpenClaw agents are deleted after each run.
