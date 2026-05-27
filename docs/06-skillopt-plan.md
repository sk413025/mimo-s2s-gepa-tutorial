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
- [x] Add a trajectory analyzer that turns rollouts into structured feedback.
- [x] Add a candidate `SKILL.md` proposer.
- [x] Validate candidate skills through fresh OpenClaw rollouts.
- [x] Add accept/reject promotion reports without touching the live skill.
- [x] Add a DSPy GEPA stage that optimizes the skill candidate proposer.
- [x] Add stricter guided and skill-driven validation tasks.

## Acceptance Commands

```bash
python -m py_compile scripts/*.py src/mimo_s2s_gepa/*.py src/mimo_s2s_gepa/skillopt/*.py
python scripts/run_baseline.py
python scripts/run_gepa.py
python scripts/run_openclaw_rollout.py
python scripts/analyze_rollouts.py
python scripts/propose_skill_candidate.py
python scripts/validate_skill_candidate.py
python scripts/run_skillopt_gepa.py
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

## Expected Trajectory Analysis Output

```text
outputs/<timestamp>_trajectory_analysis/
  trajectory_feedback.json
  rollout_evidence.json
  summary.json
```

## Analysis Criteria

- The analyzer reads rollout outputs, not the live OpenClaw session directly.
- Gemma4 returns structured feedback with success patterns, failure patterns,
  skill issues, suggested changes, evidence, and next validation tasks.
- The analyzer does not create or modify any candidate `SKILL.md`.

## Expected Skill Candidate Output

```text
outputs/<timestamp>_skill_candidate/
  initial/SKILL.md
  candidates/skill_v0001/SKILL.md
  candidates/skill_v0001/diff.md
  proposal.json
  summary.json
```

## Candidate Criteria

- The live OpenClaw skill is not overwritten.
- The candidate keeps YAML frontmatter and `name: mimo-audio`.
- The candidate remains scoped to MiMo-Audio S2S / LDV restoration.
- The candidate keeps `mimo_audio_s2s`, health, `s2s`, and `s2s-smoke` guidance.
- The candidate does not introduce TTS or unrelated workflows.

## Expected Skill Validation Output

```text
outputs/<timestamp>_skill_validation/
  baseline_report.json
  candidate_report.json
  decision.json
  summary.json
  baseline_rollout/
  candidate_rollout/
```

## Validation Criteria

- Baseline and candidate both use fresh OpenClaw rollouts.
- Candidate rollout uses `candidates/skill_v0001/SKILL.md`.
- The live OpenClaw skill is not overwritten.
- `decision.json` records pass counts, validation scores, rollout directories,
  candidate path, and accepted/rejected reason.
- Candidate is accepted only when it scores higher than baseline; ties are
  rejected.
- Validation tasks check required tool names, command evidence, generated audio
  fields, backend, duration, and generated wav file existence.
- Skill-driven tasks require OpenClaw to derive wrapper commands from the
  workspace `SKILL.md` instead of receiving the exact smoke command from the
  harness.
- Clean first-turn completion scores higher than completion that needs a
  continuation turn, so validation can distinguish fragile task execution from
  direct success.

## Expected SkillOpt GEPA Output

```text
outputs/<timestamp>_skillopt_gepa/
  baseline_report.json
  metric_calls/call_001/
    proposal.json
    candidate/SKILL.md
    candidate_report.json
    decision.json
    metric_result.json
  final_candidate/SKILL.md
  final_candidate/proposal.json
  stats.json
  summary.json
```

## SkillOpt GEPA Criteria

- `dspy.GEPA.compile(...)` is used on a DSPy program that proposes candidate
  `SKILL.md` files.
- GEPA optimizes the proposer prompt, not the OpenClaw runtime or DSPy itself.
- Every metric call validates a candidate through fresh OpenClaw rollout.
- The live OpenClaw skill is not overwritten.
- Metric feedback includes baseline pass count, candidate pass count,
  validation scores, decision reason, and generated audio paths when present.
