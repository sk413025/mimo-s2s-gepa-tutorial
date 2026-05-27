# SkillOpt-lite Implementation Plan

This project extends the original baseline and GEPA tutorial with a small
SkillOpt-like loop for the OpenClaw `mimo-audio` skill.

The live OpenClaw skill stays in:

```text
/home/sbplab/.openclaw/workspace/skills/mimo-audio/SKILL.md
```

The tutorial project owns training, validation, candidate files, diffs, and
reports. The first version never overwrites the live skill automatically.

## Tracking Checklist

- [x] Keep baseline and GEPA modes intact.
- [x] Add `configs/skillopt_light.yaml`.
- [x] Add `scripts/run_skillopt.py`.
- [x] Read the live OpenClaw skill into each SkillOpt run.
- [x] Write `initial/SKILL.md` and `candidates/skill_v0001/SKILL.md`.
- [x] Use Gemma4 to propose a candidate skill rewrite.
- [x] Validate baseline and candidate with MiMo S2S plus Gemma4 audio scoring.
- [x] Write `diff.md`, `decision.json`, and `summary.json`.
- [x] Reject candidates that reintroduce non-S2S workflow terms.
- [x] Run static checks and the three tutorial modes.

## Acceptance Commands

```bash
python -m py_compile scripts/*.py src/mimo_s2s_gepa/*.py
python scripts/run_baseline.py
python scripts/run_gepa.py
python scripts/run_skillopt.py
```

## Expected SkillOpt Output

```text
outputs/<timestamp>_skillopt/
  initial/SKILL.md
  candidates/skill_v0001/SKILL.md
  candidates/skill_v0001/diff.md
  training_report.json
  baseline_report.json
  candidate_report.json
  decision.json
  summary.json
```

## Acceptance Criteria

- The live OpenClaw skill is not overwritten.
- Candidate validation uses generated wav files, not text-only scoring.
- Gemma4 listens to generated wav files through DSPy `dspy.Audio`.
- Candidate `SKILL.md` remains scoped to MiMo-Audio S2S / LDV restoration.
- Reports show baseline score, candidate score, decision, candidate path, and
  generated audio paths.
