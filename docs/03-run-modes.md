# Run Modes

Use the top-level scripts first:

```bash
python scripts/run_baseline.py
python scripts/run_gepa.py
python scripts/run_skillopt.py
```

## Baseline

```bash
python scripts/run_baseline.py
```

This runs one S2S example:

```text
Gemma4 writes instruction -> MiMo S2S generates wav -> Gemma4 scores wav
```

Check:

```text
outputs/*_baseline/summary.json
```

## GEPA

```bash
python scripts/run_gepa.py
```

This runs a small DSPy GEPA loop for the S2S instruction-writing prompt.
Gemma4 is used for the task model, audio evaluator, and GEPA reflection model.

Check:

```text
outputs/*_gepa/summary.json
```

## SkillOpt

```bash
python scripts/run_skillopt.py
```

This runs the teaching SkillOpt flow:

```text
collect -> analyze -> optimize -> validate -> promote review-only
```

The main idea is:

```text
OpenClaw trajectory -> trajectory feedback -> DSPy GEPA proposer optimization
-> candidate SKILL.md -> fresh validation rollouts
```

Check these files first:

```text
outputs/*_trajectory_analysis/trajectory_feedback.json
outputs/*_skillopt_gepa/final_candidate/SKILL.md
outputs/*_skill_validation/decision.json
```

The live OpenClaw skill is not overwritten.

## SkillOpt Debug Steps

Run one step at a time only when you need to inspect a stage:

```bash
python scripts/run_skillopt.py collect
python scripts/run_skillopt.py analyze
python scripts/run_skillopt.py optimize
python scripts/run_skillopt.py validate
python scripts/run_skillopt.py promote
```

What each step means:

- `collect`: run OpenClaw with the workspace `mimo-audio` skill and export a
  trajectory.
- `analyze`: ask Gemma4 to summarize the trajectory into SkillOpt feedback.
- `optimize`: use DSPy GEPA to improve the skill-edit proposer.
- `validate`: compare the live skill and candidate skill with fresh OpenClaw
  rollouts.
- `promote`: write a review-only package with a diff; it does not apply changes.
