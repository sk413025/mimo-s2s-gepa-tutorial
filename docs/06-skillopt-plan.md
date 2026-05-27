# SkillOpt Implementation Notes

This is the advanced implementation note for the SkillOpt part of the tutorial.
For normal usage, start with the README and `docs/03-run-modes.md`.

## Goal

The project demonstrates a SkillOpt-style loop without modifying DSPy, GEPA,
OpenClaw, or the live OpenClaw skill.

```text
current OpenClaw mimo-audio skill
  -> fresh OpenClaw rollout
  -> exported trajectory
  -> Gemma4 trajectory analysis
  -> DSPy GEPA optimizes the skill-edit proposer prompt
  -> candidate SKILL.md
  -> fresh OpenClaw validation
  -> accept/reject decision
```

The live skill stays here:

```text
/home/sbplab/.openclaw/workspace/skills/mimo-audio/SKILL.md
```

The tutorial copies that skill into isolated OpenClaw workspaces for rollout and
validation. It never overwrites the live file.

## Implemented Pieces

- `scripts/run_skillopt.py` is the single beginner-facing SkillOpt entrypoint.
- `collect` creates an isolated OpenClaw agent and exports a trajectory bundle.
- `analyze` turns rollout evidence into structured feedback with Gemma4.
- `optimize` runs `dspy.GEPA.compile(...)` on a DSPy program that proposes
  structured skill edits.
- `validate` compares baseline and candidate skills with fresh OpenClaw rollouts.
- `promote` writes a review-only diff package and does not apply changes.

The old offline candidate rewrite path and duplicate stage scripts have been
removed. Candidate generation now goes through the DSPy GEPA proposer path.

## Key Modules

```text
src/mimo_s2s_gepa/skillopt/
  rollout.py              OpenClaw rollout orchestration
  trajectory_parser.py    OpenClaw trajectory parsing
  trajectory_analyzer.py  Gemma4 trajectory feedback
  candidate_edits.py      DSPy signature and edit constraints
  skillopt_gepa.py        DSPy GEPA proposer optimization
  validation_gate.py      baseline vs candidate rollout comparison
  promotion.py            review-only candidate diff package
```

## Acceptance Checks

Use these after changing code:

```bash
python -m py_compile scripts/*.py src/mimo_s2s_gepa/*.py src/mimo_s2s_gepa/skillopt/*.py
python scripts/run_skillopt.py --help
```

Use these for live integration checks when the services are available:

```bash
python scripts/run_skillopt.py collect
python scripts/run_skillopt.py analyze
python scripts/run_skillopt.py optimize
python scripts/run_skillopt.py validate
python scripts/run_skillopt.py promote
```

## What To Inspect

After `collect`:

```text
outputs/*_openclaw_rollout/rollout_report.json
outputs/*_openclaw_rollout/rollouts/<task_id>/trajectory/
```

After `analyze`:

```text
outputs/*_trajectory_analysis/trajectory_feedback.json
```

After `optimize`:

```text
outputs/*_skillopt_gepa/final_candidate/SKILL.md
outputs/*_skillopt_gepa/final_candidate/edits.json
outputs/*_skillopt_gepa/stats.json
```

After `validate`:

```text
outputs/*_skill_validation/decision.json
```

After `promote`:

```text
outputs/*_skill_promotion/diff.md
outputs/*_skill_promotion/candidate/SKILL.md
outputs/*_skill_promotion/live_before/SKILL.md
```

## Design Boundaries

- GEPA optimizes the DSPy proposer prompt, not OpenClaw itself.
- OpenClaw trajectories come from real OpenClaw runs, not DSPy traces.
- Candidate skills are validated with fresh rollouts.
- Acceptance requires the candidate score to be higher than the baseline score.
- The promotion stage is review-only.

## Advanced Traceability

The code also keeps an append-only candidate registry under:

```text
outputs/candidate_registry.jsonl
```

This is useful for comparing multiple runs, but it is not needed to understand
the beginner flow. The primary teaching artifacts are the per-run `summary.json`,
`trajectory_feedback.json`, candidate `SKILL.md`, and `decision.json` files.
