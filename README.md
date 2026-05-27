# MiMo S2S GEPA Tutorial

A small teaching project for using DSPy with MiMo-Audio S2S.

The project has two learning goals:

1. Use DSPy GEPA to optimize the `instruction` sent to MiMo S2S.
2. Use real OpenClaw trajectories to optimize the prompt that proposes candidate
   `mimo-audio` `SKILL.md` edits.

MiMo is not trained here. MiMo remains the audio model. DSPy and GEPA only
optimize text.

## What Runs

```text
baseline
  Gemma4 writes one S2S instruction
  MiMo S2S generates a wav
  Gemma4 listens to the wav and scores it

gepa
  DSPy GEPA improves the instruction-writing prompt
  MiMo S2S is called during evaluation
  Gemma4 provides audio feedback and reflection

skillopt
  OpenClaw runs the workspace mimo-audio skill
  trajectories are exported and analyzed
  DSPy GEPA improves the candidate skill-edit proposer
  a candidate SKILL.md is validated with fresh OpenClaw rollouts
```

The live OpenClaw skill is never overwritten automatically.

## Service Map

This tutorial uses two OpenAI-compatible model endpoints plus one OpenClaw
Gateway endpoint reachable through Tailscale:

```text
Gemma4 vLLM:         http://100.70.253.93:8000/v1
MiMo audio wrapper:  http://100.70.78.122:19080/v1
MiMo Triton:         100.70.78.122:18000
OpenClaw Gateway:    100.70.78.122:18788
```

Model endpoint checks:

```bash
curl http://100.70.253.93:8000/v1/models
curl http://100.70.78.122:19080/v1/models
```

OpenClaw is the agent runtime used by `scripts/run_skillopt.py`. It is not an
OpenAI-compatible model endpoint, so it does not expose `/v1/models`. On this
machine it is configured with `bind: tailnet` and port `18788`.

To check the Tailscale IP of the current machine:

```bash
tailscale ip -4
```

or:

```bash
ip -4 addr show tailscale0
```

## Quick Start

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

Run the three tutorials:

```bash
python scripts/run_baseline.py
python scripts/run_gepa.py
python scripts/run_skillopt.py
```

Outputs are written to:

```text
outputs/<timestamp>_<mode>/
```

For most runs, start by opening `summary.json` in the newest output directory.

## Script Guide

`baseline` is the smallest end-to-end S2S example:

```bash
python scripts/run_baseline.py
```

`gepa` shows DSPy GEPA optimizing the instruction-writing prompt:

```bash
python scripts/run_gepa.py
```

`skillopt` shows the OpenClaw trajectory loop:

```bash
python scripts/run_skillopt.py
```

The SkillOpt flow is:

```text
collect -> analyze -> optimize -> validate -> promote review-only
```

You can run one SkillOpt step while debugging:

```bash
python scripts/run_skillopt.py collect
python scripts/run_skillopt.py analyze
python scripts/run_skillopt.py optimize
python scripts/run_skillopt.py validate
python scripts/run_skillopt.py promote
```

`promote` only writes a review package and diff. It does not modify the live
OpenClaw skill.

## SkillOpt In One Picture

```text
current OpenClaw mimo-audio skill
        |
        v
fresh OpenClaw S2S rollout
        |
        v
trajectory bundle
        |
        v
Gemma4 trajectory analysis
        |
        v
DSPy GEPA optimizes skill-edit proposer
        |
        v
candidate SKILL.md
        |
        v
fresh OpenClaw validation
        |
        v
accept/reject decision
```

The important files after a SkillOpt run are:

```text
outputs/*_trajectory_analysis/trajectory_feedback.json
outputs/*_skillopt_gepa/final_candidate/SKILL.md
outputs/*_skill_validation/decision.json
```

## Project Layout

```text
configs/        service URLs and small run budgets
data/           tiny Hank sample and OpenClaw task rows
scripts/        beginner-facing entrypoints
src/            tutorial implementation
docs/           extra notes
outputs/        generated run outputs, ignored by git
```

SkillOpt-specific implementation lives under:

```text
src/mimo_s2s_gepa/skillopt/
```

## More Notes

- [docs/01-concepts.md](docs/01-concepts.md): model roles and GEPA vs SkillOpt.
- [docs/02-setup.md](docs/02-setup.md): endpoint setup.
- [docs/03-run-modes.md](docs/03-run-modes.md): what each script does.
- [docs/04-data-format.md](docs/04-data-format.md): input data shape.
- [docs/06-skillopt-plan.md](docs/06-skillopt-plan.md): implementation details.

## References

- GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning,
  arXiv:2507.19457, https://arxiv.org/abs/2507.19457
- SkillOpt: Executive Strategy for Self-Evolving Agent Skills,
  arXiv:2605.23904, https://arxiv.org/abs/2605.23904
