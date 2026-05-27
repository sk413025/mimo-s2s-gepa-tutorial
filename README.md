# MiMo S2S GEPA Tutorial

This is a small teaching project for optimizing MiMo-Audio S2S instructions and
OpenClaw S2S skill text with DSPy.

The point is not to train MiMo. MiMo S2S always remains the audio generation
model. The basic GEPA mode optimizes the text `instruction` that is sent to MiMo
S2S. The SkillOpt GEPA mode optimizes the DSPy proposer that writes candidate
OpenClaw `SKILL.md` files.

The SkillOpt direction is grounded in real OpenClaw runs: the project creates an
isolated OpenClaw agent, runs the workspace `mimo-audio` S2S skill, exports the
trajectory bundle, parses the tool evidence, proposes candidate skill text, and
validates candidates through fresh OpenClaw rollouts. The live skill is never
overwritten automatically.

## Model Roles

MiMo S2S receives an instruction plus an input audio path, then generates the
restored wav file.

Gemma4 has three separate roles:

- `task_lm`: writes the MiMo S2S instruction from the sample context.
- `evaluator_lm`: listens to the generated wav with a DSPy multimodal field,
  then returns score and feedback.
- `reflection_lm`: used only by GEPA to revise the instruction-writing prompt
  from evaluator feedback.

The audio evaluator is expressed as a normal DSPy signature field:

```python
generated_audio: dspy.Audio = dspy.InputField()
```

So the evaluator path is still DSPy-first; this project does not call LiteLLM
directly.

## Flow

```text
baseline:
  Hank sample data
    -> Gemma4 task_lm writes instruction
    -> MiMo S2S wrapper generates wav
    -> Gemma4 evaluator_lm listens to wav and scores it

gepa:
  Hank sample data
    -> Gemma4 task_lm writes instruction
    -> MiMo S2S wrapper generates wav
    -> Gemma4 evaluator_lm listens to wav and gives feedback
    -> Gemma4 reflection_lm helps GEPA revise the instruction-writing prompt

openclaw_rollout:
  isolated OpenClaw agent + copied mimo-audio skill
    -> read workspace SKILL.md
    -> exec wrapper health check
    -> exec s2s-smoke
    -> export trajectory bundle
    -> parse tool calls, tool results, and generated audio

trajectory_analysis:
  latest OpenClaw rollout report + parsed trajectories
    -> Gemma4 analyzer reads compact trajectory evidence
    -> write structured SkillOpt feedback JSON
    -> do not mutate SKILL.md

skill_candidate:
  live OpenClaw SKILL.md + trajectory_feedback.json
    -> Gemma4 proposes small structured edits
    -> apply edits to produce a candidate SKILL.md
    -> validate candidate scope and required S2S fields
    -> write candidate file, edits, and diff
    -> do not overwrite the live skill

skill_validation:
  live SKILL.md and candidate SKILL.md
    -> run fresh baseline OpenClaw rollouts
    -> run fresh candidate OpenClaw rollouts
    -> compare validation scores
    -> accept only on strict improvement

skillopt_gepa:
  trajectory_feedback.json + live SKILL.md
    -> DSPy GEPA optimizes the skill candidate proposer prompt
    -> each metric call writes a candidate SKILL.md
    -> fresh OpenClaw candidate rollout scores the candidate
    -> validation feedback is returned to GEPA
```

## Quick Start

Check the two OpenAI-compatible services:

```bash
curl http://100.70.253.93:8000/v1/models
curl http://100.70.78.122:19080/v1/models
```

Current service locations:

```text
Gemma4 vLLM
  Tailscale IP: 100.70.253.93
  Port: 8000
  OpenAI base URL: http://100.70.253.93:8000/v1
  Model id: google/gemma-4-E4B-it

MiMo audio wrapper
  Hostname: sbplab
  Tailscale IP: 100.70.78.122
  Port: 19080
  OpenAI base URL: http://100.70.78.122:19080/v1
  Model id: mimo-audio-s2s

MiMo Triton backend
  Same audio machine as the wrapper
  Port: 18000
  Direct Triton model: mimo_audio_s2s
```

To check the Tailscale IP of the machine you are on:

```bash
tailscale ip -4
```

If `tailscale` is not in your shell path, inspect the interface directly:

```bash
ip -4 addr show tailscale0
```

Run the available modes:

```bash
python scripts/run_baseline.py
python scripts/run_gepa.py
python scripts/run_openclaw_rollout.py
python scripts/analyze_rollouts.py
python scripts/propose_skill_candidate.py
python scripts/validate_skill_candidate.py
python scripts/run_skillopt_gepa.py
```

Outputs are saved under `outputs/<timestamp>_<mode>/`.

## Modes

- `baseline`: ask Gemma4 to write one instruction, run MiMo S2S once, then ask
  Gemma4 to evaluate the generated audio.
- `gepa`: run a tiny GEPA optimization loop. Gemma4 writes instructions,
  evaluates generated audio, and reflects on feedback.
- `openclaw_rollout`: run one or more isolated OpenClaw tasks with the
  `mimo-audio` skill and export trajectory bundles. This is the data collection
  layer for true trajectory-based SkillOpt. Its default config requires a
  generated audio path, so an OpenClaw run that finishes without S2S audio is
  treated as a failed validation.
- `trajectory_analysis`: ask Gemma4 to analyze a rollout run and produce
  structured SkillOpt feedback. This stage only writes analysis files; it does
  not write or edit candidate skills.
- `skill_candidate`: ask Gemma4 to propose small structured edits from
  trajectory feedback. The project applies those edits to produce a candidate
  `SKILL.md`, writes candidate artifacts and a diff, but does not overwrite the
  live OpenClaw skill.
- `skill_validation`: run fresh baseline and candidate OpenClaw rollouts, then
  accept the candidate only if it strictly improves the validation score. Ties are
  rejected.
- `skillopt_gepa`: use `dspy.GEPA` to optimize the candidate skill proposer.
  The metric reuses the OpenClaw validation path, so feedback comes from fresh
  candidate rollouts instead of only offline text review.

Baseline and GEPA runs write `summary.json` and `predictions.json`.
`openclaw_rollout` writes one subdirectory per task under `rollouts/`, plus a
top-level `rollout_report.json`. Progress is reported through Python logging
while the run is active.
`trajectory_analysis` writes `trajectory_feedback.json` and
`rollout_evidence.json`.
`skill_candidate` writes an initial skill snapshot, structured edits, candidate
skill, diff, and `proposal.json`.
`skill_validation` writes baseline/candidate rollout reports plus `decision.json`.
`skillopt_gepa` writes a fresh baseline report, per-metric-call candidates and
decisions, aggregate `stats.json`, plus a final candidate proposal.

The generated wav files are written by the MiMo audio wrapper. The run summary
records both `audio_path` and `audio_url`, for example:

```text
/home/sbplab/jiawei/triton_mimo_audio/outputs/tool/<id>.wav
http://100.70.78.122:19080/audio/<id>.wav
```

## Files

```text
configs/        service URLs and run budgets
data/           tiny Hank sample dataset
data/openclaw_skillopt_validation_tasks.jsonl
                stricter SkillOpt validation tasks for OpenClaw rollouts
scripts/        short entrypoints for each mode
src/            role-based tutorial modules
src/mimo_s2s_gepa/skillopt/
                OpenClaw runner and trajectory parsing helpers
docs/           concept notes
outputs/        run outputs, ignored by git
```

OpenClaw rollout runs add:

```text
outputs/<timestamp>_openclaw_rollout/
  rollout_report.json
  rollouts/<task_id>/
    task.json
    openclaw_result.json
    parsed_result.json
    rollout_summary.json
    trajectory/
      events.jsonl
      artifacts.json
      prompts.json
      system-prompt.txt
      tools.json
```

Trajectory analysis runs add:

```text
outputs/<timestamp>_trajectory_analysis/
  trajectory_feedback.json
  rollout_evidence.json
  summary.json
```

Skill candidate runs add:

```text
outputs/<timestamp>_skill_candidate/
  initial/SKILL.md
  candidates/skill_v0001/SKILL.md
  candidates/skill_v0001/diff.md
  candidates/skill_v0001/edits.json
  proposal.json
  summary.json
```

Skill validation runs add:

```text
outputs/<timestamp>_skill_validation/
  baseline_report.json
  candidate_report.json
  decision.json
  summary.json
  baseline_rollout/
  candidate_rollout/
```

SkillOpt validation tasks can run in two modes:

- `guided`: the harness gives OpenClaw the exact health and smoke commands.
- `skill_driven`: the harness only tells OpenClaw to read the workspace skill
  and follow it, so candidate `SKILL.md` text can affect execution.

Each task can require tool calls, command evidence, audio metadata fields,
backend identity, minimum duration, and an existing generated wav file.
Tasks that complete cleanly in one turn score higher than tasks that only pass
after a continuation turn.

SkillOpt GEPA runs add:

```text
outputs/<timestamp>_skillopt_gepa/
  baseline_report.json
  metric_calls/call_001/
    proposal.json
    candidate/edits.json
    candidate/SKILL.md
    candidate_report.json
    decision.json
    metric_result.json
  final_candidate/SKILL.md
  final_candidate/edits.json
  stats.json
  summary.json
```

## References

- GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning,
  arXiv:2507.19457, https://arxiv.org/abs/2507.19457
- SkillOpt: Executive Strategy for Self-Evolving Agent Skills,
  arXiv:2605.23904, https://arxiv.org/abs/2605.23904
