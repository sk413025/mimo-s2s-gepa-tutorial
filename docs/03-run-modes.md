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

`skillopt` reads the OpenClaw `mimo-audio` skill, writes a candidate `SKILL.md`,
validates the current skill and candidate with MiMo S2S plus Gemma4 audio
scoring, then writes a decision report.

```bash
python scripts/run_skillopt.py
```

SkillOpt-lite output includes `initial/SKILL.md`,
`candidates/skill_v0001/SKILL.md`, `diff.md`, `training_report.json`,
`baseline_report.json`, `candidate_report.json`, `decision.json`, and
`summary.json`.

For a shared server, keep `max_metric_calls` small in `configs/gepa_light.yaml`.
SkillOpt-lite always writes candidates and reports instead of overwriting the
live OpenClaw skill.
