# MiMo S2S GEPA Tutorial

This is a small teaching project for optimizing MiMo-Audio S2S instructions with DSPy GEPA.

The point is not to train MiMo. GEPA rewrites the text `instruction` that is sent to MiMo S2S.

```text
Hank sample data
  -> Gemma4 vLLM writes an instruction
  -> MiMo S2S wrapper restores the audio
  -> metric returns score + feedback
  -> GEPA can try a better instruction
```

## Quick Start

Check the two OpenAI-compatible services:

```bash
curl http://100.70.253.93:8000/v1/models
curl http://127.0.0.1:19080/v1/models
```

Run the three modes:

```bash
python scripts/run_smoke.py
python scripts/run_baseline.py
python scripts/run_gepa.py
```

Outputs are saved under `outputs/<timestamp>_<mode>/`.

## Modes

- `smoke`: use a fixed Hank instruction and only test the MiMo S2S path.
- `baseline`: ask Gemma4 to write one instruction, then run MiMo S2S once.
- `gepa`: run a tiny GEPA optimization loop.

Each run writes `summary.json`, including logical counters:

```json
{
  "task_lm_calls": 1,
  "reflection_lm_calls": 0,
  "mimo_s2s_calls": 5
}
```

## Files

```text
configs/        service URLs and run budgets
data/           tiny Hank sample dataset
scripts/        short entrypoints for each mode
src/            role-based tutorial modules
docs/           concept notes
outputs/        run outputs, ignored by git
```

