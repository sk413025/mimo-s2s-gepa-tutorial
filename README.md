# MiMo S2S GEPA Tutorial

This is a small teaching project for optimizing MiMo-Audio S2S instructions with DSPy GEPA.

The point is not to train MiMo. GEPA rewrites the text `instruction` that is sent to MiMo S2S.

```text
Hank sample data
  -> Gemma4 vLLM writes an instruction
  -> MiMo S2S wrapper restores the audio
  -> evaluator returns score + feedback
  -> GEPA can try a better instruction
```

In `baseline` and `gepa`, Gemma4 is also used as the evaluator. The evaluator
uses a DSPy multimodal field, `generated_audio: dspy.Audio = dspy.InputField()`,
so Gemma4 can listen to the generated wav. It combines that with the run
metadata, transcript channel, and rule-based evidence.

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
  "evaluator_lm_calls": 1,
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
