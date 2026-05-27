# MiMo S2S GEPA Tutorial

This is a small teaching project for optimizing MiMo-Audio S2S instructions with DSPy GEPA.

The point is not to train MiMo. MiMo S2S always remains the audio generation
model. GEPA optimizes the text `instruction` that is sent to MiMo S2S, using
Gemma4 for the text-side work around that audio model.

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

Run the two modes:

```bash
python scripts/run_baseline.py
python scripts/run_gepa.py
```

Outputs are saved under `outputs/<timestamp>_<mode>/`.

## Modes

- `baseline`: ask Gemma4 to write one instruction, run MiMo S2S once, then ask
  Gemma4 to evaluate the generated audio.
- `gepa`: run a tiny GEPA optimization loop. Gemma4 writes instructions,
  evaluates generated audio, and reflects on feedback.

Each run writes `summary.json` and `predictions.json`. Progress is reported
through Python logging while the run is active.

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
scripts/        short entrypoints for each mode
src/            role-based tutorial modules
docs/           concept notes
outputs/        run outputs, ignored by git
```
