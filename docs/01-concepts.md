# Concepts

GEPA optimizes the MiMo S2S `instruction`.

The SkillOpt direction is about optimizing an OpenClaw `SKILL.md`, but this
project no longer keeps the earlier offline candidate rewrite prototype. The
current SkillOpt foundation is the real OpenClaw trajectory path:

```text
OpenClaw agent + workspace mimo-audio skill
  -> read SKILL.md
  -> run health and s2s-smoke through exec
  -> export trajectory bundle
  -> parse tool calls, tool results, final text, and generated audio
```

Gemma4 is the text model. It has three roles in this tutorial:

- task LM: writes the MiMo S2S instruction
- reflection LM: helps GEPA revise the instruction-writing prompt
- evaluator LM: listens to the generated wav and turns run evidence into score
  and feedback

MiMo S2S is the audio model. It receives:

- `instruction`
- input audio path
- `prompt_examples_json`
- decoding settings

The evaluator gives GEPA a numeric score and plain-language feedback.

The Gemma4 evaluator receives the generated wav through a DSPy multimodal field:

```python
generated_audio: dspy.Audio = dspy.InputField()
```

It also reads metadata, transcript channel, and duration diagnostics.

The OpenClaw trajectory parser is separate from the DSPy evaluator. It records
what the agent actually did, including whether it read the workspace skill,
which commands it ran, and which generated wav path came back from S2S.
