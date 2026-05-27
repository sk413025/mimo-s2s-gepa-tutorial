# Concepts

GEPA optimizes the MiMo S2S `instruction`.

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
