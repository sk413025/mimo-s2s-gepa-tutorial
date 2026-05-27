# Concepts

GEPA optimizes the MiMo S2S `instruction`.

SkillOpt-lite optimizes a candidate OpenClaw `SKILL.md` for MiMo-Audio S2S.
It does not update model weights and does not overwrite the live skill by
default.

Gemma4 is the text model. It has three roles in this tutorial:

- task LM: writes the MiMo S2S instruction
- reflection LM: helps GEPA revise the instruction-writing prompt
- evaluator LM: listens to the generated wav and turns run evidence into score
  and feedback
- skill proposer LM: rewrites a candidate OpenClaw `SKILL.md` from run feedback

MiMo S2S is the audio model. It receives:

- `instruction`
- input audio path
- `prompt_examples_json`
- decoding settings

The evaluator gives GEPA a numeric score and plain-language feedback.

SkillOpt-lite uses the same evaluator for its validation gate. A candidate skill
is accepted only when its validation score improves over the current skill
snapshot.

The Gemma4 evaluator receives the generated wav through a DSPy multimodal field:

```python
generated_audio: dspy.Audio = dspy.InputField()
```

It also reads metadata, transcript channel, and duration diagnostics.
