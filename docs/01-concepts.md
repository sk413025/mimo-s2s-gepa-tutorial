# Concepts

This project has two related but different optimization examples.

## GEPA For MiMo S2S Instructions

The basic GEPA path optimizes the text `instruction` sent to MiMo S2S.

```text
sample -> Gemma4 writes instruction -> MiMo S2S generates wav
       -> Gemma4 evaluates wav -> GEPA revises instruction-writing prompt
```

MiMo S2S is always the audio model. It receives the instruction and input audio,
then writes the generated wav.

Gemma4 has three roles in this path:

- task LM: writes the S2S instruction
- evaluator LM: listens to the generated wav and returns score plus feedback
- reflection LM: helps GEPA revise the task prompt

The evaluator uses DSPy multimodal input:

```python
generated_audio: dspy.Audio = dspy.InputField()
```

## SkillOpt-Style OpenClaw Skill Optimization

The SkillOpt path is about OpenClaw skill text, not direct MiMo instructions.

```text
OpenClaw runs current mimo-audio SKILL.md
  -> trajectory is exported
  -> Gemma4 analyzes what happened
  -> DSPy GEPA improves the skill-edit proposer prompt
  -> candidate SKILL.md is validated by fresh OpenClaw rollouts
```

GEPA optimizes the DSPy proposer. The proposer writes structured edits for a
candidate `SKILL.md`. The project does not modify DSPy, GEPA, OpenClaw, or the
live OpenClaw skill automatically.

The important distinction:

```text
basic GEPA:   optimize the prompt that writes MiMo S2S instructions
SkillOpt:     optimize the prompt that proposes OpenClaw SKILL.md edits
```
