# Concepts

GEPA optimizes the MiMo S2S `instruction`.

Gemma4 is the text model. It writes or rewrites the instruction.

MiMo S2S is the audio model. It receives:

- `instruction`
- input audio path
- `prompt_examples_json`
- decoding settings

The metric gives GEPA a numeric score and plain-language feedback.

