from __future__ import annotations

from .openclaw_cli import OpenClawRolloutPaths


def build_rollout_message(message: str, paths: OpenClawRolloutPaths) -> str:
    return f"""Use tools now. Do not answer with a plan before tool calls.

Step 1: call the read tool on this exact workspace skill path:
{paths.skill_path}

Step 2: call the exec tool with this exact command and wait for completion:
python3 {paths.wrapper_path} health --model mimo_audio_s2s

Step 3: call the exec tool with this exact command and wait for completion:
python3 {paths.wrapper_path} s2s-smoke

If an exec call returns "Command still running", call the process tool to poll/log
the returned session until it completes. Do not stop after starting a background
process. Do not use sessions_spawn. Do not call healthcheck, mimo-audio, or
mimo_audio_s2s as tools. Do not read global/npm skill paths.

Final answer must report: resolved_request, error, backend, audio_path, audio_url,
duration_sec, text_channel, n_prompt_examples.

User task:
{message}
"""


def build_skill_driven_rollout_message(message: str, paths: OpenClawRolloutPaths) -> str:
    return f"""Use tools now. Do not answer with a plan before tool calls.

Step 1: call the read tool on this exact workspace skill path:
{paths.skill_path}

Step 2: follow that workspace skill to satisfy the user task below. Treat the
workspace skill as the source of truth for wrapper commands and reporting fields.
Do not read global/npm skill paths. Do not use sessions_spawn. Do not call
healthcheck, mimo-audio, or mimo_audio_s2s as tools unless they are explicitly
listed in the current tool set.

If an exec call returns "Command still running", call the process tool to
poll/log the returned session until it completes. Do not stop after starting a
background process.

Final answer must report: resolved_request, error, backend, audio_path,
audio_url, duration_sec, text_channel, n_prompt_examples.

User task:
{message}
"""


def build_continuation_message(last_error: str, paths: OpenClawRolloutPaths) -> str:
    return f"""Use tools now. The previous validation did not pass: {last_error}

Continue in this same session. Do not answer with a plan before tool calls.

Call exec with this exact command and wait for completion:
python3 {paths.wrapper_path} s2s-smoke

If exec returns "Command still running", call the process tool to poll/log the
returned session until it completes. Final answer must include audio_path,
audio_url, duration_sec, backend, text_channel, n_prompt_examples,
resolved_request, and error. Do not use sessions_spawn or global/npm skill paths.
"""
