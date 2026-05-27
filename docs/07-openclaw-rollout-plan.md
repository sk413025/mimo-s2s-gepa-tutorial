# OpenClaw Rollout Plan

This phase validates the real OpenClaw execution path before wiring it into the
SkillOpt loop.

The runner should:

- create a temporary isolated OpenClaw agent and workspace
- copy the current `mimo-audio` skill into that workspace
- rewrite workspace-local skill paths so the agent reads the isolated copy
- write workspace guidance that points the agent at the isolated `SKILL.md`
- load task rows from `data/openclaw_rollout_tasks.jsonl`
- run non-interactive OpenClaw tasks with `openclaw agent`
- continue the same OpenClaw session for a small number of turns if the agent
  starts a background process but stops before S2S audio is produced
- export the trajectory bundle with `openclaw sessions export-trajectory`
- parse tool calls, tool results, final assistant text, and generated audio
- delete the temporary OpenClaw agent after export

## Acceptance Command

```bash
python scripts/run_openclaw_rollout.py
```

## Expected Output

```text
outputs/<timestamp>_openclaw_rollout/
  rollout_report.json
  rollouts/<task_id>/
    task.json
    openclaw_result.json
    openclaw_result_turn_1.json
    parsed_result.json
    parsed_result_turn_1.json
    rollout_summary.json
    trajectory/
      manifest.json
      events.jsonl
      session-branch.json
      metadata.json
      artifacts.json
      prompts.json
      system-prompt.txt
      tools.json
```

## Acceptance Criteria

- The task runs through `openclaw agent`, not direct MiMo wrapper calls.
- The agent reads the isolated workspace `mimo-audio/SKILL.md`, not a bundled
  or global skill path.
- The exported trajectory contains `tool.call` and `tool.result` events.
- `parsed_result.json` includes final status, tool calls, tool results, and the
  generated `audio_path`.
- `configs/openclaw_rollout.yaml` sets `require_generated_audio: true`, so a run
  that reaches OpenClaw `success` but does not produce audio still fails.
- The temporary OpenClaw agent is deleted at the end of the run.
- Existing `baseline` and `gepa` modes remain unchanged.
