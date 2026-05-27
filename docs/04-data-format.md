# Data Format

`data/hank_sample.json` is intentionally small.

Each item has:

```json
{
  "id": "s01_m01_ref295",
  "input_audio": "/path/to/calibrated_ldv.wav",
  "prompt_examples_json": "/path/to/5shot_examples.json",
  "transcript": "廚師在節目裡表演拉麵",
  "speaker": "佳",
  "material": "塑膠杯"
}
```

`prompt_examples_json` stays fixed during this tutorial. GEPA changes the instruction.

`data/openclaw_rollout_tasks.jsonl` drives OpenClaw rollout collection. Each
line is one JSON object:

```json
{
  "task_id": "hank_s2s_smoke",
  "task_message": "請使用 mimo-audio skill 跑 Hank S2S smoke...",
  "require_generated_audio": true
}
```

`configs/openclaw_rollout.yaml` can cap the number of rows with `max_tasks` so a
shared-server tutorial run stays lightweight.
