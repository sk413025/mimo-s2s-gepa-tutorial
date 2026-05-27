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
