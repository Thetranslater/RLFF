AReaL checkpoints and runtime state are written below this directory. The RLFF
configuration also creates four append-only JSONL files during training:

- `reward-audit.jsonl`: every completion reward and trajectory reward after
  strict parsing and RLFF post-processing; it excludes prompt/trajectory text.
- `reward-detail-samples.jsonl`: full protocol payloads and raw reward-model
  responses for a stable sample of roughly 25 trajectories in the initial
  200-sample, two-epoch run.
- `reward-failures.jsonl`: every terminal reward failure after retries are
  exhausted, including all attempt requests, thinking, replies, raw responses,
  and per-attempt validation/transport errors. This file is not sampled.
- `training-metrics.jsonl`: AReaL's native `actor_loss` and `clip_ratio`, exposed
  as `policy_loss` and `clip_fraction` without recomputation.
