# RLFF Phase-D 

This directory is the self-contained transfer root for the first RLFF training run.
Run every command from this directory so all relative paths in `configs/` resolve
consistently.

## Directory contract

```text
rlff_phase_d/
  configs/
    rlff.yaml                 RLFF-owned configuration
    areal.yaml                pinned AReaL v1.0.4 native configuration
    rlff_epoch2.yaml          RLFF config for resuming into epoch two
    areal_epoch2.yaml         native config with DAPO clipping for epoch two
  data/
    episodes.jsonl            exactly 200 canonical episodes
    episodes_epoch2.jsonl     same episodes, stable mixed curriculum order
  models/
    base/Qwen2.5-7B-Instruct/ Hugging Face BF16 base model
    adapters/qwen2.5-7b-instruct-sft/ PEFT LoRA adapter continued by RLFF
  prompts/
    sft_system_v1.txt
    sft_system_v2.txt
    sft_system_v3.txt
    completion_reward_system.txt
    trajectory_reward_system.txt
  src/                       RLFF runtime and prompt renderer
  tests/                     local RLFF tests and future cloud smoke tests
  outputs/                   checkpoints, AReaL state, and audit logs
  .env.example
  pyproject.toml
```

The AReaL checkout and CUDA environment remain external dependencies. The expected
versions are AReaL `1.0.4` at commit
`37d6c6400e99a05fa3409d6a067762a44df40d3b` and SGLang `0.5.10.post1`.

## Model files

`models/base/Qwen2.5-7B-Instruct` must be an ordinary Hugging Face Transformers
directory, not a GPTQ/AWQ/bitsandbytes export. It must contain `config.json`, tokenizer
files, and either `model.safetensors` or sharded `model-*.safetensors` plus
`model.safetensors.index.json`.

`models/adapters/qwen2.5-7b-instruct-sft` must be a PEFT LoRA directory containing:

- `adapter_config.json` with `peft_type=LORA`, `task_type=CAUSAL_LM`, and a
  `base_model_name_or_path` resolving to the base-model directory above;
- `adapter_model.safetensors` (preferred) or `adapter_model.bin`;
- the exact rank, alpha, and target modules configured in both YAML files.

The checked-in configuration currently expects rank 48, alpha 96, and
`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`. The runtime sets every
attached LoRA dropout module to zero before RL updates. If the BF16 SFT run uses another topology, update both
`configs/rlff.yaml` and `configs/areal.yaml` before validation. Do not merge the adapter
into the base model: RLFF continues training this same adapter.

## Prepare and validate

1. Copy the base model and adapter into the paths above.
2. Export `DASHSCOPE_API_KEY` in the shell used to start training. `.env.example` is a
   reference template; the current CLI does not automatically source `.env` files.
3. Install this bundle into the already prepared uv/AReaL environment. The
   AReaL virtual environment does not need to contain the `pip` Python module:

   ```bash
   uv pip install -e . --no-deps
   ```

4. From this directory, validate all paths and adapter metadata:

   ```bash
   rlff validate --config configs/rlff.yaml
   ```

5. After the rollout/reward/update smoke tests are added and pass, training will use:

   ```bash
   rlff train --config configs/rlff.yaml
   ```

## Checkpoints and recovery

`configs/areal.yaml` uses two independent schedules. A versioned Hugging Face
LoRA export is written every 20 completed optimizer steps and at every epoch
boundary. RLFF removes superseded versioned exports only after AReaL finishes
the synchronous save and recovery barriers, leaving exactly the newest LoRA
directory. The fixed `recover_checkpoint/` DCP state is refreshed every 10
steps and at every epoch boundary; it includes optimizer state and is used for
exact continuation.

Recovery mode is `auto`. To resume after an interruption, rerun the same
training command with the same `experiment_name`, `trial_name`, `fileroot`,
model/adapter topology, and dataset. To intentionally start a new experiment,
change `trial_name` first. Keep `outputs/areal` on persistent storage.

The `checkpoint` block in `configs/rlff.yaml` is RLFF-side output metadata; the
effective native save and recovery schedules are the `saver` and `recover`
blocks in `configs/areal.yaml`.

To prepare epoch two, generate the ordered copy from the bundle root:

```bash
python scripts/build_epoch2_dataset.py data/episodes.jsonl data/episodes_epoch2.jsonl
```

After the first epoch checkpoint is complete, resume with the epoch-two pair:

```bash
rlff train --config configs/rlff_epoch2.yaml
```

This keeps the same AReaL experiment, trial, and recovery fileroot, while
`areal_epoch2.yaml` raises `total_train_epochs` to 2 and enables DAPO-style
asymmetric clipping (`eps_clip=0.2`, `eps_clip_higher=0.28`). The ordered
dataset uses a stable mixed curriculum: the first 100 records contain 70 easy
and 30 normal/difficult episodes; the last 100 contain 30 easy and 70
normal/difficult episodes. Each half distributes both classes throughout the
sequence while preserving the original relative order inside each class. 

Use this configuration only from the exact first-epoch boundary (the recovery
record whose next global step is 200). AReaL restores the stateful dataloader
cursor together with model and optimizer state. Resuming this configuration
from a mid-epoch checkpoint would therefore enter `episodes_epoch2.jsonl` at
the restored cursor instead of beginning its easy-first order at record zero.

The epoch-two RLFF config also enables same-episode dynamic resampling. For
each attempt, all four trajectories are generated in independent proxy
sessions and only trajectory-role rewards are requested first. If every
character has the same trajectory reward across all four trajectories, the
entire group is discarded and the same episode is sampled again with no
attempt limit. Completion-role rewards are requested only after at least one
character has differing trajectory rewards. Rejected interactions therefore
never enter AReaL's training batch and do not consume completion-reward calls.
`src/script/build/sync_cloud_bundle.py` in the main repository refreshes copied code,
prompts, packaging files, tests, and the 200-row dataset without touching model weights
or cloud outputs.
