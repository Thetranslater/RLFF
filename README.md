# RLFF Phase-D cloud bundle

This directory is the self-contained transfer root for the first RLFF training run.
Run every command from this directory so all relative paths in `configs/` resolve
consistently.

## Directory contract

```text
rlff_phase_d/
  configs/
    rlff.yaml                 RLFF-owned configuration
    areal.yaml                pinned AReaL v1.0.4 native configuration
  data/
    episodes.jsonl            exactly 200 canonical episodes
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
2. Export `DEEPSEEK_API_KEY` in the shell used to start training. `.env.example` is a
   reference template; the current CLI does not automatically source `.env` files.
3. Install this bundle into the already prepared AReaL environment:

   ```bash
   python -m pip install -e .
   ```

4. From this directory, validate all paths and adapter metadata:

   ```bash
   rlff validate --config configs/rlff.yaml
   ```

5. After the rollout/reward/update smoke tests are added and pass, training will use:

   ```bash
   rlff train --config configs/rlff.yaml
   ```

`src/script/build/sync_cloud_bundle.py` in the main repository refreshes copied code,
prompts, packaging files, tests, and the 200-row dataset without touching model weights
or cloud outputs.
