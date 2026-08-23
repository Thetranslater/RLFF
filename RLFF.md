# RLFF：使用 TRL GRPO 继续训练 SFT LoRA

当前实现位于 `src/rlff/train.py`。它不会创建新的 LoRA adapter，而是：

1. 以 bitsandbytes 8-bit 方式加载基础模型；
2. 以 `is_trainable=True` 加载已经完成 SFT 的 LoRA adapter；
3. 使用 TRL `GRPOTrainer` 继续更新同一个 adapter；
4. 每个 prompt 生成 4 个候选回答；
5. 通过 `src/LLM` 中的 Python DeepSeek 接口调用奖励模型；
6. 当前忽略 DeepSeek 返回内容，并为所有候选固定返回奖励 `0.0`。

固定奖励只能验证训练链路。由于同组候选奖励完全相同，GRPO 的相对优势为零，当前版本
基本不会产生有效的参数更新。

## 模型和 adapter

`--model-name-or-path` 指向 SFT 时使用的原始基础模型，例如 Qwen3-8B、
Qwen2.5-7B-Instruct 或 Llama Instruct。

`--adapter-name-or-path` 指向 SFT 输出的 LoRA adapter 目录，其中至少应包含：

```text
adapter_config.json
adapter_model.safetensors
```

RL 阶段继续训练原 adapter，因此 LoRA 的 `rank` 和 `alpha` 由
`adapter_config.json` 决定，无法在 GRPO 阶段改成另一组值。脚本会打印实际配置；如果不是
最初计划的 `rank=64、alpha=128`，会输出警告但允许继续运行。

## 云端环境

建议使用 Python 3.11 和 NVIDIA CUDA 环境。先依据云服务器的 CUDA/驱动安装匹配版本的
PyTorch，再安装项目训练依赖：

```bash
conda create -n rlff-trl python=3.11 -y
conda activate rlff-trl

# 按 PyTorch 官网给出的 CUDA 命令安装 torch，然后：
pip install -e '.[train]'
```

8-bit bitsandbytes 训练需要 NVIDIA CUDA；本地 AMD 环境只能执行数据 dry-run，不能进行
实际 GRPO 训练。

复制环境变量文件并填写 DeepSeek API key：

```bash
cp .env.example .env
```

主要配置为：

```dotenv
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
RLFF_REWARD_MODEL=deepseek-v4-pro
RLFF_REWARD_CONCURRENCY=4
RLFF_REWARD_TIMEOUT_MS=120000
RLFF_REWARD_RETRIES=2
RLFF_REWARD_MAX_TOKENS=1024
```

如果设置了 `LANGSMITH_TRACING=true`、`LANGSMITH_API_KEY` 和
`LANGSMITH_PROJECT`，每次 DeepSeek 奖励调用会通过独立 `langsmith` SDK 记录；不需要
LangChain 或 LangGraph。

## 数据格式

训练脚本直接读取 `dataset/rlff/<书名>/<书名>.json`：

```json
{
  "title": "书名",
  "plots": [
    {
      "plot_index": 0,
      "messages": [
        {
          "character": "Environment",
          "content": "旁白",
          "validation": []
        },
        {
          "character": "角色名",
          "content": "参考对话",
          "validation": [
            {
              "name": "Entity.attribute",
              "description": "状态含义",
              "value": true
            }
          ],
          "system": "角色提示词文件路径或空字符串"
        }
      ]
    }
  ]
}
```

每条非 `Environment` 消息都会形成一个 episode：

- 前面至多 `--history-window` 条剧情消息合并成一条 ShareGPT 风格的 `user` prompt；
- `system` 被解释为提示词文件路径，相对路径默认相对于数据集目录；
- 当前消息的 `validation` 作为 `locked_facts` 传给 DeepSeek；
- 当前 `content` 仅保留为 `reference_response`，不会放入模型输入；
- `Environment` 消息只进入历史，不作为生成目标。

可以用 `--system-prompt-file` 为全部样本统一覆盖 system prompt，或使用
`--system-root` 修改数据中相对路径的根目录。使用多个 `--target-character` 可以只训练指定
角色。

## 本地检查数据

`--dry-run` 不导入 PyTorch、TRL 或 bitsandbytes，也不会调用 DeepSeek：

```powershell
python -m rlff.train `
  "dataset/rlff/2龙与虎/2龙与虎.json" `
  --model-name-or-path "Qwen/Qwen2.5-7B-Instruct" `
  --adapter-name-or-path "saves/qwen-sft-adapter" `
  --target-character "逢坂大河" `
  --system-prompt-file "sft_system_v1.txt" `
  --max-samples 8 `
  --dry-run
```

## 云端训练示例

```bash
python -m rlff.train \
  'dataset/rlff/2龙与虎/2龙与虎.json' \
  --model-name-or-path '/root/autodl-tmp/Qwen2.5-7B-Instruct' \
  --adapter-name-or-path '/root/saves/qwen-sft-adapter' \
  --output-dir '/root/autodl-tmp/rlff-grpo-adapter' \
  --target-character '逢坂大河' \
  --system-prompt-file 'sft_system_v1.txt' \
  --max-prompt-length 3072 \
  --max-completion-length 512 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-5 \
  --num-train-epochs 1
```

默认在单卡上使用：

- 8-bit 基础模型；
- SFT adapter 原有 LoRA 配置；
- `num_generations=4`；
- batch size 1、梯度累积 4；
- BF16、TF32、gradient checkpointing；
- `paged_adamw_8bit`；
- 不启用 vLLM。

有效 generation batch size 必须能够被 4 整除。脚本会在加载模型前检查
`WORLD_SIZE × per_device_train_batch_size × gradient_accumulation_steps`。

输出目录保存的是经过 SFT 和 GRPO 连续训练后的同一个 LoRA adapter，推理时仍需同时提供
原始基础模型。
