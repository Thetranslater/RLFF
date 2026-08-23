---
library_name: peft
license: other
base_model: /root/autodl-tmp/Qwen2.5-7B-Instruct
tags:
- base_model:adapter:/root/autodl-tmp/Qwen2.5-7B-Instruct
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: train_2026-08-23-11-12-31
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# train_2026-08-23-11-12-31

This model is a fine-tuned version of [/root/autodl-tmp/Qwen2.5-7B-Instruct](https://huggingface.co//root/autodl-tmp/Qwen2.5-7B-Instruct) on the rlff_roleplay_v2_all dataset.
It achieves the following results on the evaluation set:
- Loss: 1.7568
- Num Input Tokens Seen: 4270544

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-06
- train_batch_size: 2
- eval_batch_size: 2
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- num_epochs: 2.0

### Training results



### Framework versions

- PEFT 0.18.1
- Transformers 5.5.0
- Pytorch 2.11.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2