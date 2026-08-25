#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BUNDLE_ROOT}"

PORT="${SGLANG_PORT:-30000}"
MEM_FRACTION="${SGLANG_MEM_FRACTION_STATIC:-0.80}"

exec python -m sglang.launch_server \
  --model-path models/base/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --dtype bfloat16 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --context-length 5120 \
  --mem-fraction-static "${MEM_FRACTION}" \
  --lora-paths rlff-sft=models/adapters/qwen2.5-7b-instruct-sft \
  --max-loras-per-batch 2
