#!/bin/bash
LOGDIR="/home/user/logs/llama"
APIKEY="${LLAMA_API_KEY:?请设置 LLAMA_API_KEY 环境变量（随机密钥，勿硬编码入库）}"
PRESET="/home/user/models/llamacpp/qwen3.8-27b/router-preset.ini"
MODELDIR="/home/user/models/llamacpp/qwen3.8-27b/model"
ALIAS="${LLAMA_TEST_ALIAS:-${1:-qwen-llm}}"

LOGFILE="router.log"

BINDIR="/home/user/llama.cpp/build-hip714-v0.9.0/bin"
ROUTER="$BINDIR/llama-server"
export LD_LIBRARY_PATH="/opt/rocm/core-7.14/lib:$BINDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROCBLAS_USE_HIPBLASLT=1
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0

mkdir -p "$LOGDIR"
exec "$ROUTER" \
  --host 0.0.0.0 \
  --port 8080 \
  --api-key "$APIKEY" \
  -a "$ALIAS" \
  --models-dir "$MODELDIR" \
  --models-max 1 \
  --models-preset "$PRESET" \
  --metrics \
  >> "$LOGDIR/$LOGFILE" 2>&1
