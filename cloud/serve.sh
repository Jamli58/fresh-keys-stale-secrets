#!/usr/bin/env bash
# Start vLLM for one model in the background and wait until it answers.
#   bash cloud/serve.sh llama8b
# Stops any server started by these scripts first. Records the software/GPU environment
# to results/vllm_env_<key>.txt. Server log: results/logs/vllm_<key>.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
key="${1:?usage: serve.sh <llama8b|qwen32b>}"
model_info "$key"
[ -f "$MODEL_DIR/config.json" ] || { echo "model not downloaded: run cloud/download_models.sh $key" >&2; exit 1; }

bash "$HERE/stop.sh"
mkdir -p "$LOGS"
log "starting vLLM: $SERVED from $MODEL_DIR"
# --generation-config vllm: ignore the model repo's sampling defaults; requests set temperature 0 and a seed.
# shellcheck disable=SC2086  # VLLM_ARGS is a list of flags
nohup "$VENVS/vllm/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name "$SERVED" --host 127.0.0.1 --port "$VLLM_PORT" \
  --seed 0 --generation-config vllm --gpu-memory-utilization 0.92 $VLLM_ARGS \
  > "$LOGS/vllm_$key.log" 2>&1 &
echo $! > "$VLLM_PID_FILE"

for _ in $(seq 1 180); do  # up to 15 minutes (loading 20 GB + CUDA graph capture)
  if curl -sf "$BASE_URL/models" > /dev/null; then
    log "vLLM is up (pid $(cat "$VLLM_PID_FILE"))"
    {
      echo "key=$key served=$SERVED model_dir=$MODEL_DIR args=$VLLM_ARGS"
      echo "VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
      "$VENVS/vllm/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
      nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
      curl -s "$BASE_URL/models"; echo
    } > "$PROJECT/results/vllm_env_$key.txt"
    exit 0
  fi
  if ! kill -0 "$(cat "$VLLM_PID_FILE")" 2>/dev/null; then
    echo "vLLM exited during startup; last log lines:" >&2
    tail -n 40 "$LOGS/vllm_$key.log" >&2
    exit 1
  fi
  sleep 5
done
echo "vLLM did not come up within 15 minutes; see $LOGS/vllm_$key.log" >&2
exit 1
