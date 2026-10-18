# Shared settings for the AutoDL scripts. Sourced, not executed.
# Everything large lives on the data disk (/root/autodl-tmp); the 30 GB system disk is too small.

export WORK="${WORK:-/root/autodl-tmp}"
export PROJECT="${PROJECT:-$WORK/numpriv/pilot}"
export MODELS="$WORK/models"
export VENVS="$WORK/venvs"
export LOGS="$PROJECT/results/logs"
export VLLM_PORT="${VLLM_PORT:-8000}"
export BASE_URL="http://127.0.0.1:$VLLM_PORT/v1"
export VLLM_PID_FILE="$WORK/vllm.pid"
export UV_INDEX_URL="${UV_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
export UV_CACHE_DIR="$WORK/.uv-cache"
export MODELSCOPE_CACHE="$WORK/.modelscope"
# AutoDL's academic proxy (network_turbo) must never intercept requests to the local vLLM server.
export no_proxy="localhost,127.0.0.1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"
# FlashInfer's sampler JIT-compiles kernels and needs nvcc >= 12.9 on SM 12.x (RTX 6000D / Blackwell);
# AutoDL images ship CUDA 12.8. All requests are greedy (temperature 0), so the PyTorch sampler is equivalent.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# GPU memory in MiB (first GPU; 0 in no-GPU mode). Override with GPU_MEM_MIB=... if needed.
# A GPU with >= BIG_GPU_MIB (e.g. 84 GB RTX 6000D, 80 GB A100/H100) gets more concurrency;
# only server capacity changes, never the requests (prompts, max_tokens, temperature).
BIG_GPU_MIB=70000
gpu_mem_mib() {
  if [ -n "${GPU_MEM_MIB:-}" ]; then echo "$GPU_MEM_MIB"; return; fi
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true
}

# model_info <key>: sets MS_ID, MODEL_DIR, SERVED, VLLM_ARGS, WORKERS, RUN_ID, MS_EXCLUDE.
# Served names are distinct from the local Ollama tags, so the response caches never mix.
model_info() {
  local mem big=0
  mem="$(gpu_mem_mib)"
  if [ "${mem:-0}" -ge "$BIG_GPU_MIB" ] 2>/dev/null; then big=1; fi
  case "$1" in
    llama8b)
      MS_ID="LLM-Research/Meta-Llama-3.1-8B-Instruct"
      SERVED="llama3.1-8b-instruct-bf16"
      if [ "$big" = 1 ]; then
        VLLM_ARGS="--dtype bfloat16 --max-model-len 8192 --max-num-seqs 256"
        WORKERS=128
      else
        VLLM_ARGS="--dtype bfloat16 --max-model-len 8192 --max-num-seqs 64"
        WORKERS=48
      fi
      RUN_ID="R011"
      MS_EXCLUDE="original/*"  # 16 GB duplicate of the weights in Meta's own format
      ;;
    qwen32b)
      MS_ID="Qwen/Qwen2.5-32B-Instruct-AWQ"
      SERVED="qwen2.5-32b-instruct-awq"
      if [ "$big" = 1 ]; then
        VLLM_ARGS="--max-model-len 8192 --max-num-seqs 64"
        WORKERS=64
      else
        VLLM_ARGS="--max-model-len 4096 --max-num-seqs 16"
        WORKERS=16
      fi
      RUN_ID="R012"
      MS_EXCLUDE=""
      ;;
    *)
      echo "unknown model key '$1' (expected llama8b or qwen32b)" >&2
      return 1
      ;;
  esac
  MODEL_DIR="$MODELS/$1"
  export MS_ID MODEL_DIR SERVED VLLM_ARGS WORKERS RUN_ID MS_EXCLUDE
}

log() { echo "[$(date '+%F %T')] $*"; }
