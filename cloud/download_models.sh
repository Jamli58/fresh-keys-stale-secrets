#!/usr/bin/env bash
# Download attacker models from ModelScope and verify every file against the published sha256.
# Tip: run this in AutoDL's no-GPU mode (无卡模式) - downloads are slow and GPU time is not needed.
#   bash cloud/download_models.sh              # both models
#   bash cloud/download_models.sh qwen32b      # one model
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

keys=("$@")
[ ${#keys[@]} -gt 0 ] || keys=(llama8b qwen32b)
mkdir -p "$MODELS" "$PROJECT/results"

for key in "${keys[@]}"; do
  model_info "$key"
  log "downloading $MS_ID -> $MODEL_DIR"
  exclude_args=()
  [ -z "$MS_EXCLUDE" ] || exclude_args=(--exclude "$MS_EXCLUDE")
  modelscope download --model "$MS_ID" --local_dir "$MODEL_DIR" "${exclude_args[@]}"
  log "verifying $key (sha256 of every weight/config/tokenizer file)"
  "$PROJECT/.venv/bin/python" "$PROJECT/scripts/model_provenance.py" "$MS_ID" "$MODEL_DIR" \
    "$PROJECT/results/model_provenance_$key.json" ${MS_EXCLUDE:+--exclude "$MS_EXCLUDE"}
done
df -h "$WORK"
log "downloads verified"
