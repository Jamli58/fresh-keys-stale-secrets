#!/usr/bin/env bash
# Run on the Mac. Push the pilot code to AutoDL, or pull the cloud results back.
#   export AUTODL_HOST=connect.xxx.seetacloud.com AUTODL_PORT=12345   # from the instance's SSH command
#   bash cloud/sync.sh push
#   bash cloud/sync.sh pull      # -> results/autodl/ (never overwrites local results)
# Needs the instance's SSH password or key (AutoDL console -> 登录指令).
set -euo pipefail
action="${1:?usage: sync.sh <push|pull>}"
: "${AUTODL_HOST:?set AUTODL_HOST (e.g. connect.westb.seetacloud.com)}"
: "${AUTODL_PORT:?set AUTODL_PORT}"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="/root/autodl-tmp/numpriv/pilot"
SSH="ssh -p $AUTODL_PORT"

case "$action" in
  push)
    $SSH "root@$AUTODL_HOST" "mkdir -p $REMOTE"
    # The frozen M1 config (results/R008_tuning.json) is pushed with the other results; caches, venvs and
    # anything pulled back from the cloud stay local.
    rsync -avz --delete -e "$SSH" \
      --exclude '.venv' --exclude '.venv.nosync/' --exclude '.venv 2/' --exclude '__pycache__/' \
      --exclude '.pytest_cache/' --exclude '.coverage' --exclude '.git/' --exclude '.DS_Store' \
      --exclude 'results/autodl/' --exclude 'results/llm_cache/' --exclude 'results/logs/' \
      --exclude 'results/R01[1-4]_*' --exclude 'results/LLM_SMOKE_autodl_*' \
      --exclude 'results/model_provenance_*' --exclude 'results/vllm_env_*' \
      "$LOCAL/" "root@$AUTODL_HOST:$REMOTE/"
    ;;
  pull)
    mkdir -p "$LOCAL/results/autodl"
    rsync -avz -e "$SSH" "root@$AUTODL_HOST:$REMOTE/results/" "$LOCAL/results/autodl/"
    ;;
  *) echo "action must be push or pull" >&2; exit 1 ;;
esac
