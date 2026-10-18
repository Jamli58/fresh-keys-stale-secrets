#!/usr/bin/env bash
# Stop the vLLM server started by serve.sh (no-op if none is running).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

[ -f "$VLLM_PID_FILE" ] || exit 0
pid="$(cat "$VLLM_PID_FILE")"
if kill -0 "$pid" 2>/dev/null; then
  log "stopping vLLM (pid $pid)"
  kill "$pid"
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  if kill -0 "$pid" 2>/dev/null; then
    log "vLLM did not exit after 60 s; sending SIGKILL"
    kill -9 "$pid"
  fi
fi
rm -f "$VLLM_PID_FILE"
