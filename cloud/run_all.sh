#!/usr/bin/env bash
# Both attacker models in sequence (one GPU, one vLLM server at a time). Survives SSH disconnects:
#   nohup bash cloud/run_all.sh dev > /root/autodl-tmp/run_all_dev.log 2>&1 &
#   CONFIRM_TEST=1 SHUTDOWN_AFTER=1 nohup bash cloud/run_all.sh test > /root/autodl-tmp/run_all_test.log 2>&1 &
# SHUTDOWN_AFTER=1 powers the instance off at the end (AutoDL stops billing; the data disk is kept).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
mode="${1:?usage: run_all.sh <dev|test> [run_m2.py flags]}"
shift

status=0
for key in llama8b qwen32b; do
  if ! bash "$HERE/run_m2.sh" "$key" "$mode" "$@"; then
    log "FAILED: $key $mode (continuing with the next model)"
    status=1
  fi
done
log "all models finished with status $status"

if [ "${SHUTDOWN_AFTER:-0}" = "1" ]; then
  log "shutting down in 60 s (SHUTDOWN_AFTER=1)"
  sleep 60
  shutdown -h now
fi
exit "$status"
