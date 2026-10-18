#!/usr/bin/env bash
# M2 for one attacker model: start vLLM, smoke test (R031 on the cloud backend), then the attack matrix.
#   bash cloud/run_m2.sh llama8b dev                  # dev split: prompt checks, failure rates, throughput
#   CONFIRM_TEST=1 bash cloud/run_m2.sh llama8b test  # test split, 3 defense seeds (R011/R012 + R014), touched ONCE
# Extra run_m2.py flags can follow, e.g.  bash cloud/run_m2.sh qwen32b dev --max-firms 10 --attackers A4
# KEEP_SERVER=1 leaves vLLM running afterwards.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
key="${1:?usage: run_m2.sh <llama8b|qwen32b> <dev|test> [run_m2.py flags]}"
mode="${2:?usage: run_m2.sh <llama8b|qwen32b> <dev|test> [run_m2.py flags]}"
shift 2
model_info "$key"
cd "$PROJECT"
mkdir -p "$LOGS"

case "$mode" in
  dev)
    split_args=(--split dev --defense-seeds 0)
    run="${RUN_ID}_dev_$key"
    ;;
  test)
    if [ "${CONFIRM_TEST:-0}" != "1" ]; then
      echo "the test split is touched once: set CONFIRM_TEST=1 after the dev run has been reviewed" >&2
      exit 1
    fi
    split_args=(--split test --confirm-test --defense-seeds 0 1 2)
    run="${RUN_ID}_test_$key"
    ;;
  *) echo "mode must be dev or test, got '$mode'" >&2; exit 1 ;;
esac

if [ "${KEEP_SERVER:-0}" != "1" ]; then trap 'bash "$HERE/stop.sh"' EXIT; fi
if ! curl -sf "$BASE_URL/models" | grep -q "\"$SERVED\""; then
  bash "$HERE/serve.sh" "$key"
fi

smoke="LLM_SMOKE_autodl_$key.json"
if [ ! -f "results/$smoke" ]; then
  log "R031 smoke test on the cloud backend -> results/$smoke"
  .venv/bin/python scripts/llm_smoke.py --backend openai --base-url "$BASE_URL" --models "$SERVED" \
    --workers "$WORKERS" --out "$smoke" 2>&1 | tee "$LOGS/smoke_$key.log"
fi

log "M2 $mode run $run ($SERVED, $WORKERS concurrent requests)"
.venv/bin/python scripts/run_m2.py --model "$SERVED" --base-url "$BASE_URL" --run "$run" \
  --workers "$WORKERS" "${split_args[@]}" "$@" 2>&1 | grep --line-buffered -v "HTTP Request" | tee "$LOGS/$run.log"
log "done: results/$run.json, results/${run}_rows.jsonl"
