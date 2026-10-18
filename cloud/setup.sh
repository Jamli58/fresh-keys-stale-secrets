#!/usr/bin/env bash
# One-time AutoDL setup: uv, Python 3.12, a vLLM venv and the project venv (kept separate:
# vLLM pins its own torch/numpy, numpriv needs numpy >= 2.5). Safe to re-run.
#   bash cloud/setup.sh            # VLLM_VERSION=0.x.y bash cloud/setup.sh to pin vLLM
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

# Academic proxy for GitHub (uv's Python downloads); ignored if absent.
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi

log "installing uv and modelscope"
python3 -m pip install -q -U uv modelscope
mkdir -p "$VENVS" "$LOGS"
uv python install 3.12

log "vLLM venv ($VENVS/vllm)"
[ -x "$VENVS/vllm/bin/python" ] || uv venv "$VENVS/vllm" --python 3.12
uv pip install --python "$VENVS/vllm/bin/python" "vllm${VLLM_VERSION:+==$VLLM_VERSION}"

log "project venv ($PROJECT/.venv)"
cd "$PROJECT"
[ -x .venv/bin/python ] || uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e . pytest

log "sanity checks"
.venv/bin/python -m pytest -q
"$VENVS/vllm/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || log "no GPU visible (no-GPU mode?) - fine for setup and downloads"
log "setup done"
