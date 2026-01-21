#!/usr/bin/env bash
set -euo pipefail

OPENAI_KEY="${OPENAI_KEY:-${OPENAI_API_KEY:-}}"
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"
SEEDS="${SEEDS:-0,1,2,3}"
PROJECT="${PROJECT:-cs5880-tictactoe}"
ENTITY="${ENTITY:-am893120}"
GROUP="${GROUP:-ppo-tictactoe}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
N_ENVS="${N_ENVS:-8}"

if [[ -z "$OPENAI_KEY" ]]; then
  read -r -p "Enter OPENAI_API_KEY (blank to skip OpenAI): " OPENAI_KEY || true
fi
if [[ -z "$WANDB_KEY" ]]; then
  read -r -p "Enter WANDB_API_KEY (blank for offline mode): " WANDB_KEY || true
fi

export OPENAI_API_KEY="$OPENAI_KEY"
export WANDB_API_KEY="$WANDB_KEY"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v py >/dev/null 2>&1; then
    PYTHON_BIN="py -3"
  else
    echo "Python not found. Install python3 in WSL or set PYTHON_BIN=/path/to/python." >&2
    exit 1
  fi
fi

IFS=',' read -ra SEED_LIST <<< "$SEEDS"
if [[ ${#SEED_LIST[@]} -eq 0 ]]; then
  echo "No seeds provided. Set SEEDS env var like '0,1,2,3'." >&2
  exit 1
fi

for seed in "${SEED_LIST[@]}"; do
  seed_trimmed="$(echo "$seed" | xargs)"
  if [[ -z "$seed_trimmed" ]]; then
    continue
  fi
  echo "=== Running seed $seed_trimmed ==="
  $PYTHON_BIN train_tictactoe_wandb.py \
    --project "$PROJECT" \
    --entity "$ENTITY" \
    --group "$GROUP" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --n-envs "$N_ENVS" \
    --seed "$seed_trimmed"
done
