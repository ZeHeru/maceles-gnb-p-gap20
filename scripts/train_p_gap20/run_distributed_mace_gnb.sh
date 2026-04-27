#!/usr/bin/env bash
set -euo pipefail

# Distributed training entry point for the P-GAP-20 phosphorus benchmark.
# Override PROJECT_ROOT, PYTHON_BIN, TRAIN_FILE, and the MLP_* variables as needed.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_FILE="${TRAIN_FILE:-${P_GAP20_TRAIN_FILE:-/home/yuzhu/workspace/data/P_GAP_20_fitting_data.xyz}}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node="${MLP_WORKER_GPU:-1}" \
  --nnodes="${MLP_WORKER_NUM:-1}" \
  --node_rank="${MLP_ROLE_INDEX:-0}" \
  --master_addr="${MLP_WORKER_0_HOST:-127.0.0.1}" \
  --master_port="${MLP_WORKER_0_PORT:-29500}" \
  "$PROJECT_ROOT/scripts/run_train.py" \
  --name="P_MACE_gnb.model" \
  --train_file="$TRAIN_FILE" \
  --valid_fraction=0.02 \
  --E0s="{15: -0.09753304}" \
  --energy_key="energy" \
  --forces_key="forces" \
  --model="MACE" \
  --num_interactions=2 \
  --num_channels=192 \
  --max_L=1 \
  --correlation=3 \
  --r_max=4.5 \
  --forces_weight=1000 \
  --energy_weight=40 \
  --weight_decay=5e-10 \
  --clip_grad=1.0 \
  --batch_size=32 \
  --valid_batch_size=32 \
  --max_num_epochs=500 \
  --scheduler_patience=40 \
  --patience=99999 \
  --eval_interval=1 \
  --ema \
  --swa \
  --start_swa=400 \
  --swa_lr=0.00025 \
  --swa_forces_weight=10 \
  --num_workers=16 \
  --error_table="PerAtomMAE" \
  --default_dtype="float32" \
  --device="${DEVICE:-cuda}" \
  --seed=123 \
  --distributed \
  --launcher=torchrun \
  --save_cpu \
  --use_gnb \
  --cutoff_lr=12.0 \
  --restart_latest
