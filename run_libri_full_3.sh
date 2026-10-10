#!/usr/bin/env bash
set -euo pipefail
ROOT=/project/anhlt/0607/research/work/ghep_code/Mamba-TasNet
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export TMPDIR="$ROOT"
export XDG_CACHE_HOME="$ROOT/.runtime/cache" TRITON_CACHE_DIR="$ROOT/.runtime/triton" CUDA_CACHE_PATH="$ROOT/.runtime/cuda"
export HF_HOME="$ROOT/.runtime/huggingface" TORCH_HOME="$ROOT/.runtime/torch" TORCH_EXTENSIONS_DIR="$ROOT/.runtime/extensions"
export MPLCONFIGDIR="$ROOT/.runtime/matplotlib" NUMBA_CACHE_DIR="$ROOT/.runtime/numba"
export WANDB_DIR="$ROOT/.runtime/wandb" WANDB_CACHE_DIR="$ROOT/.runtime/wandb-cache" WANDB_CONFIG_DIR="$ROOT/.runtime/wandb-config" WANDB_DATA_DIR="$ROOT/.runtime/wandb-data" WANDB_MODE=disabled
LOG="$ROOT/train_libri_gpu012_full_$(date +%Y%m%d_%H%M%S).log"
printf '[LAUNCH] GPU=0,1,2; limit_training_signal_len=False; log=%s\n' "$LOG"
/home/anhlt/.conda/envs/mamba_tasnet_v1/bin/python -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=3 \
    train_wsj0mix.py hparams/LibriMix/dpmamba_XS_jepa.yaml \
    --distributed_backend nccl --device cuda:0 \
    --data_folder /project/anhlt/LibriMix/Libri2Mix \
    --output_folder "$ROOT/results/LibriMix/dpmamba_XS_jepa_gpu012/1234" \
    --precision bf16 --limit_training_signal_len False \
    --librimix_validate_audio False --skip_prep False 2>&1 | tee "$LOG"
