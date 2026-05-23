#!/bin/bash
# =============================================================================
# Run Mamba: sequenced + shuffled
# =============================================================================
# Run on Pod. Assumes setup_mamba_workspace.sh already ran on this volume.
#
# Usage: bash run_mamba.sh 2>&1 | tee mamba-session.log
# Estimated time: ~2-3 hours on A6000 (sequential scan is slower than flash attention)
# =============================================================================

set -e

TRAIN_DIR="/workspace/classical-training"
RESULTS_DIR="$TRAIN_DIR/results/mamba"

echo "============================================"
echo "Mamba: Sequenced + Shuffled"
echo "============================================"
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

cd "$TRAIN_DIR/mamba"

# Verify setup
if [ ! -f "data/classical_sequenced/train.bin" ]; then
    echo "ERROR: workspace not set up. Run setup_mamba_workspace.sh first."
    exit 1
fi

# Set batch size based on VRAM
# Mamba 97M at block_size=2048: lighter memory than GPT (no attention matrix)
# but sequential scan has different memory profile
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "GPU VRAM: ${VRAM_MB}MB"
if [ "$VRAM_MB" -ge 24000 ]; then
    BATCH=8; ACCUM=4
elif [ "$VRAM_MB" -ge 16000 ]; then
    BATCH=4; ACCUM=8
else
    BATCH=2; ACCUM=16
fi
echo "Batch size: $BATCH, grad accumulation: $ACCUM (effective batch: $((BATCH * ACCUM)))"
echo "Effective tokens/step: $((BATCH * ACCUM * 2048))"
echo ""

mkdir -p "$RESULTS_DIR/sequenced"
mkdir -p "$RESULTS_DIR/shuffled"

# ---- Sequenced ----
echo ""
echo "============================================"
echo "[1/2] Mamba, Sequenced"
echo "Start: $(date)"
echo "============================================"

PYTHONUNBUFFERED=1 python train_mamba.py \
    --out_dir="$RESULTS_DIR/sequenced" \
    --dataset=classical_sequenced \
    --sequential_data=True \
    --batch_size=$BATCH \
    --gradient_accumulation_steps=$ACCUM \
    --d_model=768 --n_layer=24 --d_state=16 --d_conv=4 --expand=2 \
    --eval_interval=250 --eval_iters=100 --log_interval=50 \
    --always_save_checkpoint=True --wandb_log=False \
    --dropout=0.0 --bias=True --init_from=scratch \
    --max_iters=5000 --learning_rate=3e-4 --weight_decay=0.1 \
    --beta1=0.9 --beta2=0.95 --grad_clip=1.0 \
    --decay_lr=True --warmup_iters=200 --lr_decay_iters=5000 --min_lr=3e-5 \
    --device=cuda --dtype=bfloat16 --compile=False \
    --wandb_run_name=mamba-sequenced \
    2>&1 | tee "$RESULTS_DIR/sequenced-training.log" || true

echo "  Completed: $(date)"

# ---- Shuffled ----
echo ""
echo "============================================"
echo "[2/2] Mamba, Shuffled"
echo "Start: $(date)"
echo "============================================"

PYTHONUNBUFFERED=1 python train_mamba.py \
    --out_dir="$RESULTS_DIR/shuffled" \
    --dataset=classical_shuffled \
    --sequential_data=False \
    --batch_size=$BATCH \
    --gradient_accumulation_steps=$ACCUM \
    --d_model=768 --n_layer=24 --d_state=16 --d_conv=4 --expand=2 \
    --eval_interval=250 --eval_iters=100 --log_interval=50 \
    --always_save_checkpoint=True --wandb_log=False \
    --dropout=0.0 --bias=True --init_from=scratch \
    --max_iters=5000 --learning_rate=3e-4 --weight_decay=0.1 \
    --beta1=0.9 --beta2=0.95 --grad_clip=1.0 \
    --decay_lr=True --warmup_iters=200 --lr_decay_iters=5000 --min_lr=3e-5 \
    --device=cuda --dtype=bfloat16 --compile=False \
    --wandb_run_name=mamba-shuffled \
    2>&1 | tee "$RESULTS_DIR/shuffled-training.log" || true

echo "  Completed: $(date)"

cd "$TRAIN_DIR"

# ---- Summary ----
echo ""
echo "============================================"
echo "MAMBA RUNS COMPLETE"
echo "============================================"
echo "End: $(date)"
echo ""
echo "Sequenced: $(grep '^step 5000' "$RESULTS_DIR/sequenced-training.log" 2>/dev/null || echo 'check log')"
echo "Shuffled:  $(grep '^step 5000' "$RESULTS_DIR/shuffled-training.log" 2>/dev/null || echo 'check log')"
echo ""
echo "Checkpoints:"
ls -lh "$RESULTS_DIR"/*/ckpt.pt 2>/dev/null
echo ""
echo "Checksums:"
md5sum "$RESULTS_DIR"/*/ckpt.pt 2>/dev/null
