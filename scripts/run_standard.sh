#!/bin/bash
# =============================================================================
# Run standard nanoGPT: sequenced + shuffled
# =============================================================================
# Run on Pod 1. Assumes setup_workspace.sh already ran on this volume.
#
# Usage: bash run_standard.sh 2>&1 | tee standard-session.log
# Estimated time: ~1.5 hours on A6000
# =============================================================================

set -e

TRAIN_DIR="/workspace/classical-training"
RESULTS_DIR="$TRAIN_DIR/results/standard"

echo "============================================"
echo "Standard nanoGPT: Sequenced + Shuffled"
echo "============================================"
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

cd "$TRAIN_DIR"

# Verify setup
if [ ! -f "nanoGPT/data/classical_sequenced/train.bin" ]; then
    echo "ERROR: workspace not set up. Run setup_workspace.sh first."
    exit 1
fi

cd nanoGPT

# Set batch size based on VRAM
# Standard nanoGPT 91M at block_size=2048: batch 32 fits ~24GB
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "GPU VRAM: ${VRAM_MB}MB"
if [ "$VRAM_MB" -ge 24000 ]; then
    BATCH=32; ACCUM=1
elif [ "$VRAM_MB" -ge 16000 ]; then
    BATCH=16; ACCUM=2
else
    BATCH=8; ACCUM=4
fi
echo "Batch size: $BATCH, grad accumulation: $ACCUM (effective batch: $((BATCH * ACCUM)))"
echo ""

BASE_ARGS="--eval_interval=250 --eval_iters=100 --log_interval=50 --always_save_checkpoint=True --wandb_log=False --block_size=2048 --n_layer=12 --n_head=12 --n_embd=768 --dropout=0.0 --bias=True --init_from=scratch --max_iters=5000 --learning_rate=3e-4 --weight_decay=0.1 --beta1=0.9 --beta2=0.95 --grad_clip=1.0 --decay_lr=True --warmup_iters=200 --lr_decay_iters=5000 --min_lr=3e-5 --device=cuda --dtype=bfloat16 --compile=False"

mkdir -p "$RESULTS_DIR/sequenced"
mkdir -p "$RESULTS_DIR/shuffled"

# ---- Sequenced ----
echo ""
echo "============================================"
echo "[1/2] Standard GPT-2, Sequenced"
echo "Start: $(date)"
echo "============================================"

PYTHONUNBUFFERED=1 python train.py \
    --out_dir="$RESULTS_DIR/sequenced" \
    --dataset=classical_sequenced \
    --sequential_data=True \
    --use_attnres=False \
    --batch_size=$BATCH --gradient_accumulation_steps=$ACCUM \
    --seed_offset=0 \
    $BASE_ARGS \
    2>&1 | tee "$RESULTS_DIR/sequenced-training.log" || true

echo "  Completed: $(date)"

# ---- Shuffled ----
echo ""
echo "============================================"
echo "[2/2] Standard GPT-2, Shuffled"
echo "Start: $(date)"
echo "============================================"

PYTHONUNBUFFERED=1 python train.py \
    --out_dir="$RESULTS_DIR/shuffled" \
    --dataset=classical_shuffled \
    --sequential_data=False \
    --use_attnres=False \
    --batch_size=$BATCH --gradient_accumulation_steps=$ACCUM \
    --seed_offset=0 \
    $BASE_ARGS \
    2>&1 | tee "$RESULTS_DIR/shuffled-training.log" || true

echo "  Completed: $(date)"

cd "$TRAIN_DIR"

# ---- Summary ----
echo ""
echo "============================================"
echo "STANDARD RUNS COMPLETE"
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
