#!/bin/bash
# =============================================================================
# Run 1B parameter sequenced training
# =============================================================================
# 20L/16H/2048E (~1.03B params) on the same 20.7M token classical curriculum.
# LR 2e-4 following GPT-3 scaling (Table 2.1, Brown et al. 2020).
# Cosine decay over 20,000 steps. Watch the generalization gap to decide
# when to stop — val loss minus train loss widening is the signal.
#
# Usage: bash run_1b_sequenced.sh 2>&1 | tee 1b-session.log
# =============================================================================

set -e

TRAIN_DIR="/workspace/classical-training"
RESULTS_DIR="$TRAIN_DIR/results/1b-sequenced"

echo "============================================"
echo "1B Sequenced Training"
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

# Set batch size based on VRAM — conservative for 1B model
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "GPU VRAM: ${VRAM_MB}MB"
if [ "$VRAM_MB" -ge 40000 ]; then
    BATCH=8; ACCUM=4
elif [ "$VRAM_MB" -ge 24000 ]; then
    BATCH=4; ACCUM=8
else
    BATCH=2; ACCUM=16
fi
echo "Batch size: $BATCH, grad accumulation: $ACCUM (effective batch: $((BATCH * ACCUM)))"
echo ""

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "1B GPT-2, Sequenced, gap-guided stopping"
echo "Config: 20L, 16H, 2048E (~1.03B params)"
echo "LR: 2e-4 (GPT-3 scaling)"
echo "Start: $(date)"
echo "============================================"

PYTHONUNBUFFERED=1 python train.py \
    --out_dir="$RESULTS_DIR" \
    --dataset=classical_sequenced \
    --sequential_data=True \
    --use_attnres=False \
    --batch_size=$BATCH --gradient_accumulation_steps=$ACCUM \
    --seed_offset=0 \
    --eval_interval=250 --eval_iters=100 --log_interval=50 \
    --always_save_checkpoint=True --wandb_log=False \
    --block_size=2048 --n_layer=20 --n_head=16 --n_embd=2048 \
    --dropout=0.0 --bias=True --init_from=scratch \
    --max_iters=20000 --learning_rate=2e-4 --weight_decay=0.1 \
    --beta1=0.9 --beta2=0.95 --grad_clip=1.0 \
    --decay_lr=True --warmup_iters=500 --lr_decay_iters=20000 \
    --min_lr=2e-5 --device=cuda --dtype=bfloat16 --compile=False \
    2>&1 | tee "$RESULTS_DIR/training.log" || true

echo "  Completed: $(date)"

cd "$TRAIN_DIR"

echo ""
echo "============================================"
echo "1B SEQUENCED RUN COMPLETE"
echo "============================================"
echo "End: $(date)"
echo ""
echo "Key checkpoints:"
for step in 2500 5000 7500 10000 15000 20000; do
    echo "Step $step: $(grep "^step $step" "$RESULTS_DIR/training.log" 2>/dev/null || echo 'not reached')"
done
echo ""
echo "Checkpoint:"
ls -lh "$RESULTS_DIR"/ckpt.pt 2>/dev/null
echo ""
echo "Checksum:"
md5sum "$RESULTS_DIR"/ckpt.pt 2>/dev/null
