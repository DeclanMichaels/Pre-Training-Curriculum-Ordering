#!/bin/bash
# =============================================================================
# Run 1B parameter shuffled training
# =============================================================================
# Same architecture as 1B sequenced (20L/16H/2048E, ~1.03B params).
# Same LR scheme. Watch the gap to decide when to stop.
#
# Usage: bash run_1b_shuffled.sh 2>&1 | tee 1b-shuffled-session.log
# =============================================================================

set -e

TRAIN_DIR="/workspace/classical-training"
RESULTS_DIR="$TRAIN_DIR/results/1b-shuffled"

echo "============================================"
echo "1B Shuffled Training"
echo "============================================"
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

cd "$TRAIN_DIR"

if [ ! -f "nanoGPT/data/classical_shuffled/train.bin" ]; then
    echo "ERROR: shuffled data not set up."
    exit 1
fi

cd nanoGPT

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
echo "1B GPT-2, Shuffled, gap-guided stopping"
echo "Config: 20L, 16H, 2048E (~1.03B params)"
echo "LR: 2e-4 (GPT-3 scaling)"
echo "Start: $(date)"
echo "============================================"

PYTHONUNBUFFERED=1 python train.py \
    --out_dir="$RESULTS_DIR" \
    --dataset=classical_shuffled \
    --sequential_data=False \
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
echo "1B SHUFFLED RUN COMPLETE"
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
