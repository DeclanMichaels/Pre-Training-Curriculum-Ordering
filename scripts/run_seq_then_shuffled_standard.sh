#!/bin/bash
# =============================================================================
# Run standard nanoGPT: 5000 sequenced then 5000 shuffled (continuous)
# =============================================================================
# Single continuous 10,000-step run with numbered checkpoints every 250 steps.
# First 5000 steps on curriculum-ordered data, then 5000 on shuffled data.
# Produces 40 geometry snapshots for the full evolution story.
#
# Usage: bash run_seq_then_shuffled_standard.sh 2>&1 | tee seq-then-shuffled-standard.log
# Estimated time: ~2.5 hours on A6000/A40 at batch 32
# =============================================================================

set -e

TRAIN_DIR="/workspace/classical-training"
RESULTS_DIR="$TRAIN_DIR/results/standard-seq-then-shuffled"

echo "============================================"
echo "Standard nanoGPT: Sequenced then Shuffled"
echo "  Phase 1: steps 0-5000 on sequenced data"
echo "  Phase 2: steps 5000-10000 on shuffled data"
echo "============================================"
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

cd "$TRAIN_DIR"

if [ ! -f "nanoGPT/data/classical_sequenced/train.bin" ]; then
    echo "ERROR: workspace not set up. Run setup_workspace.sh first."
    exit 1
fi

cd nanoGPT

# Set batch size based on VRAM (standard nanoGPT thresholds)
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

mkdir -p "$RESULTS_DIR"

# Verify numbered checkpoint patch is in place
if ! grep -q "ckpt_" train.py; then
    echo "ERROR: Numbered checkpoint patch not found in train.py."
    echo "Run setup_workspace.sh again."
    exit 1
fi
echo "Numbered checkpoint patch verified."
echo ""

# ---- Phase 1: Sequenced (steps 0-5000) ----
echo "============================================"
echo "Phase 1: Sequenced data (steps 0-5000)"
echo "Start: $(date)"
echo "============================================"
echo ""

PYTHONUNBUFFERED=1 python train.py \
    --out_dir="$RESULTS_DIR" \
    --dataset=classical_sequenced \
    --sequential_data=True \
    --use_attnres=False \
    --batch_size=$BATCH --gradient_accumulation_steps=$ACCUM \
    --eval_interval=250 --eval_iters=100 --log_interval=50 \
    --always_save_checkpoint=True \
    --wandb_log=False \
    --block_size=2048 \
    --n_layer=12 --n_head=12 --n_embd=768 \
    --dropout=0.0 --bias=True \
    --init_from=scratch \
    --max_iters=5000 \
    --learning_rate=3e-4 \
    --weight_decay=0.1 \
    --beta1=0.9 --beta2=0.95 \
    --grad_clip=1.0 \
    --decay_lr=True \
    --warmup_iters=200 \
    --lr_decay_iters=10000 \
    --min_lr=3e-5 \
    --device=cuda --dtype=bfloat16 \
    --compile=False \
    2>&1 | tee "$RESULTS_DIR/phase1-sequenced.log" || true

echo ""
echo "Phase 1 completed: $(date)"
echo "Phase 1 final: $(grep '^step 5000' "$RESULTS_DIR/phase1-sequenced.log" 2>/dev/null || echo 'check log')"
echo ""

# ---- Phase 2: Shuffled (steps 5000-10000) ----
echo "============================================"
echo "Phase 2: Shuffled data (steps 5000-10000)"
echo "Start: $(date)"
echo "============================================"
echo ""

PYTHONUNBUFFERED=1 python train.py \
    --out_dir="$RESULTS_DIR" \
    --init_from=resume \
    --dataset=classical_shuffled \
    --sequential_data=False \
    --use_attnres=False \
    --batch_size=$BATCH --gradient_accumulation_steps=$ACCUM \
    --eval_interval=250 --eval_iters=100 --log_interval=50 \
    --always_save_checkpoint=True \
    --wandb_log=False \
    --block_size=2048 \
    --n_layer=12 --n_head=12 --n_embd=768 \
    --dropout=0.0 --bias=True \
    --max_iters=10000 \
    --learning_rate=3e-4 \
    --weight_decay=0.1 \
    --beta1=0.9 --beta2=0.95 \
    --grad_clip=1.0 \
    --decay_lr=True \
    --warmup_iters=200 \
    --lr_decay_iters=10000 \
    --min_lr=3e-5 \
    --device=cuda --dtype=bfloat16 \
    --compile=False \
    2>&1 | tee "$RESULTS_DIR/phase2-shuffled.log" || true

echo ""
echo "Phase 2 completed: $(date)"
echo ""

cd "$TRAIN_DIR"

# ---- Summary ----
echo "============================================"
echo "COMBINED RUN COMPLETE"
echo "============================================"
echo "End: $(date)"
echo ""
echo "Phase 1 (sequenced): $(grep '^step 5000' "$RESULTS_DIR/phase1-sequenced.log" 2>/dev/null || echo 'check log')"
echo "Phase 2 (shuffled):  $(grep '^step 10000' "$RESULTS_DIR/phase2-shuffled.log" 2>/dev/null || echo 'check log')"
echo ""
echo "Checkpoints:"
ls "$RESULTS_DIR"/ckpt_*.pt 2>/dev/null | wc -l
echo "numbered snapshots"
echo ""
echo "Disk usage:"
du -sh "$RESULTS_DIR"
