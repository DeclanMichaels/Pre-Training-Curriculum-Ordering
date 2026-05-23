#!/bin/bash
# populate_repo.sh
# Copies paper-related artifacts from the experiment directory into the
# clean GitHub repo. Run from anywhere on the Mac.
#
# Usage: bash populate_repo.sh

set -e

CLOUD="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Claude.AI"
SRC="$CLOUD/-Experiment-Platform-/experiments/ScratchTrainingCuration"
DEST="$CLOUD/Pre-Training-Curriculum-Ordering"

echo "=== Populating Pre-Training-Curriculum-Ordering repo ==="
echo "Source: $SRC"
echo "Dest:   $DEST"
echo ""

# --- Directory structure ---
mkdir -p "$DEST/paper"
mkdir -p "$DEST/config"
mkdir -p "$DEST/scripts"
mkdir -p "$DEST/corpus"
mkdir -p "$DEST/results/analysis"
mkdir -p "$DEST/results/geometry/sequenced"
mkdir -p "$DEST/results/geometry/shuffled"
mkdir -p "$DEST/results/geometry/sequenced-10k"
mkdir -p "$DEST/results/geometry/continuation"
mkdir -p "$DEST/results/geometry/attnres-sequenced"
mkdir -p "$DEST/results/geometry/attnres-shuffled"
mkdir -p "$DEST/results/seeds"
mkdir -p "$DEST/results/training-logs"
mkdir -p "$DEST/results/samples"

# =====================================================================
# PAPER
# =====================================================================
echo "Copying paper..."
cp "$SRC/papers/curriculum-ordering-paper.pdf" "$DEST/paper/"
cp "$SRC/papers/curriculum-ordering-paper.md" "$DEST/paper/"
# HTML version if it exists
[ -f "$SRC/papers/curriculum-ordering-paper.html" ] && \
    cp "$SRC/papers/curriculum-ordering-paper.html" "$DEST/paper/"

# =====================================================================
# CONFIG
# =====================================================================
echo "Copying config..."
cp "$SRC/config/concepts.json" "$DEST/config/"
cp "$SRC/config/manifest.json" "$DEST/config/"
cp "$SRC/config/train_classical_sequenced.py" "$DEST/config/"
cp "$SRC/config/train_classical_shuffled.py" "$DEST/config/"

# =====================================================================
# SCRIPTS (paper-relevant only)
# =====================================================================
echo "Copying scripts..."
# Corpus assembly
cp "$SRC/scripts/fetch.py" "$DEST/scripts/"
[ -f "$SRC/scripts/fetch_manual.py" ] && cp "$SRC/scripts/fetch_manual.py" "$DEST/scripts/"
cp "$SRC/scripts/clean.py" "$DEST/scripts/"
cp "$SRC/scripts/assemble.py" "$DEST/scripts/"
cp "$SRC/scripts/build_manifest.py" "$DEST/scripts/"
cp "$SRC/scripts/train_tokenizer.py" "$DEST/scripts/"

# nanoGPT patching and data prep
cp "$SRC/scripts/patch_nanogpt.py" "$DEST/scripts/"
[ -f "$SRC/scripts/patch_attnres.py" ] && cp "$SRC/scripts/patch_attnres.py" "$DEST/scripts/"
cp "$SRC/scripts/prepare_nanogpt.py" "$DEST/scripts/"

# Analysis
cp "$SRC/scripts/extract_geometry.py" "$DEST/scripts/"
cp "$SRC/scripts/analyze_geometry.py" "$DEST/scripts/"
cp "$SRC/scripts/compute_dimensionality_flex.py" "$DEST/scripts/"
cp "$SRC/scripts/train_sae.py" "$DEST/scripts/"
cp "$SRC/scripts/sample_model.py" "$DEST/scripts/"

# Seed analysis
cp "$SRC/scripts/extract_all_seeds.py" "$DEST/scripts/"
cp "$SRC/scripts/analyze_all_seeds.py" "$DEST/scripts/"

# =====================================================================
# CORPUS METADATA (no .bin files — too large, regenerable)
# =====================================================================
echo "Copying corpus metadata..."
cp "$SRC/corpus/datasets/assembly_summary.json" "$DEST/corpus/"
cp "$SRC/corpus/datasets/sequenced.meta.json" "$DEST/corpus/"
cp "$SRC/corpus/datasets/shuffled-chunks.meta.json" "$DEST/corpus/"
cp "$SRC/corpus/datasets/shuffled-docs.meta.json" "$DEST/corpus/"
[ -f "$SRC/corpus/datasets/shuffled-docs-order.json" ] && \
    cp "$SRC/corpus/datasets/shuffled-docs-order.json" "$DEST/corpus/"

# =====================================================================
# RESULTS - Analysis JSONs
# =====================================================================
echo "Copying analysis results..."
for f in \
    standard-113-analysis.json \
    continuation-5k-analysis.json \
    continuation-vs-shuffled-analysis.json \
    attnres-113-analysis.json \
    sequenced-10k-vs-shuffled-analysis.json \
    sequenced-10k-vs-5k-analysis.json \
    sequenced-10k-vs-continuation-analysis.json \
    dimensionality_comparison.json \
    multi_seed_analysis.json \
    viewer_data.json \
    geometry_analysis.json; do
    [ -f "$SRC/results/analysis/$f" ] && cp "$SRC/results/analysis/$f" "$DEST/results/analysis/"
done

# =====================================================================
# RESULTS - Geometry extractions (hidden state JSONs, not checkpoints)
# =====================================================================
echo "Copying geometry data..."
# Copy the geometry JSON/npz files from each condition
for condition in sequenced shuffled sequenced-10k; do
    if [ -d "$SRC/results/geometry/$condition" ]; then
        cp "$SRC/results/geometry/$condition"/*.json "$DEST/results/geometry/$condition/" 2>/dev/null || true
        cp "$SRC/results/geometry/$condition"/*.npz "$DEST/results/geometry/$condition/" 2>/dev/null || true
    fi
done

# Standard conditions (the final paper runs)
for condition in standard-sequenced standard-shuffled standard-4k-then-shuffled standard-5k-then-shuffled; do
    if [ -d "$SRC/results/geometry/$condition" ]; then
        mkdir -p "$DEST/results/geometry/$condition"
        cp "$SRC/results/geometry/$condition"/*.json "$DEST/results/geometry/$condition/" 2>/dev/null || true
        cp "$SRC/results/geometry/$condition"/*.npz "$DEST/results/geometry/$condition/" 2>/dev/null || true
    fi
done

# AttnRes
for condition in attnres-sequenced attnres-shuffled; do
    if [ -d "$SRC/results/geometry/$condition" ]; then
        cp "$SRC/results/geometry/$condition"/*.json "$DEST/results/geometry/$condition/" 2>/dev/null || true
        cp "$SRC/results/geometry/$condition"/*.npz "$DEST/results/geometry/$condition/" 2>/dev/null || true
    fi
done

# =====================================================================
# RESULTS - Seed stability
# =====================================================================
echo "Copying seed results..."
for seed in seed-1337 seed-2024 seed-4242 seed-7777 seed-9999; do
    if [ -d "$SRC/results/geometry/$seed" ]; then
        mkdir -p "$DEST/results/seeds/$seed"
        cp "$SRC/results/geometry/$seed"/*.json "$DEST/results/seeds/$seed/" 2>/dev/null || true
        cp "$SRC/results/geometry/$seed"/*.npz "$DEST/results/seeds/$seed/" 2>/dev/null || true
    fi
done
[ -f "$SRC/results/seed-session.log" ] && cp "$SRC/results/seed-session.log" "$DEST/results/seeds/"
[ -d "$SRC/results/seed-results" ] && cp -r "$SRC/results/seed-results/"* "$DEST/results/seeds/" 2>/dev/null || true

# =====================================================================
# RESULTS - Training logs
# =====================================================================
echo "Copying training logs..."
[ -f "$SRC/results/standard/sequenced-training.log" ] && \
    cp "$SRC/results/standard/sequenced-training.log" "$DEST/results/training-logs/"
[ -f "$SRC/results/standard/shuffled-training.log" ] && \
    cp "$SRC/results/standard/shuffled-training.log" "$DEST/results/training-logs/"
[ -f "$SRC/results/combined-session.log" ] && \
    cp "$SRC/results/combined-session.log" "$DEST/results/training-logs/"
[ -f "$SRC/results/attnres-session.log" ] && \
    cp "$SRC/results/attnres-session.log" "$DEST/results/training-logs/"
[ -f "$SRC/results/sequenced-10k-training.log" ] && \
    cp "$SRC/results/sequenced-10k-training.log" "$DEST/results/training-logs/"

# =====================================================================
# RESULTS - Qualitative samples
# =====================================================================
echo "Copying qualitative samples..."
[ -f "$SRC/results/all-samples.txt" ] && cp "$SRC/results/all-samples.txt" "$DEST/results/samples/"
[ -f "$SRC/results/summary.csv" ] && cp "$SRC/results/summary.csv" "$DEST/results/"

# =====================================================================
# RESULTS - SAE summaries (not weights — too large)
# =====================================================================
echo "Copying SAE summaries..."
mkdir -p "$DEST/results/sae"
for saedir in "$SRC/results/sae"/*/; do
    dirname=$(basename "$saedir")
    mkdir -p "$DEST/results/sae/$dirname"
    [ -f "$saedir/sae_summary.json" ] && cp "$saedir/sae_summary.json" "$DEST/results/sae/$dirname/"
done

# =====================================================================
# RUN SCRIPTS (from results or root)
# =====================================================================
echo "Copying run scripts..."
for f in \
    "$SRC/run_standard.sh" \
    "$SRC/run_mamba.sh" \
    "$SRC/run_1b_shuffled.sh" \
    "$SRC/run_1b_sequenced.sh" \
    "$SRC/results/run_seq_then_shuffled_standard.sh"; do
    [ -f "$f" ] && cp "$f" "$DEST/scripts/"
done

# =====================================================================
# MODEL FILES (custom architectures)
# =====================================================================
echo "Copying model files..."
[ -f "$SRC/model_mamba.py" ] && cp "$SRC/model_mamba.py" "$DEST/scripts/"
[ -f "$SRC/model_attnres.py" ] && cp "$SRC/model_attnres.py" "$DEST/scripts/"

# =====================================================================
# CLEAN UP
# =====================================================================
# Remove any .DS_Store that crept in
find "$DEST" -name ".DS_Store" -delete 2>/dev/null || true
# Remove the placeholder file from git init
[ -f "$DEST/file" ] && rm "$DEST/file"

# =====================================================================
# SUMMARY
# =====================================================================
echo ""
echo "=== Done ==="
echo ""
echo "Directory structure:"
find "$DEST" -not -path '*/\.git/*' -not -name '.git' | head -80
echo ""
echo "Files copied. Still needed:"
echo "  1. README.md (Claude will generate)"
echo "  2. .gitignore"
echo "  3. LICENSE"
echo "  4. Review and commit"
