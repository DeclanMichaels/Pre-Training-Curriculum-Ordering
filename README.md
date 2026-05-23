# Pre-Training Curriculum Ordering

**We Should Consider Educating Models Before Training Them**

*Educated Pretraining Produces Fundamentally Different Representational Geometry Than Shuffled Pretraining on Identical Data*

Declan Michaels · [Cross-Cultural Alignment Study (CCAS)](https://moral-os.com) · May 2026

Pre-registered: [osf.io/2vcq6](https://osf.io/2vcq6)

---

## Summary

We trained two identical language models (GPT-2 small, 91.2M parameters) on the same 20.7M-token corpus of public-domain classical texts. The only variable was data presentation order: one model read the texts in a developmental sequence modeled on classical education (physical world → fables → ancients → logic → rhetoric → science → drama); the other read randomly shuffled 2,048-token chunks. Standard pretraining practice otherwise.

The educated model shows a smaller generalization gap (0.026 vs 1.47), progressive expansion of representational dimensionality through the network, and stronger domain clustering at every transformer layer. The shuffled model achieves lower absolute loss but overfits, compresses its representations at mid-network, and builds less structured internal geometry.

A continuation experiment — educating first, then training on shuffled data — produces richer representations than either approach alone, suggesting education and training are complementary phases, not alternatives.

## Key Findings

**Generalization gap.** The educated model maintains a gap of 0.026 after 5,000 steps. The shuffled model's gap widens to 1.47. Stable across five random seeds (educated: 0.10 ± 0.005; shuffled: 1.56 ± 0.01).

**Geometry diverges.** Mean CKA between conditions is 0.562 across 12 transformer layers. Permutation test at layer 6: p < 0.001.

**Dimensionality.** The educated model expands from 14 to 25 effective dimensions (input to output). The shuffled model collapses to 10-13 at mid-network with participation ratio near 1.5.

**Domain clustering.** The educated model maintains positive silhouette at all 12 layers (peak 0.057). The shuffled model peaks at 0.028.

**Education persists.** When the educated model continues training on shuffled data, domain clustering survives even as overall geometry shifts toward the shuffled model (continuation-vs-shuffled CKA: 0.860, but domain silhouette remains stronger at all layers).

## Repository Structure

```
paper/                          # The paper (PDF, markdown, HTML)
config/
  concepts.json                 # 54 concept words × 3 domains (probe inventory)
  manifest.json                 # Corpus manifest (113 texts, 7 stages)
  train_classical_sequenced.py  # nanoGPT config for sequenced condition
  train_classical_shuffled.py   # nanoGPT config for shuffled condition
scripts/
  # Corpus assembly pipeline
  fetch.py                      # Download texts from Project Gutenberg et al.
  fetch_manual.py               # Manual download helpers
  clean.py                      # Strip headers/footers, normalize text
  assemble.py                   # Assemble texts into stage-ordered corpus
  build_manifest.py             # Generate manifest.json from raw texts
  train_tokenizer.py            # Train SentencePiece BPE tokenizer (8K vocab)
  # nanoGPT integration
  patch_nanogpt.py              # Patch train.py for sequential data loading
  patch_attnres.py              # Patch for AttnRes architecture variant
  prepare_nanogpt.py            # Create train/val splits for nanoGPT
  # Analysis
  extract_geometry.py           # Extract hidden states for CKA/silhouette
  analyze_geometry.py           # Compute CKA, silhouette, MDS, bootstrap CIs
  compute_dimensionality_flex.py # Effective dimensionality & participation ratio
  train_sae.py                  # Train sparse autoencoders per layer
  sample_model.py               # Generate text samples from checkpoints
  extract_all_seeds.py          # Extract geometry across seed runs
  analyze_all_seeds.py          # Aggregate seed stability analysis
  # Model variants
  model_attnres.py              # AttnRes architecture (Section 5.1)
  model_mamba.py                # Mamba SSM (future work)
  # Run scripts
  run_standard.sh               # Main experiment (sequenced + shuffled)
  run_seq_then_shuffled_standard.sh  # Continuation experiment
corpus/
  assembly_summary.json         # Corpus assembly stats
  sequenced.meta.json           # Sequenced dataset metadata
  shuffled-chunks.meta.json     # Shuffled dataset metadata
  shuffled-docs.meta.json       # Document-level shuffle metadata
results/
  analysis/                     # CKA, silhouette, dimensionality JSONs
  geometry/                     # Per-layer hidden state extractions
  seeds/                        # Five-seed stability data
  training-logs/                # Loss curves for all conditions
  samples/                      # Qualitative generation samples
  sae/                          # Sparse autoencoder summaries
  summary.csv                   # Loss/gap summary across conditions
```

## Reproducing the Experiment

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA (bfloat16 support)
- GPU with ≥24GB VRAM (A40, A100, or A6000)
- ~$30 in GPU rental (RunPod or equivalent)

```bash
pip install sentencepiece numpy torch
```

### Step 1: Clone nanoGPT

```bash
git clone https://github.com/karpathy/nanoGPT.git
```

### Step 2: Assemble the corpus

The corpus consists of 113 public-domain texts. The manifest (`config/manifest.json`) lists every text with its source URL and stage assignment.

```bash
python scripts/fetch.py --manifest config/manifest.json --output corpus/raw/
python scripts/clean.py --input corpus/raw/ --output corpus/clean/
python scripts/train_tokenizer.py --input corpus/clean/ --vocab-size 8000 --output corpus/tokenized/
python scripts/assemble.py --manifest config/manifest.json --input corpus/clean/ --tokenizer corpus/tokenized/tokenizer.model --output corpus/datasets/
```

This produces three `.bin` files: `sequenced.bin` (stage-ordered), `shuffled-chunks.bin` (2048-token chunks, shuffled), and `shuffled-docs.bin` (document-level shuffle).

### Step 3: Prepare nanoGPT data directories

```bash
python scripts/patch_nanogpt.py --nanogpt-dir nanoGPT
python scripts/prepare_nanogpt.py --nanogpt-dir nanoGPT
```

This patches nanoGPT's `train.py` to support sequential data loading and creates `train.bin`/`val.bin` splits with a shared validation set held out from all conditions.

### Step 4: Train

**Sequenced (educated) condition:**
```bash
cd nanoGPT
python train.py \
    --dataset=classical_sequenced \
    --sequential_data=True \
    --out_dir=../results/sequenced \
    --max_iters=5000 \
    --eval_interval=250 \
    --learning_rate=3e-4 \
    --min_lr=3e-5 \
    --batch_size=4 \
    --gradient_accumulation_steps=8 \
    --dtype=bfloat16
```

**Shuffled (trained) condition:**
```bash
python train.py \
    --dataset=classical_shuffled \
    --out_dir=../results/shuffled \
    --max_iters=5000 \
    --eval_interval=250 \
    --learning_rate=3e-4 \
    --min_lr=3e-5 \
    --batch_size=4 \
    --gradient_accumulation_steps=8 \
    --dtype=bfloat16
```

**Continuation (educated → shuffled):**
```bash
# After sequenced training completes, copy checkpoint and continue on shuffled data:
cp results/sequenced/ckpt.pt results/continuation/ckpt.pt
python train.py \
    --dataset=classical_shuffled \
    --init_from=resume \
    --out_dir=../results/continuation \
    --max_iters=10000 \
    --lr_decay_iters=10000 \
    --eval_interval=250 \
    --dtype=bfloat16
```

### Step 5: Extract geometry and analyze

```bash
python scripts/extract_geometry.py --checkpoint results/sequenced/ckpt.pt --name sequenced --output results/geometry/sequenced/
python scripts/extract_geometry.py --checkpoint results/shuffled/ckpt.pt --name shuffled --output results/geometry/shuffled/
python scripts/analyze_geometry.py --model-a sequenced --model-b shuffled --output results/analysis/
python scripts/compute_dimensionality_flex.py sequenced shuffled
```

### Step 6: Train sparse autoencoders (exploratory)

```bash
python scripts/train_sae.py --checkpoint results/sequenced/ckpt.pt --output results/sae/sequenced/
python scripts/train_sae.py --checkpoint results/shuffled/ckpt.pt --output results/sae/shuffled/
```

### Step 7: Generate qualitative samples

```bash
python scripts/sample_model.py --checkpoint results/sequenced/ckpt.pt --tokenizer corpus/tokenized/tokenizer.model
python scripts/sample_model.py --checkpoint results/shuffled/ckpt.pt --tokenizer corpus/tokenized/tokenizer.model
```

## What's Not in This Repo

- **Model checkpoints.** Each is ~400MB for 91M parameters. Regenerable by running the training.
- **Binary corpus files.** ~120MB total. Regenerable from the manifest and assembly pipeline.
- **nanoGPT source.** Clone it separately; our patches modify `train.py` in place.
- **1B scaling experiments.** Preliminary results (described in the paper's future work) will be published separately.

## Pre-Registration

The experiment was pre-registered on the Open Science Framework prior to training: [osf.io/2vcq6](https://osf.io/2vcq6). Four hypotheses were registered with specific success criteria. H1 (geometry differs) and H3 (domain clustering) were confirmed. H2 (foundational retention) was directional but not decisive. H4 (qualitative generation) was exploratory.

## AI-Assisted Methodology

This research uses explicit AI-assisted methodology. Claude (Anthropic) assisted with code development, analysis pipeline construction, statistical review, and manuscript drafting. All experimental design decisions, interpretations, and conclusions are the researcher's. This acknowledgment is a deliberate choice: it is more honest than hiding AI involvement, and it drives rigor because reviewers hold AI-assisted work to a higher standard.

## Citation

```
Michaels, D. (2026). We Should Consider Educating Models Before Training Them:
Educated Pretraining Produces Fundamentally Different Representational Geometry
Than Shuffled Pretraining on Identical Data. Cross-Cultural Alignment Study (CCAS).
https://moral-os.com
```

## License

This work is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Declan Michaels · declan@moral-os.com · [moral-os.com](https://moral-os.com)
