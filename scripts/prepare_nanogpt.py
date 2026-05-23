"""
Prepare nanoGPT data directories from assembled datasets.
Creates the train.bin / val.bin split that nanoGPT expects.

Usage: python scripts/prepare_nanogpt.py [--val-pct 5] [--nanogpt-dir ../nanoGPT]

DESIGN: Validation text is shared and held out from ALL conditions.

All three .bin files contain the same chunks in different orders. We select
val chunks from shuffled-chunks.bin (ensuring era-balanced sampling), hash
each val chunk, then for every condition we exclude any chunk whose content
matches a val chunk hash. This guarantees no model trains on val text,
regardless of chunk ordering.

Creates three dataset directories inside nanoGPT/data/:
  - classical_sequenced/     (experimental condition)
  - classical_shuffled_docs/ (Option 3)
  - classical_shuffled/      (primary baseline)
"""
import hashlib
import json
import os
import pickle
import sys
import argparse
import numpy as np

DATASET_DIR = "corpus/datasets"
TOKENIZER_MODEL = "corpus/tokenized/tokenizer.model"

DATASETS = {
    "sequenced": {"src": "sequenced.bin", "nanogpt_name": "classical_sequenced"},
    "shuffled-docs": {"src": "shuffled-docs.bin", "nanogpt_name": "classical_shuffled_docs"},
    "shuffled-chunks": {"src": "shuffled-chunks.bin", "nanogpt_name": "classical_shuffled"},
}


def get_vocab_size():
    """Read vocab size from trained tokenizer."""
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(TOKENIZER_MODEL)
        return sp.get_piece_size()
    except ImportError:
        print("ERROR: sentencepiece not installed. Cannot determine vocab size.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: could not load tokenizer at {TOKENIZER_MODEL}: {e}")
        print("Run scripts/train_tokenizer.py first.")
        sys.exit(1)


def chunk_hash(chunk_array):
    """Hash a uint16 chunk for content matching."""
    return hashlib.md5(chunk_array.tobytes()).digest()


def create_shared_val(chunk_size, val_pct, seed=42):
    """Create a shared validation set from the shuffled-chunks dataset.

    Returns (val_tokens, val_hashes) where val_tokens is a uint16 array
    and val_hashes is a set of md5 digests for content-based exclusion.
    """
    shuffled_path = os.path.join(DATASET_DIR, "shuffled-chunks.bin")
    if not os.path.exists(shuffled_path):
        print(f"ERROR: {shuffled_path} not found. Run scripts/assemble.py first.")
        sys.exit(1)

    data = np.fromfile(shuffled_path, dtype=np.uint16)
    n_tokens = len(data)
    n_chunks = n_tokens // chunk_size

    if n_chunks == 0:
        print(f"ERROR: dataset has {n_tokens} tokens but chunk_size is {chunk_size}.")
        sys.exit(1)

    # Select validation chunks by index (from shuffled data = era-balanced)
    n_val_chunks = max(1, int(n_chunks * val_pct / 100))
    rng = np.random.RandomState(seed)
    val_indices = rng.choice(n_chunks, size=n_val_chunks, replace=False)

    # Extract val tokens and build hash set for content matching
    val_chunks = []
    val_hashes = set()
    for idx in sorted(val_indices):
        start = idx * chunk_size
        chunk = data[start:start + chunk_size]
        val_chunks.append(chunk)
        val_hashes.add(chunk_hash(chunk))

    val_tokens = np.concatenate(val_chunks)
    return val_tokens, val_hashes, n_chunks


def build_train_bin(src_path, val_hashes, chunk_size):
    """Build train.bin by excluding any chunk whose content matches val.

    Uses md5 hashing for content-based exclusion. This correctly holds out
    val text from ALL conditions regardless of chunk ordering.
    """
    data = np.fromfile(src_path, dtype=np.uint16)
    n_tokens = len(data)
    n_chunks = n_tokens // chunk_size

    train_chunks = []
    excluded = 0
    for i in range(n_chunks):
        start = i * chunk_size
        chunk = data[start:start + chunk_size]
        if chunk_hash(chunk) in val_hashes:
            excluded += 1
        else:
            train_chunks.append(chunk)

    if not train_chunks:
        print("ERROR: no training chunks remain after val exclusion.")
        sys.exit(1)

    return np.concatenate(train_chunks), excluded


def main():
    parser = argparse.ArgumentParser(description="Prepare nanoGPT data directories")
    parser.add_argument("--val-pct", type=int, default=5, help="Validation set percentage (default 5)")
    parser.add_argument("--nanogpt-dir", type=str, default="nanoGPT",
                        help="Path to nanoGPT repository (default: ./nanoGPT)")
    parser.add_argument("--chunk-size", type=int, default=2048,
                        help="Chunk size used in assemble.py (default 2048)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for val chunk selection")
    args = parser.parse_args()

    nanogpt_data = os.path.join(args.nanogpt_dir, "data")
    if not os.path.isdir(args.nanogpt_dir):
        print(f"nanoGPT directory not found: {args.nanogpt_dir}")
        print(f"Clone it first: git clone https://github.com/karpathy/nanoGPT.git")
        sys.exit(1)

    vocab_size = get_vocab_size()
    vocab_size_padded = ((vocab_size + 63) // 64) * 64
    print(f"Vocab size: {vocab_size} (padded to {vocab_size_padded} for efficiency)")

    # Verify chunk_size matches what assemble.py used
    summary_path = os.path.join(DATASET_DIR, "assembly_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        assembled_chunk_size = summary.get("chunk_size")
        if assembled_chunk_size and assembled_chunk_size != args.chunk_size:
            print(f"ERROR: --chunk-size {args.chunk_size} does not match")
            print(f"       assemble.py used chunk_size={assembled_chunk_size}")
            print(f"       Re-run with --chunk-size {assembled_chunk_size}")
            sys.exit(1)

    # Create shared validation set from shuffled-chunks
    print(f"\nCreating shared validation set (seed={args.seed}, {args.val_pct}%)...")
    val_tokens, val_hashes, total_chunks = create_shared_val(
        args.chunk_size, args.val_pct, args.seed
    )
    n_val_chunks = len(val_hashes)
    print(f"  Selected {n_val_chunks} of {total_chunks} chunks for validation")
    print(f"  Val tokens: {len(val_tokens):,}")

    # Process each dataset condition
    for name, info in DATASETS.items():
        src = os.path.join(DATASET_DIR, info["src"])
        if not os.path.exists(src):
            print(f"\n[SKIP] {name}: {src} not found")
            continue

        dst = os.path.join(nanogpt_data, info["nanogpt_name"])
        os.makedirs(dst, exist_ok=True)
        print(f"\n[{name}] {src} -> {dst}")

        # Exclude val chunks by content hash.
        # NOTE: sequenced and shuffled-chunks have IDENTICAL chunks (same
        # boundaries, different order), so hash matching is exact. shuffled-docs
        # has DIFFERENT chunk boundaries (documents concatenated in different
        # order), so fewer hash matches are expected. This is documented in
        # meta.json and acceptable because shuffled-docs is not the primary
        # comparison condition.
        train_tokens, excluded = build_train_bin(src, val_hashes, args.chunk_size)

        if excluded != n_val_chunks:
            print(f"  WARNING: excluded {excluded} chunks, expected {n_val_chunks}")
            print(f"  (minor mismatch possible from EOS token boundary effects)")

        # Write train.bin
        train_path = os.path.join(dst, "train.bin")
        train_tokens.tofile(train_path)
        train_mb = os.path.getsize(train_path) / 1024 / 1024

        # Write shared val.bin (identical for all conditions)
        val_path = os.path.join(dst, "val.bin")
        val_tokens.tofile(val_path)
        val_mb = os.path.getsize(val_path) / 1024 / 1024

        print(f"  Train: {len(train_tokens):,} tokens ({train_mb:.1f} MB), {excluded} chunks excluded")
        print(f"  Val:   {len(val_tokens):,} tokens ({val_mb:.1f} MB) [shared across conditions]")

        # Write meta.pkl (nanoGPT reads vocab_size from this)
        meta_pkl = {"vocab_size": vocab_size_padded}
        meta_pkl_path = os.path.join(dst, "meta.pkl")
        with open(meta_pkl_path, 'wb') as f:
            pickle.dump(meta_pkl, f)

        # Write meta.json for our own reference
        meta = {
            "vocab_size": vocab_size_padded,
            "vocab_size_raw": vocab_size,
            "tokenizer": TOKENIZER_MODEL,
            "source_dataset": name,
            "val_pct": args.val_pct,
            "val_seed": args.seed,
            "val_shared": True,
            "val_excluded_by": "content_hash",
            "chunk_size": args.chunk_size,
            "train_tokens": int(len(train_tokens)),
            "val_tokens": int(len(val_tokens)),
            "chunks_excluded": excluded,
        }
        meta_path = os.path.join(dst, "meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    print(f"\nDone. nanoGPT data directories created in {nanogpt_data}/")
    print(f"\nTo train:")
    print(f"  cd {args.nanogpt_dir}")
    print(f"  python train.py config/train_classical_sequenced.py")
    print(f"  python train.py config/train_classical_shuffled.py")


if __name__ == "__main__":
    main()
