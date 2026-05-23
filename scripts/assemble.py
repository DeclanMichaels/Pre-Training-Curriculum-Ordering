"""
Assemble training datasets from cleaned corpus.
Produces three dataset variants:
  1. sequenced.bin       - chunks in curriculum order (the experimental condition)
  2. shuffled-docs.bin   - whole documents randomized, then chunked (Option 3 prep)
  3. shuffled-chunks.bin - all chunks pooled and randomized (primary baseline)

Usage: python scripts/assemble.py [--chunk-size 2048] [--seed 42]

Designed for Option 2 (shuffled chunks) as primary comparison,
with Option 3 (shuffled docs) prepared for later analysis.

EOS tokens are inserted between documents so the model learns document boundaries.
Shuffle seeds: doc-shuffle uses --seed, chunk-shuffle uses --seed + 1.
This ensures the two shuffled datasets are independently randomized
with reproducible, distinct sequences.
"""
import os
import sys
import json
import random
import argparse
import numpy as np

CLEAN_DIR = "corpus/clean"
MANIFEST = "config/manifest.json"
DATASET_DIR = "corpus/datasets"
TOKENIZER_MODEL = "corpus/tokenized/tokenizer.model"


def load_manifest_order():
    """Return list of entries in curriculum order."""
    with open(MANIFEST) as f:
        manifest = json.load(f)
    entries = []
    for t in manifest["texts"]:
        entries.append({
            "stage": t["stage"],
            "seq": int(t["seq"]),  # ensure int for formatting
            "slug": t["slug"],
            "title": t["title"],
        })
    entries.sort(key=lambda x: (x["stage"], x["seq"]))
    return entries


def find_clean_file(stage, seq, slug):
    """Find the cleaned text file for a manifest entry."""
    stage_dir = os.path.join(CLEAN_DIR, f"stage-{stage}")
    if not os.path.isdir(stage_dir):
        return None
    prefix = f"{seq:02d}-"
    for fname in sorted(os.listdir(stage_dir)):
        if fname.startswith(prefix) and fname.endswith('.txt'):
            return os.path.join(stage_dir, fname)
    return None


def tokenize_text(text, sp_model):
    """Tokenize text using SentencePiece model. Returns list of int token IDs."""
    return sp_model.encode(text)


def chunk_tokens(token_ids, chunk_size):
    """Split token list into fixed-size chunks. Drop remainder."""
    n_chunks = len(token_ids) // chunk_size
    return [token_ids[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]


def write_dataset(chunks, path):
    """Write dataset as numpy binary (flat uint16 array).
    Also writes a .meta.json with chunk count and size.
    """
    all_tokens = []
    for chunk in chunks:
        all_tokens.extend(chunk)

    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(path)

    meta = {
        "n_chunks": len(chunks),
        "chunk_size": len(chunks[0]) if chunks else 0,
        "n_tokens": len(all_tokens),
        "dtype": "uint16",
        "file_bytes": len(all_tokens) * 2,
    }
    meta_path = path.replace('.bin', '.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


def main():
    parser = argparse.ArgumentParser(description="Assemble training datasets")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Tokens per training sequence")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for doc-shuffle (chunk-shuffle uses seed+1)")
    parser.add_argument("--no-tokenizer", action="store_true",
                        help="Work with raw bytes (for testing before tokenizer is trained)")
    args = parser.parse_args()

    os.makedirs(DATASET_DIR, exist_ok=True)

    # Load tokenizer
    sp_model = None
    eos_id = None
    if not args.no_tokenizer:
        try:
            import sentencepiece as spm
            if os.path.exists(TOKENIZER_MODEL):
                sp_model = spm.SentencePieceProcessor()
                sp_model.load(TOKENIZER_MODEL)
                eos_id = sp_model.eos_id()
                vocab_size = sp_model.get_piece_size()
                print(f"Loaded tokenizer: vocab size {vocab_size}, EOS id {eos_id}")
                if vocab_size > 65535:
                    print(f"ERROR: vocab size {vocab_size} exceeds uint16 max (65535).")
                    print("Retrain tokenizer with --vocab-size <= 65535 or modify this script to use uint32.")
                    sys.exit(1)
            else:
                print(f"Tokenizer not found at {TOKENIZER_MODEL}")
                print("Run scripts/train_tokenizer.py first, or use --no-tokenizer for byte-mode testing")
                sys.exit(1)
        except ImportError:
            print("sentencepiece not installed. Use --no-tokenizer for byte-mode testing.")
            sys.exit(1)

    # Load manifest and find files
    entries = load_manifest_order()
    print(f"Manifest: {len(entries)} texts in curriculum order")

    documents = []
    total_tokens = 0
    missing = 0

    for entry in entries:
        path = find_clean_file(entry["stage"], entry["seq"], entry["slug"])
        if path is None:
            print(f"  [MISSING] Stage {entry['stage']}, #{entry['seq']}: {entry['title']}")
            missing += 1
            continue

        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        if sp_model:
            tokens = tokenize_text(text, sp_model)
        else:
            tokens = list(text.encode('utf-8'))

        documents.append({"entry": entry, "tokens": tokens})
        total_tokens += len(tokens)
        print(f"  Stage {entry['stage']}, #{entry['seq']}: {entry['title']} -- {len(tokens):,} tokens")

    print(f"\nTotal: {len(documents)} documents, {total_tokens:,} tokens, {missing} missing")

    # Build concatenated token stream with EOS between documents
    # This is the raw material; chunking happens after.
    print(f"\nInserting EOS tokens between documents...")
    sequenced_stream = []
    for i, doc in enumerate(documents):
        sequenced_stream.extend(doc["tokens"])
        if eos_id is not None:
            sequenced_stream.append(eos_id)
        elif not args.no_tokenizer:
            pass  # shouldn't happen, but don't insert garbage
        # In byte mode, use newline bytes as separator
        else:
            sequenced_stream.extend([10, 10])  # two newlines

    eos_count = len(documents) if eos_id is not None else 0
    print(f"  {eos_count} EOS tokens inserted, stream length: {len(sequenced_stream):,}")

    # Chunk the sequenced stream
    print(f"Chunking at {args.chunk_size} tokens per sequence...")
    all_sequenced_chunks = chunk_tokens(sequenced_stream, args.chunk_size)

    total_in_chunks = len(all_sequenced_chunks) * args.chunk_size
    remainder = len(sequenced_stream) - total_in_chunks
    print(f"Total chunks: {len(all_sequenced_chunks)}")
    print(f"Remainder (dropped): {remainder:,} tokens ({remainder/len(sequenced_stream)*100:.1f}%)")

    # Dataset 1: Sequenced (curriculum order)
    path1 = os.path.join(DATASET_DIR, "sequenced.bin")
    meta1 = write_dataset(all_sequenced_chunks, path1)
    print(f"\nSequenced: {meta1['n_chunks']} chunks, {meta1['n_tokens']:,} tokens -> {path1}")

    # Dataset 2: Shuffled documents
    # Rebuild streams with documents in random order, EOS between each
    rng_docs = random.Random(args.seed)
    doc_indices = list(range(len(documents)))
    rng_docs.shuffle(doc_indices)

    shuffled_doc_stream = []
    for idx in doc_indices:
        shuffled_doc_stream.extend(documents[idx]["tokens"])
        if eos_id is not None:
            shuffled_doc_stream.append(eos_id)
        elif args.no_tokenizer:
            shuffled_doc_stream.extend([10, 10])

    shuffled_doc_chunks = chunk_tokens(shuffled_doc_stream, args.chunk_size)

    path2 = os.path.join(DATASET_DIR, "shuffled-docs.bin")
    meta2 = write_dataset(shuffled_doc_chunks, path2)
    print(f"Shuffled-docs: {meta2['n_chunks']} chunks, {meta2['n_tokens']:,} tokens -> {path2}")

    # Log doc order for reproducibility
    doc_order_path = os.path.join(DATASET_DIR, "shuffled-docs-order.json")
    with open(doc_order_path, 'w') as f:
        order_log = [{"original_index": idx, "title": documents[idx]["entry"]["title"]}
                     for idx in doc_indices]
        json.dump({"seed": args.seed, "order": order_log}, f, indent=2)

    # Dataset 3: Shuffled chunks (uses seed+1 for independent shuffle)
    rng_chunks = random.Random(args.seed + 1)
    shuffled_all = list(all_sequenced_chunks)
    rng_chunks.shuffle(shuffled_all)

    path3 = os.path.join(DATASET_DIR, "shuffled-chunks.bin")
    meta3 = write_dataset(shuffled_all, path3)
    print(f"Shuffled-chunks: {meta3['n_chunks']} chunks, {meta3['n_tokens']:,} tokens -> {path3}")

    # Summary
    summary = {
        "seed_docs": args.seed,
        "seed_chunks": args.seed + 1,
        "chunk_size": args.chunk_size,
        "eos_token_id": eos_id,
        "n_documents": len(documents),
        "n_missing": missing,
        "total_tokens_raw": total_tokens,
        "total_tokens_with_eos": len(sequenced_stream),
        "total_chunks": len(all_sequenced_chunks),
        "remainder_tokens_dropped": remainder,
        "datasets": {
            "sequenced": meta1,
            "shuffled_docs": meta2,
            "shuffled_chunks": meta3,
        },
        "tokenizer": TOKENIZER_MODEL if sp_model else "byte-level (testing)",
    }
    summary_path = os.path.join(DATASET_DIR, "assembly_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
