"""
Train a BPE tokenizer on the cleaned corpus using SentencePiece.

Usage: python scripts/train_tokenizer.py [--vocab-size 8000]

Trains on all cleaned text files, saves model to corpus/tokenized/tokenizer.model.
Vocab size default 8000 is appropriate for ~12M token corpus.
Larger models (32K) would be underfitting on this corpus size.
"""
import os
import sys
import glob
import tempfile
import argparse

CLEAN_DIR = "corpus/clean"
OUTPUT_DIR = "corpus/tokenized"
MODEL_PREFIX = os.path.join(OUTPUT_DIR, "tokenizer")

# uint16 max for assemble.py compatibility
MAX_VOCAB_FOR_UINT16 = 65535


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on cleaned corpus")
    parser.add_argument("--vocab-size", type=int, default=8000,
                        help="Vocabulary size (default 8000)")
    parser.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"],
                        help="SentencePiece model type")
    args = parser.parse_args()

    if args.vocab_size > MAX_VOCAB_FOR_UINT16:
        print(f"Warning: vocab size {args.vocab_size} exceeds uint16 max ({MAX_VOCAB_FOR_UINT16}).")
        print("assemble.py stores token IDs as uint16. Use --vocab-size <= 65535 or modify assemble.py to use uint32.")
        sys.exit(1)

    try:
        import sentencepiece as spm
    except ImportError:
        print("Install sentencepiece: pip install sentencepiece")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    txt_files = sorted(glob.glob(os.path.join(CLEAN_DIR, "stage-*", "*.txt")))
    if not txt_files:
        print(f"No cleaned text files found in {CLEAN_DIR}/stage-*/")
        print("Run scripts/clean.py first.")
        sys.exit(1)

    print(f"Found {len(txt_files)} cleaned text files")

    # Concatenate into a single temp file for SentencePiece
    total_chars = 0
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp_path = tmp.name
            for fpath in txt_files:
                with open(fpath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    tmp.write(text)
                    tmp.write('\n')
                    total_chars += len(text)

        print(f"Total corpus: {total_chars:,} characters ({total_chars/1024/1024:.1f} MB)")
        print(f"Training {args.model_type} tokenizer with vocab size {args.vocab_size}...")

        spm.SentencePieceTrainer.train(
            input=tmp_path,
            model_prefix=MODEL_PREFIX,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=0.9995,
            num_threads=os.cpu_count() or 4,
            split_digits=True,
            byte_fallback=True,
            # Increased from 16384: Montaigne, Tolstoy, Melville have long paragraphs.
            # SentencePiece splits on newlines and truncates beyond this limit.
            max_sentence_length=65536,
            pad_id=3,
            unk_id=0,
            bos_id=1,
            eos_id=2,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Quick test
    sp = spm.SentencePieceProcessor()
    sp.load(f"{MODEL_PREFIX}.model")
    test_sentences = [
        "The fox could not reach the grapes.",
        "Water runs downhill because of gravity.",
        "To be or not to be, that is the question.",
        "If two triangles have two sides equal, the angles are equal.",
    ]
    print(f"\nTokenizer trained: {sp.get_piece_size()} tokens")
    print("\nSample tokenizations:")
    for s in test_sentences:
        tokens = sp.encode(s, out_type=str)
        ids = sp.encode(s)
        print(f"  '{s}'")
        print(f"    tokens: {tokens}")
        print(f"    ids:    {ids}")
        print(f"    count:  {len(ids)}")

    print(f"\nModel saved: {MODEL_PREFIX}.model")
    print(f"Vocab saved: {MODEL_PREFIX}.vocab")


if __name__ == "__main__":
    main()
