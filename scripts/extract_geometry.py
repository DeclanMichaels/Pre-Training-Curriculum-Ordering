#!/usr/bin/env python3
"""
Extract hidden states from trained nanoGPT models for concept inventory.

Feeds bare concept words into each model, captures activation vectors
at every transformer layer. No prompting, no instruction following.
The model processes the token(s) and we record how it represents them.

Usage:
    python scripts/extract_geometry.py --model sequenced
    python scripts/extract_geometry.py --model shuffled
    python scripts/extract_geometry.py --model both

Produces .npz files in results/geometry/{model_name}/ containing
hidden state arrays of shape (n_layers+1, hidden_dim) per concept.

Requires: trained checkpoint in nanoGPT/out-classical-{model}/ckpt.pt
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import sentencepiece as spm

# Add nanoGPT to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
NANOGPT_DIR = PROJECT_ROOT / "nanoGPT"
sys.path.insert(0, str(NANOGPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from model import GPTConfig, GPT

TOKENIZER_MODEL = PROJECT_ROOT / "corpus" / "tokenized" / "tokenizer.model"
CONCEPTS_FILE = PROJECT_ROOT / "config" / "concepts.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "geometry"

MODEL_DIRS = {
    "sequenced": NANOGPT_DIR / "out-classical-sequenced",
    "shuffled": NANOGPT_DIR / "out-classical-shuffled",
    "shuffled-docs": NANOGPT_DIR / "out-classical-shuffled-docs",
}

# Also check project root for standalone checkpoint copies
STANDALONE_CKPTS = {
    "sequenced": PROJECT_ROOT / "sequenced-ckpt.pt",
    "shuffled": PROJECT_ROOT / "shuffled-ckpt.pt",
}


def find_checkpoint(model_name):
    """Find checkpoint file, checking both nanoGPT output dir and standalone copies."""
    # Check nanoGPT output directory
    if model_name in MODEL_DIRS:
        ckpt = MODEL_DIRS[model_name] / "ckpt.pt"
        if ckpt.exists():
            return ckpt

    # Check standalone copies
    if model_name in STANDALONE_CKPTS:
        ckpt = STANDALONE_CKPTS[model_name]
        if ckpt.exists():
            return ckpt

    # Check training-results.tar.gz extraction
    tar_path = PROJECT_ROOT / "nanoGPT" / f"out-classical-{model_name}" / "ckpt.pt"
    if tar_path.exists():
        return tar_path

    return None


def load_model(ckpt_path, device='cpu', use_attnres=False):
    """Load a trained nanoGPT checkpoint."""
    print(f"  Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if use_attnres:
        from model_attnres import GPTConfig as AttnResConfig, GPT as AttnResGPT
        config = AttnResConfig(**checkpoint['model_args'])
        model = AttnResGPT(config)
    else:
        config = GPTConfig(**checkpoint['model_args'])
        model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    iter_num = checkpoint.get('iter_num', '?')
    best_val = checkpoint.get('best_val_loss', '?')
    print(f"  Loaded: iter {iter_num}, val loss {best_val}")
    print(f"  Architecture: {config.n_layer}L, {config.n_head}H, {config.n_embd}E")

    return model, config


def load_concepts():
    """Load concept inventory. Handles both RCP V2 and rep-geo formats."""
    if not CONCEPTS_FILE.exists():
        print(f"Concepts file not found at {CONCEPTS_FILE}")
        print("Copying from RCP V2 stimuli...")
        # Try to find it in the experiment platform
        rcp_concepts = PROJECT_ROOT.parent / "rcp-v2" / "stimuli" / "concepts.json"
        if rcp_concepts.exists():
            import shutil
            CONCEPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rcp_concepts, CONCEPTS_FILE)
        else:
            print(f"ERROR: Cannot find concepts.json at {CONCEPTS_FILE} or {rcp_concepts}")
            sys.exit(1)

    with open(CONCEPTS_FILE) as f:
        data = json.load(f)

    # Handle both key formats
    domains = data.get("concepts") or data.get("domains")
    if not domains:
        print("ERROR: concepts.json must have 'concepts' or 'domains' key")
        sys.exit(1)

    concepts = []
    for domain, items in domains.items():
        for concept in items:
            concepts.append({"domain": domain, "concept": concept})

    return concepts


def extract_hidden_states(model, tokenizer, text, device='cpu', pooling='mean'):
    """
    Forward pass on a concept word/phrase, capturing hidden states at every layer.

    Uses forward hooks on transformer blocks to capture intermediate activations.

    Returns:
        np.ndarray of shape (n_layers+1, hidden_dim)
        Layer 0 = embedding output. Layers 1..N = transformer block outputs.
    """
    # Tokenize
    token_ids = tokenizer.encode(text)
    if not token_ids:
        # Fallback: try with a space prefix (SentencePiece sometimes needs it)
        token_ids = tokenizer.encode(" " + text)
    if not token_ids:
        print(f"    WARNING: empty tokenization for '{text}', using UNK")
        token_ids = [tokenizer.unk_id()]

    x = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Storage for hidden states
    hidden_states = []

    # Hook to capture embedding output
    def embed_hook(module, input, output):
        hidden_states.append(output.detach())

    # Hook to capture each transformer block output
    def block_hook(module, input, output):
        hidden_states.append(output.detach())

    # Register hooks
    hooks = []

    # Embedding: wte + wpe + drop
    # In nanoGPT, the embedding is computed in the forward method:
    #   tok_emb = self.transformer.wte(idx)
    #   pos_emb = self.transformer.wpe(pos)
    #   x = self.transformer.drop(tok_emb + pos_emb)
    # We hook the drop layer to get the combined embedding
    hooks.append(model.transformer.drop.register_forward_hook(embed_hook))

    # Transformer blocks
    for block in model.transformer.h:
        hooks.append(block.register_forward_hook(block_hook))

    # Forward pass
    with torch.no_grad():
        model(x)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Pool across sequence dimension
    layers = []
    for h in hidden_states:
        # h shape: (1, seq_len, hidden_dim)
        h = h.squeeze(0)  # (seq_len, hidden_dim)

        if pooling == 'mean':
            pooled = h.mean(dim=0)
        elif pooling == 'last':
            pooled = h[-1]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        layers.append(pooled.cpu().float().numpy())

    return np.stack(layers)  # (n_layers+1, hidden_dim)


def extract_model(model_name, concepts, device='cpu', pooling='mean'):
    """Extract hidden states for all concepts from one model."""
    ckpt_path = find_checkpoint(model_name)
    if ckpt_path is None:
        print(f"ERROR: No checkpoint found for '{model_name}'")
        print(f"  Checked: {MODEL_DIRS.get(model_name, 'N/A')}")
        print(f"  Checked: {STANDALONE_CKPTS.get(model_name, 'N/A')}")
        return False

    print(f"\nExtracting: {model_name}")
    model, config = load_model(ckpt_path, device)

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL))

    # Output directory
    output_dir = RESULTS_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract
    times = []
    all_hidden = []

    for i, concept_info in enumerate(concepts):
        t0 = time.time()

        hidden = extract_hidden_states(
            model, sp, concept_info["concept"],
            device=device, pooling=pooling,
        )

        all_hidden.append(hidden)

        # Save individual .npz
        filename = f"{i:03d}_{concept_info['domain']}_{concept_info['concept'].replace(' ', '_')}.npz"
        np.savez_compressed(
            output_dir / filename,
            hidden_states=hidden,
            concept=concept_info["concept"],
            domain=concept_info["domain"],
            model=model_name,
            pooling=pooling,
        )

        elapsed = time.time() - t0
        times.append(elapsed)

        if (i + 1) % 18 == 0 or i == 0:
            print(f"  [{i+1}/{len(concepts)}] {concept_info['concept']}: "
                  f"{hidden.shape}, {elapsed:.3f}s")

    # Save combined array
    combined = np.stack(all_hidden)  # (n_concepts, n_layers+1, hidden_dim)
    np.savez_compressed(
        output_dir / "all_hidden_states.npz",
        hidden_states=combined,
        concepts=[c["concept"] for c in concepts],
        domains=[c["domain"] for c in concepts],
        model=model_name,
        pooling=pooling,
        n_layers=config.n_layer,
        hidden_dim=config.n_embd,
    )

    # Save metadata
    meta = {
        "model": model_name,
        "checkpoint": str(ckpt_path),
        "device": device,
        "pooling": pooling,
        "n_concepts": len(concepts),
        "n_layers": config.n_layer + 1,  # +1 for embedding layer
        "hidden_dim": config.n_embd,
        "shape": list(combined.shape),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "total_seconds": sum(times),
    }
    with open(output_dir / "extraction_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Done: {combined.shape} in {sum(times):.1f}s")
    print(f"  Saved to {output_dir}")

    # Cleanup
    del model
    gc.collect()

    return True


def main():
    parser = argparse.ArgumentParser(description="Extract hidden states from trained models")
    parser.add_argument("--model", type=str, default="both",
                        choices=["sequenced", "shuffled", "shuffled-docs", "both", "all"],
                        help="Which model to extract (default: both). Ignored if --checkpoint is set.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit path to a checkpoint file. Use with --name.")
    parser.add_argument("--name", type=str, default=None,
                        help="Output directory name under results/geometry/. Required with --checkpoint.")
    parser.add_argument("--attnres", action="store_true",
                        help="Use AttnRes model class instead of standard GPT.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu recommended for extraction)")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "last"],
                        help="Pooling method for multi-token concepts")
    args = parser.parse_args()

    # Load concepts
    concepts = load_concepts()
    print(f"Loaded {len(concepts)} concepts across {len(set(c['domain'] for c in concepts))} domains")

    # Explicit checkpoint mode
    if args.checkpoint:
        if not args.name:
            print("ERROR: --name is required when using --checkpoint")
            sys.exit(1)
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"ERROR: Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        print(f"\nExtracting: {args.name} from {ckpt_path}")
        model, config = load_model(str(ckpt_path), args.device, use_attnres=args.attnres)
        sp = spm.SentencePieceProcessor()
        sp.load(str(TOKENIZER_MODEL))
        output_dir = RESULTS_DIR / args.name
        output_dir.mkdir(parents=True, exist_ok=True)
        times = []
        all_hidden = []
        for i, concept_info in enumerate(concepts):
            t0 = time.time()
            hidden = extract_hidden_states(model, sp, concept_info["concept"], device=args.device, pooling=args.pooling)
            all_hidden.append(hidden)
            filename = f"{i:03d}_{concept_info['domain']}_{concept_info['concept'].replace(' ', '_')}.npz"
            np.savez_compressed(output_dir / filename, hidden_states=hidden, concept=concept_info["concept"], domain=concept_info["domain"], model=args.name, pooling=args.pooling)
            elapsed = time.time() - t0
            times.append(elapsed)
            if (i + 1) % 18 == 0 or i == 0:
                print(f"  [{i+1}/{len(concepts)}] {concept_info['concept']}: {hidden.shape}, {elapsed:.3f}s")
        combined = np.stack(all_hidden)
        np.savez_compressed(output_dir / "all_hidden_states.npz", hidden_states=combined, concepts=[c["concept"] for c in concepts], domains=[c["domain"] for c in concepts], model=args.name, pooling=args.pooling, n_layers=config.n_layer, hidden_dim=config.n_embd)
        meta = {"model": args.name, "checkpoint": str(ckpt_path), "device": args.device, "pooling": args.pooling, "n_concepts": len(concepts), "n_layers": config.n_layer + 1, "hidden_dim": config.n_embd, "shape": list(combined.shape), "extracted_at": datetime.now(timezone.utc).isoformat(), "total_seconds": sum(times)}
        with open(output_dir / "extraction_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Done: {combined.shape} in {sum(times):.1f}s")
        print(f"  Saved to {output_dir}")
        del model
        gc.collect()
        return

    # Determine which models to extract
    if args.model == "both":
        models = ["sequenced", "shuffled"]
    elif args.model == "all":
        models = ["sequenced", "shuffled", "shuffled-docs"]
    else:
        models = [args.model]

    # Extract
    for model_name in models:
        ok = extract_model(model_name, concepts, device=args.device, pooling=args.pooling)
        if not ok:
            print(f"  Skipping {model_name}")

    print("\nExtraction complete. Run scripts/analyze_geometry.py next.")


if __name__ == "__main__":
    main()
