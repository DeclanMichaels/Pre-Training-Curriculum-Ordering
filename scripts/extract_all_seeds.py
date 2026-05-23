#!/usr/bin/env python3
"""
Extract hidden states from all seed-pair checkpoints.

Usage:
    python scripts/extract_all_seeds.py

Reads checkpoints from results/seed-results/seed-{N}/{sequenced,shuffled}-ckpt.pt
Outputs to results/geometry/seed-{N}/{sequenced,shuffled}/all_hidden_states.npz
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import sentencepiece as spm

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
NANOGPT_DIR = PROJECT_ROOT / "nanoGPT"
sys.path.insert(0, str(NANOGPT_DIR))

from model import GPTConfig, GPT

TOKENIZER_MODEL = PROJECT_ROOT / "corpus" / "tokenized" / "tokenizer.model"
CONCEPTS_FILE = PROJECT_ROOT / "config" / "concepts.json"
SEED_RESULTS_DIR = PROJECT_ROOT / "results" / "seed-results"
GEOMETRY_DIR = PROJECT_ROOT / "results" / "geometry"

SEEDS = [1337, 2024, 4242, 7777, 9999]
CONDITIONS = ["sequenced", "shuffled"]

# Also extract from the standalone run-2 checkpoints if present
STANDALONE = {
    "run2-sequenced": PROJECT_ROOT / "sequenced-ckpt.pt",
    "run2-shuffled": PROJECT_ROOT / "shuffled-ckpt.pt",
}


def load_concepts():
    with open(CONCEPTS_FILE) as f:
        data = json.load(f)
    domains = data.get("concepts") or data.get("domains")
    concepts = []
    for domain, items in domains.items():
        for concept in items:
            concepts.append({"domain": domain, "concept": concept})
    return concepts


def load_model(ckpt_path, device='cpu'):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint['model_args'])
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)
    return model, config


def extract_hidden_states(model, tokenizer, text, device='cpu'):
    token_ids = tokenizer.encode(text)
    if not token_ids:
        token_ids = tokenizer.encode(" " + text)
    if not token_ids:
        token_ids = [tokenizer.unk_id()]

    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    hidden_states = []

    def embed_hook(module, input, output):
        hidden_states.append(output.detach())

    def block_hook(module, input, output):
        hidden_states.append(output.detach())

    hooks = []
    hooks.append(model.transformer.drop.register_forward_hook(embed_hook))
    for block in model.transformer.h:
        hooks.append(block.register_forward_hook(block_hook))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    layers = []
    for h in hidden_states:
        h = h.squeeze(0)
        pooled = h.mean(dim=0)
        layers.append(pooled.cpu().float().numpy())

    return np.stack(layers)


def extract_checkpoint(ckpt_path, concepts, sp, output_dir, label, device='cpu'):
    print(f"  Loading {label}: {ckpt_path.name}")
    model, config = load_model(ckpt_path, device)

    all_hidden = []
    for concept_info in concepts:
        hidden = extract_hidden_states(model, sp, concept_info["concept"], device)
        all_hidden.append(hidden)

    combined = np.stack(all_hidden)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "all_hidden_states.npz",
        hidden_states=combined,
        concepts=[c["concept"] for c in concepts],
        domains=[c["domain"] for c in concepts],
        model=label,
        n_layers=config.n_layer,
        hidden_dim=config.n_embd,
    )

    del model
    import gc
    gc.collect()

    print(f"    {combined.shape} saved to {output_dir}")
    return combined


def main():
    concepts = load_concepts()
    print(f"Loaded {len(concepts)} concepts")

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL))

    total_extractions = 0

    # Extract seed pairs
    for seed in SEEDS:
        seed_dir = SEED_RESULTS_DIR / f"seed-{seed}"
        if not seed_dir.exists():
            print(f"\n[SKIP] seed-{seed}: directory not found")
            continue

        for condition in CONDITIONS:
            ckpt = seed_dir / f"{condition}-ckpt.pt"
            if not ckpt.exists():
                print(f"\n[SKIP] seed-{seed}/{condition}: checkpoint not found")
                continue

            out_dir = GEOMETRY_DIR / f"seed-{seed}" / condition
            if (out_dir / "all_hidden_states.npz").exists():
                print(f"\n[SKIP] seed-{seed}/{condition}: already extracted")
                total_extractions += 1
                continue

            print(f"\n[{total_extractions+1}] Extracting seed-{seed}/{condition}")
            t0 = time.time()
            extract_checkpoint(ckpt, concepts, sp, out_dir,
                             f"seed-{seed}-{condition}")
            print(f"    {time.time()-t0:.1f}s")
            total_extractions += 1

    # Extract standalone run-2 checkpoints
    for label, ckpt in STANDALONE.items():
        if not ckpt.exists():
            continue
        out_dir = GEOMETRY_DIR / label
        if (out_dir / "all_hidden_states.npz").exists():
            print(f"\n[SKIP] {label}: already extracted")
            continue
        print(f"\n[{total_extractions+1}] Extracting {label}")
        t0 = time.time()
        extract_checkpoint(ckpt, concepts, sp, out_dir, label)
        print(f"    {time.time()-t0:.1f}s")
        total_extractions += 1

    print(f"\nDone. {total_extractions} extractions.")


if __name__ == "__main__":
    main()
