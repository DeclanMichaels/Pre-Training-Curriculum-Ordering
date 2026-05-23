#!/usr/bin/env python3
"""
Train Sparse Autoencoders on model hidden states.

Pipeline:
  1. Run corpus tokens through model, collect hidden states at all layers
  2. Train a sparse autoencoder per layer
  3. Save trained SAEs and feature activation statistics

Usage:
    python scripts/train_sae.py --checkpoint results/standard/sequenced/ckpt.pt --name standard-sequenced
    python scripts/train_sae.py --checkpoint results/standard/shuffled/ckpt.pt --name standard-shuffled
    python scripts/train_sae.py --checkpoint results/standard-5k-then-shuffled/ckpt_10000.pt --name standard-5k-then-shuffled

Optional:
    --n-tokens 200000       Number of tokens to collect activations for (default: 200000)
    --expansion 16          SAE expansion factor (default: 16, gives 768*16=12288 features)
    --sae-epochs 5          Training epochs for SAE (default: 5)
    --sae-lr 1e-3           SAE learning rate (default: 1e-3)
    --l1-coeff 1e-3         L1 sparsity penalty (default: 1e-3)
    --sae-batch 256         SAE training batch size (default: 256)
    --attnres               Use AttnRes model class
    --device cpu             Device (default: cpu, use mps for M-series Mac)
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
NANOGPT_DIR = PROJECT_ROOT / "nanoGPT"
sys.path.insert(0, str(NANOGPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from model import GPTConfig, GPT

TOKENIZER_MODEL = PROJECT_ROOT / "corpus" / "tokenized" / "tokenizer.model"
RESULTS_DIR = PROJECT_ROOT / "results" / "sae"


# ============================================================
# Sparse Autoencoder
# ============================================================

class SparseAutoencoder(nn.Module):
    """Simple sparse autoencoder with L1 penalty on hidden activations."""

    def __init__(self, input_dim, n_features):
        super().__init__()
        self.encoder = nn.Linear(input_dim, n_features)
        self.decoder = nn.Linear(n_features, input_dim)
        # Tie decoder bias to zero — reconstruction should be unbiased
        self.decoder.bias.data.zero_()

    def forward(self, x):
        # Encode
        hidden = F.relu(self.encoder(x))
        # Decode
        reconstructed = self.decoder(hidden)
        return reconstructed, hidden

    def get_feature_activations(self, x):
        """Return just the sparse feature activations."""
        with torch.no_grad():
            return F.relu(self.encoder(x))


# ============================================================
# Activation Collection
# ============================================================

def load_model(ckpt_path, device='cpu', use_attnres=False):
    """Load a trained nanoGPT checkpoint."""
    print(f"  Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if use_attnres:
        from model_attnres import GPTConfig as AC, GPT as AG
        config = AC(**checkpoint['model_args'])
        model = AG(config)
    else:
        config = GPTConfig(**checkpoint['model_args'])
        model = GPT(config)

    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    print(f"  Loaded: {config.n_layer} layers, {config.n_embd} hidden dim")
    return model, config


def collect_activations(model, config, data_path, n_tokens, device='cpu', block_size=2048):
    """
    Run tokens through model, collect hidden states at every layer.

    Returns dict mapping layer_index -> np.array of shape (n_tokens, hidden_dim)
    """
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL))

    # Load training data
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    total_tokens = len(data)
    n_tokens = min(n_tokens, total_tokens - block_size)

    n_layers = config.n_layer + 1  # +1 for embedding
    hidden_dim = config.n_embd

    # Pre-allocate storage
    all_activations = {layer: [] for layer in range(n_layers)}

    # Process in blocks
    n_blocks = n_tokens // block_size
    print(f"  Collecting activations: {n_blocks} blocks of {block_size} tokens ({n_tokens} total)")

    collected = 0
    t0 = time.time()

    for block_idx in range(n_blocks):
        start = block_idx * block_size
        x = torch.tensor(data[start:start + block_size].astype(np.int64),
                         dtype=torch.long, device=device).unsqueeze(0)

        # Collect hidden states via hooks
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

        # Store activations — subsample to every 4th token to manage memory
        for layer_idx, h in enumerate(hidden_states):
            # h shape: (1, seq_len, hidden_dim)
            tokens = h.squeeze(0)[::4].cpu().numpy()  # subsample
            all_activations[layer_idx].append(tokens)

        collected += block_size
        if (block_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    Block {block_idx + 1}/{n_blocks} ({collected} tokens, {elapsed:.1f}s)")

    # Concatenate
    result = {}
    for layer_idx in range(n_layers):
        result[layer_idx] = np.concatenate(all_activations[layer_idx], axis=0)
        print(f"    Layer {layer_idx}: {result[layer_idx].shape}")

    return result


# ============================================================
# SAE Training
# ============================================================

def train_sae(activations, input_dim, n_features, epochs=5, lr=1e-3,
              l1_coeff=1e-3, batch_size=256, device='cpu'):
    """Train a sparse autoencoder on activation vectors."""

    sae = SparseAutoencoder(input_dim, n_features).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    # Convert to tensor
    X = torch.tensor(activations, dtype=torch.float32)
    n_samples = X.shape[0]

    history = []

    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(n_samples)
        total_recon_loss = 0.0
        total_l1_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch = X[perm[i:i + batch_size]].to(device)

            reconstructed, hidden = sae(batch)

            recon_loss = F.mse_loss(reconstructed, batch)
            l1_loss = l1_coeff * hidden.abs().mean()
            loss = recon_loss + l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_l1_loss += l1_loss.item()
            n_batches += 1

        avg_recon = total_recon_loss / n_batches
        avg_l1 = total_l1_loss / n_batches

        # Compute sparsity stats
        with torch.no_grad():
            sample = X[:1000].to(device)
            acts = sae.get_feature_activations(sample)
            alive = (acts > 0).float().mean(dim=0)  # fraction of samples each feature fires on
            n_alive = (alive > 0.01).sum().item()  # features that fire on >1% of samples
            avg_l0 = (acts > 0).float().sum(dim=1).mean().item()  # avg active features per sample

        history.append({
            "epoch": epoch,
            "recon_loss": round(avg_recon, 6),
            "l1_loss": round(avg_l1, 6),
            "n_alive_features": int(n_alive),
            "avg_l0": round(avg_l0, 1),
        })

        print(f"      Epoch {epoch}: recon={avg_recon:.4f} l1={avg_l1:.4f} "
              f"alive={n_alive}/{n_features} avg_L0={avg_l0:.0f}")

    return sae, history


def analyze_sae(sae, activations, concepts_hidden, concepts, domains, device='cpu'):
    """Analyze trained SAE: feature statistics and concept activations."""

    # Feature activation statistics on corpus
    X = torch.tensor(activations, dtype=torch.float32).to(device)
    with torch.no_grad():
        acts = sae.get_feature_activations(X)

    acts_np = acts.cpu().numpy()

    # Per-feature statistics
    feature_stats = []
    n_features = acts_np.shape[1]
    for f in range(n_features):
        col = acts_np[:, f]
        firing_rate = float(np.mean(col > 0))
        mean_activation = float(np.mean(col[col > 0])) if np.any(col > 0) else 0.0
        feature_stats.append({
            "feature_id": f,
            "firing_rate": round(firing_rate, 6),
            "mean_activation": round(mean_activation, 6),
        })

    alive_features = sum(1 for fs in feature_stats if fs["firing_rate"] > 0.01)
    dead_features = sum(1 for fs in feature_stats if fs["firing_rate"] < 0.001)

    # Concept feature activations
    C = torch.tensor(concepts_hidden, dtype=torch.float32).to(device)
    with torch.no_grad():
        concept_acts = sae.get_feature_activations(C)
    concept_acts_np = concept_acts.cpu().numpy()

    # Top features per concept
    concept_features = []
    for i, (concept, domain) in enumerate(zip(concepts, domains)):
        top_indices = np.argsort(concept_acts_np[i])[::-1][:10]
        top_values = concept_acts_np[i][top_indices]
        concept_features.append({
            "concept": concept,
            "domain": domain,
            "top_features": [{"feature_id": int(idx), "activation": round(float(val), 4)}
                             for idx, val in zip(top_indices, top_values) if val > 0],
        })

    return {
        "n_features": n_features,
        "alive_features": alive_features,
        "dead_features": dead_features,
        "feature_stats": feature_stats,
        "concept_features": concept_features,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train SAEs on model hidden states")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--n-tokens", type=int, default=200000)
    parser.add_argument("--expansion", type=int, default=16)
    parser.add_argument("--sae-epochs", type=int, default=5)
    parser.add_argument("--sae-lr", type=float, default=1e-3)
    parser.add_argument("--l1-coeff", type=float, default=1e-3)
    parser.add_argument("--sae-batch", type=int, default=256)
    parser.add_argument("--attnres", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    output_dir = RESULTS_DIR / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\n{'='*60}")
    print(f"SAE Pipeline: {args.name}")
    print(f"{'='*60}")
    model, config = load_model(str(ckpt_path), args.device, use_attnres=args.attnres)

    hidden_dim = config.n_embd
    n_features = hidden_dim * args.expansion
    n_layers = config.n_layer + 1

    print(f"  Hidden dim: {hidden_dim}")
    print(f"  SAE features: {n_features} ({args.expansion}x expansion)")
    print(f"  Layers: {n_layers}")

    # Find training data
    data_path = PROJECT_ROOT / "nanoGPT" / "data" / "classical_sequenced" / "train.bin"
    if not data_path.exists():
        print(f"ERROR: Training data not found: {data_path}")
        sys.exit(1)

    # Collect activations
    print(f"\nCollecting activations ({args.n_tokens} tokens)...")
    t0 = time.time()
    activations = collect_activations(model, config, str(data_path),
                                      args.n_tokens, device=args.device)
    print(f"  Collection complete: {time.time() - t0:.1f}s")

    # Load concept inventory for feature analysis
    concepts_file = PROJECT_ROOT / "config" / "concepts.json"
    with open(concepts_file) as f:
        concept_data = json.load(f)
    concepts_list = []
    for domain, items in concept_data["concepts"].items():
        for concept in items:
            concepts_list.append({"concept": concept, "domain": domain})

    # Load concept hidden states from geometry extraction
    geometry_dir = PROJECT_ROOT / "results" / "geometry" / args.name
    concept_hidden_path = geometry_dir / "all_hidden_states.npz"
    if concept_hidden_path.exists():
        concept_data_npz = np.load(concept_hidden_path, allow_pickle=True)
        concept_hidden = concept_data_npz["hidden_states"]
        print(f"  Loaded concept hidden states: {concept_hidden.shape}")
    else:
        print(f"  WARNING: No concept hidden states at {concept_hidden_path}")
        concept_hidden = None

    # Free the main model from memory
    del model
    if args.device == "mps":
        torch.mps.empty_cache()

    # Train SAE per layer
    all_layer_results = {}

    for layer in range(n_layers):
        label = "embed" if layer == 0 else f"L{layer}"
        print(f"\n  {'='*50}")
        print(f"  Layer {label}: training SAE ({n_features} features)")
        print(f"  {'='*50}")

        layer_acts = activations[layer].astype(np.float32)

        sae, history = train_sae(
            layer_acts, hidden_dim, n_features,
            epochs=args.sae_epochs, lr=args.sae_lr,
            l1_coeff=args.l1_coeff, batch_size=args.sae_batch,
            device=args.device,
        )

        # Analyze
        if concept_hidden is not None:
            concept_layer = concept_hidden[:, layer, :].astype(np.float32)
            analysis = analyze_sae(
                sae, layer_acts, concept_layer,
                [c["concept"] for c in concepts_list],
                [c["domain"] for c in concepts_list],
                device=args.device,
            )
        else:
            analysis = {"n_features": n_features, "alive_features": 0, "dead_features": n_features}

        # Save SAE weights
        torch.save(sae.state_dict(), output_dir / f"sae_layer_{layer:02d}.pt")

        # Save analysis
        layer_result = {
            "layer": layer,
            "label": label,
            "training_history": history,
            "analysis": {
                "alive_features": analysis["alive_features"],
                "dead_features": analysis["dead_features"],
                "n_features": analysis["n_features"],
            },
        }

        # Save concept features separately (large)
        if "concept_features" in analysis:
            with open(output_dir / f"concept_features_layer_{layer:02d}.json", "w") as f:
                json.dump(analysis["concept_features"], f, indent=2)

        # Save full feature stats separately (large)
        if "feature_stats" in analysis:
            with open(output_dir / f"feature_stats_layer_{layer:02d}.json", "w") as f:
                json.dump(analysis["feature_stats"], f)

        all_layer_results[label] = layer_result
        print(f"    Alive: {analysis['alive_features']}/{n_features} "
              f"Dead: {analysis['dead_features']}/{n_features}")

    # Save summary
    summary = {
        "model": args.name,
        "checkpoint": str(ckpt_path),
        "n_tokens": args.n_tokens,
        "expansion": args.expansion,
        "n_features": n_features,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "sae_epochs": args.sae_epochs,
        "l1_coeff": args.l1_coeff,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "layers": all_layer_results,
    }

    with open(output_dir / "sae_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SAE training complete: {args.name}")
    print(f"Results: {output_dir}")
    print(f"{'='*60}")

    # Print alive feature comparison
    print(f"\n{'Layer':<8} {'Alive':<10} {'Dead':<10}")
    print("-" * 28)
    for label, result in all_layer_results.items():
        a = result["analysis"]["alive_features"]
        d = result["analysis"]["dead_features"]
        print(f"{label:<8} {a:<10} {d:<10}")


if __name__ == "__main__":
    main()
