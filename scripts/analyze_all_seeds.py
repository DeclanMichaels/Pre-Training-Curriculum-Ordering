#!/usr/bin/env python3
"""
Analyze CKA across all seeds and produce viewer-ready JSON.

Usage:
    python scripts/analyze_all_seeds.py

Reads from results/geometry/seed-{N}/{sequenced,shuffled}/all_hidden_states.npz
Outputs results/analysis/multi_seed_analysis.json and results/analysis/viewer_data.json
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
GEOMETRY_DIR = PROJECT_ROOT / "results" / "geometry"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"

SEEDS = [1337, 2024, 4242, 7777, 9999]
CONDITIONS = ["sequenced", "shuffled"]


def linear_cka(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    XX = X @ X.T
    YY = Y @ Y.T
    hsic_xy = np.sum(XX * YY)
    hsic_xx = np.sum(XX * XX)
    hsic_yy = np.sum(YY * YY)
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)


def silhouette_score(X, labels):
    n = len(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    X_norm = X / norms
    cos_sim = X_norm @ X_norm.T
    dists = 1.0 - cos_sim

    silhouettes = np.zeros(n)
    for i in range(n):
        own_label = labels[i]
        own_mask = labels == own_label
        other_labels = unique_labels[unique_labels != own_label]
        own_dists = dists[i, own_mask]
        if own_mask.sum() > 1:
            a_i = own_dists.sum() / (own_mask.sum() - 1)
        else:
            a_i = 0.0
        b_i = np.inf
        for other in other_labels:
            other_mask = labels == other
            mean_dist = dists[i, other_mask].mean()
            b_i = min(b_i, mean_dist)
        if max(a_i, b_i) > 0:
            silhouettes[i] = (b_i - a_i) / max(a_i, b_i)
    return float(np.mean(silhouettes))


def mds_3d(similarity_matrix):
    """Classical MDS to 3 dimensions from a distance matrix."""
    n = similarity_matrix.shape[0]
    # Convert similarity to distance
    D = 1.0 - similarity_matrix
    D = np.maximum(D, 0)

    # Double centering
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D ** 2) @ H

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take top 3
    pos_mask = eigenvalues[:3] > 0
    coords = np.zeros((n, 3))
    for i in range(3):
        if i < len(eigenvalues) and eigenvalues[i] > 0:
            coords[:, i] = eigenvectors[:, i] * np.sqrt(eigenvalues[i])

    return coords


def load_hidden(seed, condition):
    path = GEOMETRY_DIR / f"seed-{seed}" / condition / "all_hidden_states.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "hidden": data["hidden_states"],
        "concepts": list(data["concepts"]),
        "domains": list(data["domains"]),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all extractions
    print("Loading hidden states...")
    all_data = {}
    for seed in SEEDS:
        for cond in CONDITIONS:
            data = load_hidden(seed, cond)
            if data:
                all_data[(seed, cond)] = data
                print(f"  seed-{seed}/{cond}: {data['hidden'].shape}")
            else:
                print(f"  seed-{seed}/{cond}: MISSING")

    if not all_data:
        print("ERROR: No extractions found.")
        return

    # Get concept info from first available
    first = next(iter(all_data.values()))
    concepts = first["concepts"]
    domains = first["domains"]
    n_concepts = len(concepts)
    n_layers = first["hidden"].shape[1]
    hidden_dim = first["hidden"].shape[2]

    unique_domains = sorted(set(domains))
    domain_labels = np.array([unique_domains.index(d) for d in domains])

    print(f"\n{n_concepts} concepts, {n_layers} layers, {hidden_dim} dims")
    print(f"Domains: {unique_domains}")
    print(f"Loaded {len(all_data)} extractions")

    # ---- Per-seed CKA (sequenced vs shuffled) ----
    print(f"\n{'='*60}")
    print("Cross-condition CKA per seed")
    print(f"{'='*60}")

    seed_cka = {}
    for seed in SEEDS:
        seq_key = (seed, "sequenced")
        shuf_key = (seed, "shuffled")
        if seq_key not in all_data or shuf_key not in all_data:
            print(f"  seed-{seed}: incomplete, skipping")
            continue

        seq_h = all_data[seq_key]["hidden"]
        shuf_h = all_data[shuf_key]["hidden"]

        layer_ckas = []
        for layer in range(n_layers):
            cka = linear_cka(seq_h[:, layer, :], shuf_h[:, layer, :])
            layer_ckas.append(cka)

        seed_cka[seed] = layer_ckas
        mean = np.mean(layer_ckas)
        below90 = sum(1 for v in layer_ckas if v < 0.90)
        print(f"  seed-{seed}: mean CKA={mean:.4f}, {below90}/{n_layers} below 0.90")

    # Aggregate across seeds
    if seed_cka:
        all_ckas = np.array(list(seed_cka.values()))  # (n_seeds, n_layers)
        mean_per_layer = all_ckas.mean(axis=0)
        std_per_layer = all_ckas.std(axis=0)

        print(f"\n  Mean across seeds per layer:")
        for layer in range(n_layers):
            label = "embed" if layer == 0 else f"L{layer:2d}"
            print(f"    {label}: {mean_per_layer[layer]:.4f} +/- {std_per_layer[layer]:.4f}")
        print(f"  Grand mean: {mean_per_layer.mean():.4f}")

    # ---- Per-seed silhouette (3-domain) ----
    print(f"\n{'='*60}")
    print("Domain silhouette per seed")
    print(f"{'='*60}")

    seed_silhouette = {}
    for seed in SEEDS:
        for cond in CONDITIONS:
            key = (seed, cond)
            if key not in all_data:
                continue
            h = all_data[key]["hidden"]
            sils = []
            for layer in range(n_layers):
                sils.append(silhouette_score(h[:, layer, :], domain_labels))
            seed_silhouette[(seed, cond)] = sils

    # Print summary
    for seed in SEEDS:
        seq_sils = seed_silhouette.get((seed, "sequenced"))
        shuf_sils = seed_silhouette.get((seed, "shuffled"))
        if seq_sils and shuf_sils:
            seq_mean = np.mean(seq_sils)
            shuf_mean = np.mean(shuf_sils)
            seq_wins = sum(1 for s, sh in zip(seq_sils, shuf_sils) if s > sh)
            print(f"  seed-{seed}: seq mean={seq_mean:.4f}, shuf mean={shuf_mean:.4f}, seq wins {seq_wins}/{n_layers}")

    # ---- Build viewer data ----
    print(f"\n{'='*60}")
    print("Building viewer data")
    print(f"{'='*60}")

    # For each seed+condition+layer, compute:
    # 1. Cosine similarity matrix
    # 2. MDS 3D coordinates
    viewer_models = []

    for seed in SEEDS:
        for cond in CONDITIONS:
            key = (seed, cond)
            if key not in all_data:
                continue

            h = all_data[key]["hidden"]
            model_layers = []

            for layer in range(n_layers):
                X = h[:, layer, :]
                # Cosine similarity
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                X_norm = X / norms
                sim = (X_norm @ X_norm.T).tolist()

                # MDS coordinates
                sim_matrix = np.array(sim)
                coords = mds_3d(sim_matrix)

                model_layers.append({
                    "layer": layer,
                    "layer_label": "embed" if layer == 0 else f"L{layer}",
                    "coords": coords.tolist(),
                    "silhouette": silhouette_score(X, domain_labels),
                })

            viewer_models.append({
                "seed": seed,
                "condition": cond,
                "label": f"Seed {seed} ({cond})",
                "layers": model_layers,
            })

            print(f"  seed-{seed}/{cond}: {n_layers} layers projected")

    viewer_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "concepts": concepts,
        "domains": domains,
        "unique_domains": unique_domains,
        "n_layers": n_layers,
        "domain_colors": {
            "physical": "#4ECDC4",
            "institutional": "#FFE66D",
            "moral": "#FF6B6B",
        },
        "models": viewer_models,
        "cka_per_seed": {str(s): ckas for s, ckas in seed_cka.items()},
        "cka_mean_per_layer": mean_per_layer.tolist() if seed_cka else [],
        "cka_std_per_layer": std_per_layer.tolist() if seed_cka else [],
    }

    viewer_path = OUTPUT_DIR / "viewer_data.json"
    with open(viewer_path, "w") as f:
        json.dump(viewer_data, f)
    print(f"\nViewer data: {viewer_path} ({viewer_path.stat().st_size // 1024} KB)")

    # ---- Save full analysis ----
    analysis = {
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "n_seeds": len(seed_cka),
        "seeds": SEEDS,
        "n_concepts": n_concepts,
        "n_layers": n_layers,
        "cka_per_seed_per_layer": {str(s): ckas for s, ckas in seed_cka.items()},
        "cka_mean_per_layer": mean_per_layer.tolist() if seed_cka else [],
        "cka_std_per_layer": std_per_layer.tolist() if seed_cka else [],
        "cka_grand_mean": float(mean_per_layer.mean()) if seed_cka else None,
        "silhouette_per_seed_condition_layer": {
            f"{s}-{c}": sils for (s, c), sils in seed_silhouette.items()
        },
    }

    analysis_path = OUTPUT_DIR / "multi_seed_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis: {analysis_path}")

    print(f"\nDone. Open the viewer with the generated viewer_data.json.")


if __name__ == "__main__":
    main()
