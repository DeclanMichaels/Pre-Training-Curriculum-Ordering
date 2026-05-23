#!/usr/bin/env python3
"""
Compute effective dimensionality for any extracted models.

Usage:
    python scripts/compute_dimensionality_flex.py model1 model2 [model3 ...]
    python scripts/compute_dimensionality_flex.py standard-sequenced standard-shuffled standard-5k-then-shuffled
"""

import json
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
GEOMETRY_DIR = PROJECT_ROOT / "results" / "geometry"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"


def effective_dim(X, threshold=0.90):
    """Number of singular values needed to explain threshold% of variance."""
    X_centered = X - X.mean(axis=0)
    s = np.linalg.svd(X_centered, compute_uv=False)
    cumvar = np.cumsum(s**2) / np.sum(s**2)
    return int(np.searchsorted(cumvar, threshold) + 1)


def participation_ratio(X):
    """Participation ratio: (sum(eigenvalues))^2 / sum(eigenvalues^2).
    Smoother measure of effective dimensionality. Range [1, n_dims]."""
    X_centered = X - X.mean(axis=0)
    s = np.linalg.svd(X_centered, compute_uv=False)
    eigenvalues = s**2
    return float(np.sum(eigenvalues)**2 / np.sum(eigenvalues**2))


def analyze_model(model_name):
    """Compute dimensionality at every layer for a model."""
    npz_path = GEOMETRY_DIR / model_name / "all_hidden_states.npz"
    if not npz_path.exists():
        print(f"  ERROR: {npz_path} not found")
        return None

    data = np.load(npz_path, allow_pickle=True)
    hidden = data["hidden_states"]  # (n_concepts, n_layers, hidden_dim)
    n_layers = hidden.shape[1]

    results = []
    for layer in range(n_layers):
        X = hidden[:, layer, :]
        edim = effective_dim(X, 0.90)
        pr = participation_ratio(X)
        results.append({
            "layer": layer,
            "effective_dim_90": edim,
            "participation_ratio": round(pr, 2),
        })

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/compute_dimensionality_flex.py model1 model2 ...")
        print("Model names are subdirectory names under results/geometry/")
        sys.exit(1)

    model_names = sys.argv[1:]
    all_results = {}

    print(f"Computing dimensionality for {len(model_names)} models...\n")

    for name in model_names:
        print(f"  {name}:")
        results = analyze_model(name)
        if results is None:
            continue
        all_results[name] = results
        dims = [r["effective_dim_90"] for r in results]
        print(f"    Dims: {' → '.join(str(d) for d in dims)}")
        print(f"    Range: {min(dims)}-{max(dims)}")
        print()

    # Print comparison table
    print(f"\n{'Layer':<8}", end="")
    for name in all_results:
        short = name[:20]
        print(f"{short:<22}", end="")
    print()
    print("-" * (8 + 22 * len(all_results)))

    n_layers = len(next(iter(all_results.values())))
    for layer in range(n_layers):
        label = "embed" if layer == 0 else f"L{layer}"
        print(f"{label:<8}", end="")
        for name in all_results:
            edim = all_results[name][layer]["effective_dim_90"]
            pr = all_results[name][layer]["participation_ratio"]
            print(f"{edim:>3} (PR {pr:>5.1f})       ", end="")
        print()

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "dimensionality_comparison.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
