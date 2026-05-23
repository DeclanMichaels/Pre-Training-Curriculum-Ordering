#!/usr/bin/env python3
"""
Analyze representational geometry from extracted hidden states.

Computes:
  - Cross-condition CKA at every layer (H1)
  - Within-domain silhouette scores for Stage 0/1 concepts (H2)
  - Three-domain silhouette scores (H3)
  - Permutation test for CKA significance
  - Bootstrap CIs for all metrics

Usage:
    python scripts/analyze_geometry.py
    python scripts/analyze_geometry.py --n-bootstrap 10000

Requires: extraction results in results/geometry/{sequenced,shuffled}/
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results" / "geometry"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "analysis"


# ============================================================
# CKA
# ============================================================

def linear_cka(X, Y):
    """
    Linear CKA between matrices X (n, d1) and Y (n, d2).
    Measures similarity of representational geometry independent of
    dimensionality and rotation.
    """
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


# ============================================================
# Silhouette score
# ============================================================

def silhouette_score(X, labels):
    """
    Silhouette score for clustering quality.
    X: (n, d) feature matrix
    labels: (n,) integer cluster labels

    Returns mean silhouette across all samples. Range [-1, 1].
    Higher = tighter, better-separated clusters.

    Uses cosine distance. Implemented with numpy (no scipy dependency).
    """
    n = len(labels)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return 0.0

    # Cosine distance matrix: 1 - cosine_similarity
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # avoid division by zero
    X_norm = X / norms
    cos_sim = X_norm @ X_norm.T
    dists = 1.0 - cos_sim

    silhouettes = np.zeros(n)
    for i in range(n):
        own_label = labels[i]
        own_mask = labels == own_label
        other_labels = unique_labels[unique_labels != own_label]

        # a(i): mean distance to same-cluster points
        own_dists = dists[i, own_mask]
        if own_mask.sum() > 1:
            a_i = own_dists.sum() / (own_mask.sum() - 1)
        else:
            a_i = 0.0

        # b(i): min mean distance to any other cluster
        b_i = np.inf
        for other in other_labels:
            other_mask = labels == other
            mean_dist = dists[i, other_mask].mean()
            b_i = min(b_i, mean_dist)

        if max(a_i, b_i) > 0:
            silhouettes[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouettes[i] = 0.0

    return float(np.mean(silhouettes))


# ============================================================
# Loading
# ============================================================

def load_hidden_states(model_name):
    """Load combined hidden states array for a model."""
    path = RESULTS_DIR / model_name / "all_hidden_states.npz"
    if not path.exists():
        raise FileNotFoundError(f"No extraction found: {path}")

    data = np.load(path, allow_pickle=True)
    hidden = data["hidden_states"]  # (n_concepts, n_layers+1, hidden_dim)
    concepts = list(data["concepts"])
    domains = list(data["domains"])

    print(f"  {model_name}: {hidden.shape}")
    return hidden, concepts, domains


def get_domain_labels(domains):
    """Convert domain strings to integer labels."""
    unique = sorted(set(domains))
    label_map = {d: i for i, d in enumerate(unique)}
    return np.array([label_map[d] for d in domains]), unique


# ============================================================
# Analysis
# ============================================================

def compute_layer_cka(hidden_a, hidden_b, n_layers):
    """Compute CKA at each layer between two models."""
    results = []
    for layer in range(n_layers):
        X = hidden_a[:, layer, :]
        Y = hidden_b[:, layer, :]
        cka = linear_cka(X, Y)
        results.append({"layer": layer, "cka": round(cka, 6)})
    return results


def permutation_test_cka(hidden_a, hidden_b, layer, n_perms=10000, seed=42):
    """
    Permutation test: is the observed CKA lower than expected by chance?
    Shuffle concept labels in one model, recompute CKA.
    """
    rng = np.random.default_rng(seed)
    X = hidden_a[:, layer, :]
    Y = hidden_b[:, layer, :]

    observed_cka = linear_cka(X, Y)

    null_ckas = np.empty(n_perms)
    n = X.shape[0]
    for i in range(n_perms):
        perm = rng.permutation(n)
        null_ckas[i] = linear_cka(X[perm], Y)

    p_value = float(np.mean(null_ckas >= observed_cka))

    return {
        "observed_cka": round(observed_cka, 6),
        "null_mean": round(float(np.mean(null_ckas)), 6),
        "null_std": round(float(np.std(null_ckas)), 6),
        "p_value": round(p_value, 6),
        "p_interpretation": "Fraction of null permutations with CKA >= observed. Low p means observed structure is real, not noise.",
        "n_perms": n_perms,
    }


def bootstrap_cka(hidden_a, hidden_b, layer, n_bootstrap=10000, seed=42):
    """Bootstrap 95% CI for CKA at a specific layer."""
    rng = np.random.default_rng(seed)
    X = hidden_a[:, layer, :]
    Y = hidden_b[:, layer, :]
    n = X.shape[0]

    point = linear_cka(X, Y)

    boot_ckas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_ckas[i] = linear_cka(X[idx], Y[idx])

    ci_low, ci_high = np.percentile(boot_ckas, [2.5, 97.5])

    return {
        "cka": round(point, 6),
        "ci_95_low": round(float(ci_low), 6),
        "ci_95_high": round(float(ci_high), 6),
        "ci_width": round(float(ci_high - ci_low), 6),
        "boot_mean": round(float(np.mean(boot_ckas)), 6),
        "boot_std": round(float(np.std(boot_ckas)), 6),
    }


def compute_silhouette_by_layer(hidden, labels, n_layers):
    """Compute silhouette score at each layer."""
    results = []
    for layer in range(n_layers):
        X = hidden[:, layer, :]
        score = silhouette_score(X, labels)
        results.append({"layer": layer, "silhouette": round(score, 6)})
    return results


def bootstrap_silhouette(hidden, labels, layer, n_bootstrap=10000, seed=42):
    """Bootstrap 95% CI for silhouette score."""
    rng = np.random.default_rng(seed)
    X = hidden[:, layer, :]
    n = X.shape[0]

    point = silhouette_score(X, labels)

    boot_scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_scores[i] = silhouette_score(X[idx], labels[idx])

    ci_low, ci_high = np.percentile(boot_scores, [2.5, 97.5])

    return {
        "silhouette": round(point, 6),
        "ci_95_low": round(float(ci_low), 6),
        "ci_95_high": round(float(ci_high), 6),
        "boot_mean": round(float(np.mean(boot_scores)), 6),
        "boot_std": round(float(np.std(boot_scores)), 6),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze representational geometry")
    parser.add_argument("--model-a", type=str, default="sequenced",
                        help="First model name under results/geometry/ (default: sequenced)")
    parser.add_argument("--model-b", type=str, default="shuffled",
                        help="Second model name under results/geometry/ (default: shuffled)")
    parser.add_argument("--output", type=str, default="geometry_analysis.json",
                        help="Output filename in results/analysis/ (default: geometry_analysis.json)")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--n-perms", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load both models
    print("Loading hidden states...")
    hidden_seq, concepts_seq, domains_seq = load_hidden_states(args.model_a)
    hidden_shuf, concepts_shuf, domains_shuf = load_hidden_states(args.model_b)

    # Verify alignment
    assert concepts_seq == concepts_shuf, "Concept lists don't match"
    assert domains_seq == domains_shuf, "Domain lists don't match"

    concepts = concepts_seq
    domains = domains_seq
    n_concepts = len(concepts)
    n_layers = hidden_seq.shape[1]  # includes embedding layer
    hidden_dim = hidden_seq.shape[2]

    domain_labels, domain_names = get_domain_labels(domains)

    print(f"\n{n_concepts} concepts, {n_layers} layers (including embedding), {hidden_dim} hidden dim")
    print(f"Domains: {domain_names}")

    # ---- H1: Cross-condition CKA ----
    print(f"\n{'='*60}")
    print(f"H1: Cross-condition CKA ({args.model_a} vs {args.model_b})")
    print(f"{'='*60}")

    layer_cka = compute_layer_cka(hidden_seq, hidden_shuf, n_layers)
    cka_values = [r["cka"] for r in layer_cka]
    mean_cka = np.mean(cka_values)
    below_090 = sum(1 for v in cka_values if v < 0.90)

    print(f"\n  Per-layer CKA:")
    for r in layer_cka:
        label = "embed" if r["layer"] == 0 else f"L{r['layer']:2d}"
        bar = "#" * int(r["cka"] * 50)
        flag = " <-- below 0.90" if r["cka"] < 0.90 else ""
        print(f"    {label}: {r['cka']:.4f}  {bar}{flag}")

    print(f"\n  Mean CKA: {mean_cka:.4f}")
    print(f"  Layers below 0.90: {below_090}/{n_layers}")
    print(f"  H1 threshold: CKA < 0.90 at majority of layers")
    print(f"  H1 {'CONFIRMED' if below_090 > n_layers / 2 else 'NOT CONFIRMED'}")

    # Permutation test at mid-network layer
    mid_layer = n_layers // 2
    print(f"\n  Permutation test at layer {mid_layer} ({args.n_perms} permutations)...")
    perm_result = permutation_test_cka(hidden_seq, hidden_shuf, mid_layer,
                                        n_perms=args.n_perms, seed=args.seed)
    print(f"    Observed CKA: {perm_result['observed_cka']:.4f}")
    print(f"    Null mean:    {perm_result['null_mean']:.4f} +/- {perm_result['null_std']:.4f}")
    print(f"    p-value:      {perm_result['p_value']:.4f}")

    # Bootstrap CIs at key layers
    print(f"\n  Bootstrap CIs ({args.n_bootstrap} iterations)...")
    boot_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    boot_results = {}
    for layer in boot_layers:
        label = "embed" if layer == 0 else f"L{layer}"
        result = bootstrap_cka(hidden_seq, hidden_shuf, layer,
                               n_bootstrap=args.n_bootstrap, seed=args.seed)
        boot_results[label] = result
        print(f"    {label}: CKA={result['cka']:.4f}  "
              f"95% CI=[{result['ci_95_low']:.4f}, {result['ci_95_high']:.4f}]")

    # ---- H2: Foundational retention ----
    print(f"\n{'='*60}")
    print("H2: Foundational retention (physical + moral domain concepts)")
    print(f"{'='*60}")

    # Physical domain = Stage 0 proxy, Moral domain = Stage 1 proxy
    phys_mask = np.array([d == "physical" for d in domains])
    moral_mask = np.array([d == "moral" for d in domains])
    foundation_mask = phys_mask | moral_mask

    print(f"\n  Physical concepts: {phys_mask.sum()}")
    print(f"  Moral concepts: {moral_mask.sum()}")
    print(f"  Foundation (combined): {foundation_mask.sum()}")

    foundation_labels = domain_labels[foundation_mask]

    print(f"\n  Silhouette scores (foundation concepts only):")
    seq_sil = compute_silhouette_by_layer(hidden_seq[foundation_mask], foundation_labels, n_layers)
    shuf_sil = compute_silhouette_by_layer(hidden_shuf[foundation_mask], foundation_labels, n_layers)

    h2_seq_better = 0
    for s, sh in zip(seq_sil, shuf_sil):
        label = "embed" if s["layer"] == 0 else f"L{s['layer']:2d}"
        diff = s["silhouette"] - sh["silhouette"]
        winner = "SEQ" if diff > 0 else "SHUF"
        if diff > 0:
            h2_seq_better += 1
        print(f"    {label}: seq={s['silhouette']:+.4f}  shuf={sh['silhouette']:+.4f}  diff={diff:+.4f}  ({winner})")

    print(f"\n  Sequenced better at {h2_seq_better}/{n_layers} layers")

    # Bootstrap at mid-layer
    mid = n_layers // 2
    print(f"\n  Bootstrap silhouette at layer {mid}...")
    boot_seq = bootstrap_silhouette(hidden_seq[foundation_mask], foundation_labels, mid,
                                    n_bootstrap=args.n_bootstrap, seed=args.seed)
    boot_shuf = bootstrap_silhouette(hidden_shuf[foundation_mask], foundation_labels, mid,
                                     n_bootstrap=args.n_bootstrap, seed=args.seed)
    overlap = boot_seq["ci_95_low"] < boot_shuf["ci_95_high"] and boot_shuf["ci_95_low"] < boot_seq["ci_95_high"]
    print(f"    Sequenced: {boot_seq['silhouette']:.4f}  CI=[{boot_seq['ci_95_low']:.4f}, {boot_seq['ci_95_high']:.4f}]")
    print(f"    Shuffled:  {boot_shuf['silhouette']:.4f}  CI=[{boot_shuf['ci_95_low']:.4f}, {boot_shuf['ci_95_high']:.4f}]")
    print(f"    CIs overlap: {overlap}")

    # ---- H3: Domain clustering ----
    print(f"\n{'='*60}")
    print("H3: Three-domain clustering (physical, institutional, moral)")
    print(f"{'='*60}")

    print(f"\n  Silhouette scores (all concepts, 3-domain partition):")
    seq_3d = compute_silhouette_by_layer(hidden_seq, domain_labels, n_layers)
    shuf_3d = compute_silhouette_by_layer(hidden_shuf, domain_labels, n_layers)

    h3_seq_better = 0
    for s, sh in zip(seq_3d, shuf_3d):
        label = "embed" if s["layer"] == 0 else f"L{s['layer']:2d}"
        diff = s["silhouette"] - sh["silhouette"]
        winner = "SEQ" if diff > 0 else "SHUF"
        if diff > 0:
            h3_seq_better += 1
        print(f"    {label}: seq={s['silhouette']:+.4f}  shuf={sh['silhouette']:+.4f}  diff={diff:+.4f}  ({winner})")

    print(f"\n  Sequenced better at {h3_seq_better}/{n_layers} layers")

    # Bootstrap at mid-layer
    print(f"\n  Bootstrap silhouette (3-domain) at layer {mid}...")
    boot_3d_seq = bootstrap_silhouette(hidden_seq, domain_labels, mid,
                                       n_bootstrap=args.n_bootstrap, seed=args.seed)
    boot_3d_shuf = bootstrap_silhouette(hidden_shuf, domain_labels, mid,
                                        n_bootstrap=args.n_bootstrap, seed=args.seed)
    overlap_3d = boot_3d_seq["ci_95_low"] < boot_3d_shuf["ci_95_high"] and boot_3d_shuf["ci_95_low"] < boot_3d_seq["ci_95_high"]
    print(f"    Sequenced: {boot_3d_seq['silhouette']:.4f}  CI=[{boot_3d_seq['ci_95_low']:.4f}, {boot_3d_seq['ci_95_high']:.4f}]")
    print(f"    Shuffled:  {boot_3d_shuf['silhouette']:.4f}  CI=[{boot_3d_shuf['ci_95_low']:.4f}, {boot_3d_shuf['ci_95_high']:.4f}]")
    print(f"    CIs overlap: {overlap_3d}")

    # ---- Save all results ----
    print(f"\n{'='*60}")
    print("Saving results")
    print(f"{'='*60}")

    all_results = {
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "n_concepts": n_concepts,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "domains": domain_names,
        "n_bootstrap": args.n_bootstrap,
        "n_perms": args.n_perms,
        "seed": args.seed,
        "h1_cross_condition_cka": {
            "per_layer": layer_cka,
            "mean_cka": round(mean_cka, 6),
            "layers_below_090": below_090,
            "total_layers": n_layers,
            "confirmed": below_090 > n_layers / 2,
            "permutation_test": perm_result,
            "bootstrap_cis": boot_results,
        },
        "h2_foundational_retention": {
            "sequenced_silhouette": seq_sil,
            "shuffled_silhouette": shuf_sil,
            "sequenced_better_count": h2_seq_better,
            "bootstrap_mid_layer": {
                "layer": mid,
                "sequenced": boot_seq,
                "shuffled": boot_shuf,
                "cis_overlap": overlap,
            },
        },
        "h3_domain_clustering": {
            "sequenced_silhouette": seq_3d,
            "shuffled_silhouette": shuf_3d,
            "sequenced_better_count": h3_seq_better,
            "bootstrap_mid_layer": {
                "layer": mid,
                "sequenced": boot_3d_seq,
                "shuffled": boot_3d_shuf,
                "cis_overlap": overlap_3d,
            },
        },
    }

    output_path = OUTPUT_DIR / args.output
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {output_path}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n  H1 (geometry differs):       {'CONFIRMED' if all_results['h1_cross_condition_cka']['confirmed'] else 'NOT CONFIRMED'}")
    print(f"      Mean CKA: {mean_cka:.4f}, {below_090}/{n_layers} layers below 0.90")
    print(f"      Permutation p-value: {perm_result['p_value']}")
    print(f"\n  H2 (foundational retention):  Sequenced better at {h2_seq_better}/{n_layers} layers")
    print(f"      Mid-layer CIs overlap: {overlap}")
    print(f"\n  H3 (domain clustering):       Sequenced better at {h3_seq_better}/{n_layers} layers")
    print(f"      Mid-layer CIs overlap: {overlap_3d}")


if __name__ == "__main__":
    main()
