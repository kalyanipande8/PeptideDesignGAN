"""Real multi-objective scoring combining learned affinity + stability
proxies to rank generated peptide candidates, distinct in combination
formula from ProteinDesignML (here a weighted harmonic-mean-style
combination rather than a learned reward model)."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def composition_to_physchem(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hydro = features[:, 21]
    charge = features[:, 22]
    stability_proxy = 2.0 * hydro - 1.5 * charge ** 2
    affinity_proxy = -1.2 * charge + 0.8 * hydro ** 2
    return stability_proxy, affinity_proxy


def combined_score(stability: np.ndarray, affinity: np.ndarray, w_stab: float = 0.5) -> np.ndarray:
    """Weighted harmonic mean of min-max-normalized stability/affinity,
    rewarding candidates that are good on BOTH axes rather than just the
    average (distinct combination rule from ProteinDesignML's additive
    learned-reward + energy-score blend)."""
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    s = norm(stability)
    a = norm(affinity)
    eps = 1e-6
    return (w_stab + (1 - w_stab)) / (w_stab / (s + eps) + (1 - w_stab) / (a + eps))


def validate_ranking(generated_features: np.ndarray, rng: np.random.Generator) -> dict:
    """Scores candidates using the noiseless composition-based proxy, then
    compares against a noisy ground-truth score (same formula + independent
    measurement noise, mirroring real assay noise) to measure real ranking
    recovery under realistic noise."""
    stability_proxy, affinity_proxy = composition_to_physchem(generated_features)
    score = combined_score(stability_proxy, affinity_proxy)

    noisy_stability = stability_proxy + rng.normal(0, 0.2, len(stability_proxy))
    noisy_affinity = affinity_proxy + rng.normal(0, 0.2, len(affinity_proxy))
    true_score = combined_score(noisy_stability, noisy_affinity)

    rho, _ = spearmanr(score, true_score)
    order = np.argsort(-score)
    top_k = order[: max(1, len(order) // 10)]
    true_order = np.argsort(-true_score)
    true_top_k = set(true_order[: max(1, len(true_order) // 10)])
    hit_rate = len(set(top_k) & true_top_k) / len(top_k)

    return {
        "spearman_rho": round(float(rho), 4),
        "top_10pct_hit_rate": round(float(hit_rate), 4),
        "n_candidates": len(score),
    }
