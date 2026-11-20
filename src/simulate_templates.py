"""Simulate 800+ peptide structural templates represented as amino-acid
composition + physicochemical feature vectors (distinct from
ProteinDesignML's generic structural-template features)."""
from __future__ import annotations

import numpy as np

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBICITY = {  # Kyte-Doolittle scale
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
    "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
    "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

N_TEMPLATES = 820
SEED = 505
FEATURE_DIM = 20 + 3  # 20 AA composition fractions + length, mean hydrophobicity, net charge


def _random_peptide(rng: np.random.Generator, length: int) -> str:
    return "".join(rng.choice(AMINO_ACIDS, length))


def _composition_features(seq: str) -> np.ndarray:
    counts = np.array([seq.count(a) for a in AMINO_ACIDS], dtype=float)
    comp = counts / len(seq)
    length_norm = len(seq) / 30.0
    mean_hydro = np.mean([HYDROPHOBICITY[c] for c in seq]) / 5.0
    charge = sum(1 for c in seq if c in "KR") - sum(1 for c in seq if c in "DE")
    charge_norm = charge / len(seq)
    return np.concatenate([comp, [length_norm, mean_hydro, charge_norm]])


def simulate(n: int = N_TEMPLATES, seed: int = SEED):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(10, 25, n)
    sequences = [_random_peptide(rng, L) for L in lengths]
    features = np.stack([_composition_features(s) for s in sequences])

    # hidden true stability/affinity targets, driven by composition independent
    # of the generative model: hydrophobic moment + charge balance heuristic
    hydro_col = features[:, 20 + 1]
    charge_col = features[:, 20 + 2]
    true_stability = 2.0 * hydro_col - 1.5 * charge_col ** 2 + rng.normal(0, 0.15, n)
    true_affinity = -1.2 * charge_col + 0.8 * hydro_col ** 2 + rng.normal(0, 0.15, n)

    return {
        "sequences": sequences,
        "features": features.astype(np.float32),
        "true_stability": true_stability.astype(np.float32),
        "true_affinity": true_affinity.astype(np.float32),
    }
