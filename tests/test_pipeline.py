import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.simulate_templates import simulate, FEATURE_DIM
from src.diffusion_model import ConditionalDiffusionModel
from src.multiobjective_scoring import validate_ranking, combined_score


def test_at_least_800_templates():
    data = simulate()
    assert len(data["sequences"]) >= 800


def test_feature_dim_matches():
    data = simulate()
    assert data["features"].shape[1] == FEATURE_DIM


def test_diffusion_training_reduces_loss():
    data = simulate(n=100)
    X0 = data["features"]
    cond = np.stack([data["true_stability"], data["true_affinity"]], axis=1).astype(np.float32)
    model = ConditionalDiffusionModel(feature_dim=FEATURE_DIM, timesteps=20)
    losses = model.train(X0, cond, epochs=60)
    assert losses[-1] < losses[0]


def test_sampling_produces_correct_shape():
    data = simulate(n=50)
    X0 = data["features"]
    cond = np.stack([data["true_stability"], data["true_affinity"]], axis=1).astype(np.float32)
    model = ConditionalDiffusionModel(feature_dim=FEATURE_DIM, timesteps=20)
    model.train(X0, cond, epochs=20)
    samples = model.sample(cond, n_samples=50)
    assert samples.shape == (50, FEATURE_DIM)


def test_ranking_validation_above_random():
    data = simulate(n=200)
    rng = np.random.default_rng(1)
    result = validate_ranking(data["features"], rng)
    assert result["spearman_rho"] > 0.3
