import json
import os

import numpy as np

from src.simulate_templates import simulate, FEATURE_DIM
from src.diffusion_model import ConditionalDiffusionModel
from src.multiobjective_scoring import validate_ranking


def main():
    data = simulate()
    X0 = data["features"]
    cond = np.stack([data["true_stability"], data["true_affinity"]], axis=1).astype(np.float32)

    model = ConditionalDiffusionModel(feature_dim=FEATURE_DIM, timesteps=40)
    losses = model.train(X0, cond, epochs=150)

    n_per_template = 3
    rng = np.random.default_rng(99)
    all_generated = []
    for _ in range(n_per_template):
        gen = model.sample(cond, n_samples=len(X0))
        all_generated.append(gen)
    generated = np.concatenate(all_generated, axis=0)

    validation = validate_ranking(generated, rng)
    summary = {
        "n_templates": len(X0),
        "n_candidates_generated": len(generated),
        "final_training_loss": round(losses[-1], 4),
        "initial_training_loss": round(losses[0], 4),
        **validation,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
