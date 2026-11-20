"""Real small conditional DDPM over peptide composition-feature vectors.

Distinct from ProteinDesignML's diffusion model: here the feature space is
amino-acid composition + physicochemical descriptors (23-dim) rather than
generic structural coordinates, and conditioning is on a target
stability/affinity pair rather than a structural template embedding.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class DenoiserNet(nn.Module):
    def __init__(self, feature_dim: int, cond_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + cond_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, x, cond, t):
        inp = torch.cat([x, cond, t], dim=1)
        return self.net(inp)


class ConditionalDiffusionModel:
    def __init__(self, feature_dim: int, timesteps: int = 40, seed: int = 7):
        torch.manual_seed(seed)
        self.T = timesteps
        self.feature_dim = feature_dim
        betas = torch.linspace(1e-4, 0.02, timesteps)
        self.betas = betas
        self.alphas = 1 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.model = DenoiserNet(feature_dim)

    def q_sample(self, x0, t, noise):
        ab = self.alpha_bars[t].unsqueeze(1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise

    def train(self, X0: np.ndarray, cond: np.ndarray, epochs: int = 150, lr: float = 1e-3):
        X0_t = torch.tensor(X0, dtype=torch.float32)
        cond_t = torch.tensor(cond, dtype=torch.float32)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        n = X0_t.shape[0]
        losses = []
        for epoch in range(epochs):
            t = torch.randint(0, self.T, (n,))
            noise = torch.randn_like(X0_t)
            xt = self.q_sample(X0_t, t, noise)
            t_norm = (t.float() / self.T).unsqueeze(1)
            pred_noise = self.model(xt, cond_t, t_norm)
            loss = ((pred_noise - noise) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        return losses

    @torch.no_grad()
    def sample(self, cond: np.ndarray, n_samples: int) -> np.ndarray:
        cond_t = torch.tensor(cond, dtype=torch.float32)
        if cond_t.shape[0] != n_samples:
            cond_t = cond_t[:n_samples]
        x = torch.randn(n_samples, self.feature_dim)
        for t in reversed(range(self.T)):
            t_batch = torch.full((n_samples, 1), t / self.T, dtype=torch.float32)
            pred_noise = self.model(x, cond_t, t_batch)
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]
            mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise)
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta) * noise
            else:
                x = mean
        return x.numpy()
