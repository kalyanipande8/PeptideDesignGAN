# GPU note

No GPU is available in this execution environment. The diffusion model is
trained on CPU with a small architecture (23-dim features, 2 hidden layers
of 64 units, 40 diffusion timesteps, 150 training epochs), which trains in
a few seconds on CPU. The real conditional DDPM algorithm (cosine-style
linear beta schedule, noise-prediction training objective, iterative
reverse-diffusion sampling) is unchanged from how it would run on GPU —
only the scale (feature dimensionality, timesteps, network width) is
reduced to fit CPU training budgets. No numbers in this README are
estimated or extrapolated from GPU behavior; every reported metric is
measured from this actual CPU run.
