# PeptideDesignGAN

A generative peptide design pipeline using diffusion-based sampling over
amino-acid composition/physicochemical feature vectors, conditioned on
target stability/affinity, with multi-objective ranking of candidates.

> Built generative peptide design pipeline using diffusion-based sampling
> conditioned on 800+ structural templates.
> Implemented multi-objective scoring combining affinity and stability
> signals to rank 2,400+ design candidates.

## Relationship to ProteinDesignML

This repo is intentionally similar in spirit to the earlier
`ProteinDesignML` project (also a real conditional DDPM + multi-objective
scoring), since both resume bullets describe generative design + ranking
pipelines. The implementations are independent and genuinely different:
this project represents candidates as amino-acid composition + Kyte-Doolittle
hydrophobicity + net-charge features (peptide-sequence-level, 23-dim) rather
than generic structural-template coordinates, and combines stability/affinity
into a single score via a weighted harmonic mean of normalized proxies
rather than ProteinDesignML's learned-reward + energy-score blend.

## What's real vs. synthetic — read this first

**Real:** `src/diffusion_model.py` implements a genuine small conditional
DDPM (linear beta schedule, noise-prediction training objective via
`q_sample`, real iterative reverse-diffusion `sample` loop) trained on CPU
(see `docs/gpu_note.md` — no GPU available here). `src/multiobjective_scoring.py`
computes a real composition-based stability/affinity proxy and a weighted
harmonic-mean combined score, validated via Spearman rank correlation
against a noisy ground-truth score (same formula + independent measurement
noise, mirroring real assay variability).

**Synthetic:** there is no real peptide-binding dataset in this environment.
`src/simulate_templates.py` generates 820 random peptide sequences (length
10-24) and computes real amino-acid composition/physicochemical features
from them; the "true" stability and affinity values are a hidden linear/
quadratic function of hydrophobicity and net charge, independent of the
diffusion model and scoring code.

## Results (real, from `run_all.py`)

```
820 templates -> 2,460 candidates generated (820 x 3 samples)

Diffusion training loss: 1.0087 -> 0.2633 (converges)
Spearman rho (score vs. noisy ground truth): 0.8679
Top-10% hit rate: 72.36%
```

## Usage

```bash
pip install -r requirements.txt
python3 run_all.py
python3 -m pytest tests/ -v
```

## Project structure

```
src/
  simulate_templates.py    # 820-template composition/physchem feature simulation
  diffusion_model.py         # conditional DDPM (train + reverse-sampling)
  multiobjective_scoring.py   # composition-based proxy scoring + ranking validation
docs/
  gpu_note.md                 # GPU availability disclosure
run_all.py
tests/test_pipeline.py
```

## Tech stack

Python 3.9+, PyTorch, numpy, scipy, pytest. No GPU used (see
`docs/gpu_note.md`).

## Limitations

- No real peptide-binding assay data; all templates and scores are
  simulated.
- The diffusion model is trained at small scale (23-dim features, 40
  timesteps) suitable for CPU; a production system would use thousands of
  timesteps and a larger network on GPU.
- The "true" stability/affinity function is a simple linear/quadratic
  composition-based heuristic, not a physics-based or learned oracle.
