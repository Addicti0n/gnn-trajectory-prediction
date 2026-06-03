# Multimodal Trajectory Prediction for Autonomous Vehicles using Goal-Conditioned Graph Neural Networks

> MSc Thesis — MSc in Artificial Intelligence and Deep Learning, University of West Attica
> Author: Panagiotis Routsis

## Overview

Predicting the future trajectories of surrounding vehicles is a critical component for the
safe operation of Autonomous Vehicles (AVs). Recent Graph Neural Network (GNN) architectures
capture complex spatio-temporal interactions among traffic participants, but frequently produce
**spatially inconsistent** predictions — trajectories that intersect non-drivable areas
(pedestrian islands, off-road terrain).

This thesis proposes an **Encoder–Interactor–Decoder** GNN architecture that addresses this
limitation through a novel **Map-Aware Loss Function**.

## Architecture

| Stage | Component | Purpose |
|-------|-----------|---------|
| Encoder | Gated Recurrent Units (GRU) | Lightweight temporal encoding of agent history |
| Interactor | Graph Attention Networks (GAT) + Goal Conditioning | Cross-context interaction modeling |
| Decoder | Laplacian Mixture Density Network (MDN) | Multimodal trajectory distributions |

## Novelty — Map-Aware Loss

A differentiable **off-road penalty** that explicitly constrains predictions to respect the
HD-map drivable-area boundaries, minimizing physically implausible trajectories.

## Dataset

- **nuScenes** (`v1.0-mini` for development; full set for final training)
- Prediction protocol: **2 s history → 6 s future** at 2 Hz
  - History: 5 timesteps (t-4 … t0)
  - Future: 12 timesteps (t+1 … t+12)

## Evaluation Metrics

- minADE (minimum Average Displacement Error)
- minFDE (minimum Final Displacement Error)
- Miss Rate
- **Off-Road Rate** (primary contribution target)

## Baseline

- GC-GAT (Gulzar et al., 2025)

## Project Status

- [x] Environment setup + nuScenes v1.0-mini acquired
- [x] Spatio-temporal graph construction (history) — *in progress*
- [ ] Future trajectory extraction (labels)
- [ ] Dataset-wide windowing / Dataset class
- [ ] HD-map / drivable-area integration
- [ ] Model (GRU + GAT + Laplacian MDN)
- [ ] Map-Aware Loss
- [ ] Training + evaluation
- [ ] Visualizations

## Repository Structure

```
GNN_Thesis_Project/
├── data/              # nuScenes (git-ignored — not committed)
├── data_builder.py    # spatio-temporal graph construction
├── requirements.txt
└── README.md
```

## References

1. Gulzar, Muhammad & Muhammad (2025). *GC-GAT: Multimodal Vehicular Trajectory Prediction Using Graph Goal Conditioning and Cross-Context Attention.* IEEE RA-L.
2. Veličković et al. (2018). *Graph Attention Networks.* ICLR.
3. Liu et al. (2024). *LaFormer: Trajectory Prediction with Lane-Aware Scene Constraints.* CVPR.
4. Caesar et al. (2020). *nuScenes: A Multimodal Dataset for Autonomous Driving.* CVPR.
