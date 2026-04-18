# Sequential Prediction for Robot Trajectories Using a Transformer

A decoder-only GPT-style transformer trained from scratch in PyTorch to predict the next action token in robot motion sequences — applied to the VLA Foundation Dataset.

---

## Overview

Robots need to plan sequences of actions — just like language models predict the next word. This project applies modern LLM techniques (RoPE, RMSNorm, SwiGLU) to robotic trajectory data, framing robot action prediction as a next-token prediction task.

The model is built entirely from scratch using PyTorch and trained on a T4 GPU via Google Colab.

---

## Problem Statement

Given a sequence of past robot actions, the model must predict the next action token — an autoregressive next-token prediction task. Each token is a quantized integer representing a discrete robot movement. The random baseline accuracy is only **0.39%** — any learning above this is meaningful.

---

## Dataset

| Property | Value |
|---|---|
| Source | VLA Foundation Dataset |
| Total trajectories | 10,000 |
| Train / Validation split | 9,000 / 1,000 |
| Sequence length | 50 tokens |
| Vocab size | 256 discrete action tokens |

- **States**: 3D spatial positions (X, Y, Z) of the robot end-effector
- **Actions**: Quantized integer tokens representing discrete robot movements

---

## Model Architecture

A decoder-only transformer with **8.46M trainable parameters**, using modern components from recent LLM research.

| Component | Details |
|---|---|
| Token Embedding | Vocab 256 → dim 384 |
| Positional Encoding | RoPE (Rotary Positional Embedding) — better generalisation than absolute |
| Attention | Causal self-attention, 6 layers, 8 heads, head dim 48 |
| Feed-Forward | SwiGLU-style with SiLU activation, hidden dim 1024 |
| Normalisation | RMSNorm (more stable than LayerNorm, used in Llama/PaLM) |
| Output | Linear LM head → vocab size 256 |
| Dropout | 0.1 throughout |

```
Input tokens
    │
Token Embedding (256 → 384)
    │
┌── Transformer Block × 6 ──────────────────────┐
│   RMSNorm                                      │
│   Causal Self-Attention + RoPE (8 heads)       │
│   RMSNorm                                      │
│   FeedForward (SwiGLU, hidden=1024)            │
└────────────────────────────────────────────────┘
    │
RMSNorm
    │
LM Head (384 → 256)
    │
Next action token prediction
```

---

## Training Setup

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Weight decay | 0.01 |
| Batch size | 32 |
| Epochs | 15 |
| Gradient clipping | 1.0 |
| Hardware | T4 GPU (Google Colab) |

---

## Results

| Metric | Value |
|---|---|
| Train Loss | 4.56 |
| Validation Loss | 4.68 |
| Train-Val Gap | 0.12 (mild overfitting) |
| Train Accuracy | 2.92% |
| **Validation Accuracy** | **1.64%** |
| Random Baseline | 0.39% |
| Train Perplexity | 95.28 |
| **Validation Perplexity** | **107.67** |
| Random Chance Perplexity | 256 |

**Key takeaways:**
- Validation accuracy is **4× above the random baseline** — the model is learning real sequential structure
- Perplexity dropped from 256 (random) to ~108 — significant reduction in uncertainty
- Mild overfitting (gap = 0.12) is manageable for this training duration
- Low absolute accuracy reflects high token entropy in robot actions, not model failure

---

## Visualisations

The notebook generates the following plots:

- `loss_and_accuracy_curves.png` — training vs validation loss and accuracy over epochs
- `attention_maps_grid.png` — multi-head, multi-layer attention heatmaps
- `pred_vs_gt_trajectory.png` — 3-panel predicted vs ground-truth trajectory comparison
- `experiment_comparison.png` — ablation study across layer depth, learning rate, sequence length
- `trajectory_plot.png` — static 3D robot trajectory plot
- `trajectory_animation.gif` — animated 3D robot trajectory

---

## Experiments

Three ablation experiments run via `run_experiment()`:

| Experiment | Variable | Values Tested |
|---|---|---|
| Layer depth | `NUM_LAYERS` | 4 vs 6 |
| Learning rate | `LEARNING_RATE` | 5e-5 vs 1e-4 |
| Sequence length | `MAX_SEQ_LEN` | 30 vs 50 |

---

## Notebook Structure

```
1. Model Architecture     — RMSNorm, RoPE, Causal Self-Attention, FeedForward, Transformer Block
2. Training Pipeline      — train/val loop with loss + token accuracy tracking
3. Analysis               — loss, perplexity, auto-diagnosis (overfitting/underfitting)
4. Attention Maps         — multi-head, multi-layer heatmap grid + entropy analysis
5. Prediction vs GT       — 3D trajectory comparison with token match rate
6. Experiments            — layer depth, learning rate, sequence length ablations
7. Trajectory Animation   — animated 3D robot path saved as GIF
```

---

## Requirements

```bash
pip install torch numpy matplotlib
```

Run on Google Colab with GPU (T4 or better) for best performance.

---

## Conclusion

This project demonstrates that modern LLM architecture techniques transfer effectively to non-language sequential data. The decoder-only transformer learns meaningful patterns from robot trajectories with only 15 epochs of training on 9,000 sequences.

---

## Future Work

- **More data and longer training** — action tokens have high entropy; 30+ epochs with a larger dataset should improve accuracy significantly
- **Learning rate scheduler** — warmup + cosine decay for faster and more stable convergence
- **Cross-attention on state inputs** — conditioning on 3D position states (currently unused in the decoder) could substantially improve trajectory prediction quality
- **Beam search / top-k sampling** — explore better decoding strategies beyond greedy prediction

---

## Files Saved

| File | Description |
|---|---|
| `checkpoints/best_model.pt` | Best model weights (lowest val loss) |
| `loss_and_accuracy_curves.png` | Training curves |
| `attention_maps_grid.png` | Attention visualisation |
| `pred_vs_gt_trajectory.png` | Prediction comparison |
| `experiment_comparison.png` | Ablation results |
| `trajectory_animation.gif` | Animated trajectory |
