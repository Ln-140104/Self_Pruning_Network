# Self-Pruning Neural Network — Analysis Report

## 1. Why Does an L1 Penalty on Sigmoid Gates Encourage Sparsity?

Each weight `w_ij` is scaled by a gate `g_ij = sigmoid(s_ij) ∈ (0, 1)`. The training loss is:

```
Total Loss = CrossEntropyLoss  +  lambda * (1/N) * sum_ij sigmoid(s_ij)
```

where N is the total number of gate parameters (~3.8M in this network).

### L1 vs L2 — why L1 produces exact zeros

| Penalty | Gradient w.r.t. gate g | Behaviour as g → 0 |
|:-------:|:----------------------:|:-------------------:|
| L1: `lambda * |g|` | `-lambda` (constant) | Gate is driven all the way to 0 — true pruning |
| L2: `lambda * g²` | `-2*lambda*g` (shrinks) | Gradient vanishes near 0 — gate stalls above 0 |

With L1, every gate receives a **constant downward gradient of -lambda per update step**, regardless of how small the gate already is. For unimportant weights, the classifier gradient cannot overcome this constant pull, so the gate collapses to exactly 0. For important weights, the classifier gradient is large enough to resist and hold the gate open. This is the same mechanism that makes LASSO regression produce exact zeros while ridge regression does not.

### Why normalise by gate count?

Using `.mean()` instead of `.sum()` keeps the sparsity loss in (0, 1) for any network size. This makes lambda directly interpretable as a per-gate penalty and avoids the loss exploding with larger architectures.

### Why a separate gate optimizer?

Gates and weights operate at different gradient scales. The accuracy gradient on a gate is proportional to how much that weight helps classification — large for important weights, near zero for unimportant ones. With a shared optimizer at `lr=1e-3`, the accuracy gradient of even marginally useful weights overpowers the sparsity penalty, so no gates close. Using a dedicated gate optimizer at `lr=0.1` (100× higher) gives the sparsity gradient enough force to close unimportant gates while still allowing the classifier to hold important ones open.

---

## 2. Results

| Lambda | Test Accuracy | Sparsity (%) | Pruned Gates          |
|:------:|:-------------:|:------------:|:---------------------:|
| 0.5    | **63.12%**    | 80.53%       | 3,063,154 / 3,803,648 |
| 1.0    | 61.85%        | 88.33%       | 3,359,908 / 3,803,648 |
| 2.0    | 61.44%        | **93.69%**   | 3,563,814 / 3,803,648 |

### Analysis

- **Sparsity increases monotonically with lambda**, confirming the L1 penalty is working as designed.
- **Accuracy is remarkably robust**: going from lambda=0.5 (80.5% pruned) to lambda=2.0 (93.7% pruned) costs only **1.68% accuracy** — from 63.12% to 61.44%.
- **At lambda=2.0**, over 3.56 million of 3.8 million gates are closed. The network achieves 61.44% CIFAR-10 accuracy using fewer than 7% of its original connections.
- The sparse loss (`sparse` column in training logs) decreases each epoch, confirming gates are being actively closed during training — not post-hoc.

---

## 3. Gate Value Distribution

The gate histogram for lambda=0.5 (lowest sparsity, richest distribution) shows a clear **bimodal shape**:

- **Large spike near 0**: the ~80% of weights the network deemed unnecessary — gates driven to zero by the constant L1 gradient.
- **Cluster near 0.5–1.0**: the ~20% of weights the network kept — gates held open by strong classifier gradients.

The absence of values in the middle (0.1–0.4) confirms gates are making **hard binary decisions** — open or closed — rather than soft-scaling weights. This is the hallmark of successful learned sparsity.

---

## 4. Implementation Summary

| Component | Choice | Reason |
|-----------|--------|--------|
| Gate activation | `sigmoid` | Smooth, bounded in (0,1), differentiable everywhere |
| Sparsity loss | `mean` of all gate values | Keeps loss in (0,1); lambda is size-independent |
| Gate initialisation | `0.0` → sigmoid(0)=0.5 | Neutral start; gradient decides open or closed |
| Gate optimizer | Adam, lr=0.1 | High LR makes sparsity gradient dominate for unimportant weights |
| Weight optimizer | Adam, lr=1e-3 | Standard rate for classification task |
| LR schedule | Cosine annealing on both | Smooth decay without premature convergence |
| Sparsity threshold | 0.01 | Gate < 0.01 contributes < 1% of weight magnitude — practically zero |
