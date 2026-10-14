"""What a circuit optimises: class weights for the DP's stopping risk, and how an operating point is chosen.

The DP stops with the prediction argmax_y w_y P(y | state), risking sum_{y != y'} w_y P(y | state) (dp.stop_loss).
- accuracy: w = 1.
- balanced accuracy: w_y = 1 / (K pi_y), pi the training class shares (two classes: predict 1 when P(1) > pi_1).
Both are linear in the confusion matrix, so the DP optimises them exactly. At a budget, every method (the circuit and
the baselines) takes the operating point with the best validation metric among those whose validation cost is within
the budget: the constrained problem itself, estimated on validation. Spending the most within the budget instead
assumes the metric rises with spending, which finite samples and an imperfect belief model do not guarantee.
"""
from __future__ import annotations

import numpy as np

OBJECTIVES = ("accuracy", "balanced_accuracy")


def weights(objective: str, labels: np.ndarray) -> np.ndarray:
    """(1, K) class weights the DP is solved with."""
    K = labels.shape[1]
    if objective == "accuracy":
        return np.ones((1, K))
    if objective == "balanced_accuracy":
        counts = labels.sum(axis=0)                     # annotations pooled, as metrics.scores does
        present = counts > 0
        w = np.zeros(K)                                 # a class absent from training is never predicted
        w[present] = counts.sum() / (present.sum() * counts[present])
        return w[None]
    raise ValueError(f"objective is one of {OBJECTIVES}")


def pick(val_cost: np.ndarray, val_metric: np.ndarray, budget: float):
    """The operating point within `budget` on validation (an index into the val arrays): the best metric, the cheapest
    on ties. With nothing within the budget, the cheapest."""
    cost, metric = np.asarray(val_cost).ravel(), np.asarray(val_metric).ravel()
    ok = np.flatnonzero(cost <= budget + 1e-12)
    i = int(np.argmin(cost)) if not len(ok) else int(ok[np.lexsort((cost[ok], -metric[ok]))[0]])
    return tuple(int(k) for k in np.unravel_index(i, np.shape(val_cost)))
