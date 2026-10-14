"""Metrics of predictions against annotation counts.

Accuracy: agreement with a random annotator of a random item (each item weighs one). Balanced accuracy: the mean over
classes of the share of that class's annotations predicted as that class (annotations pooled, so an item with more
annotators weighs more; with one label per item both are the usual definitions). Brier score of a belief p over the
classes: half the squared distance to the item's annotation shares, (1/2) sum_k (p_k - share_k)^2, averaged over items
(for two classes the usual (P(1) - y)^2).
"""
from __future__ import annotations

import numpy as np

METRICS = ("accuracy", "balanced_accuracy")


def scores(pred: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """pred (..., n) classes, labels (n, K) counts -> {"accuracy", "balanced_accuracy"}, each of shape (...)."""
    labels = np.asarray(labels, dtype=np.float64)
    pred = np.asarray(pred)
    K = labels.shape[1]
    share = labels / np.maximum(labels.sum(axis=1, keepdims=True), 1e-12)
    onehot = np.eye(K)[pred]                                                    # (..., n, K)
    agree = (onehot * labels).sum(axis=-2)                                      # (..., K) annotations predicted right
    total = labels.sum(axis=0)
    present = total > 0
    out = {"accuracy": (onehot * share).sum(axis=-1).mean(axis=-1),
           "balanced_accuracy": (agree / np.maximum(total, 1e-12))[..., present].mean(axis=-1)}
    return out


def brier_rows(prob: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """prob (..., n, K) beliefs, labels (n, K) counts -> (..., n) each row's Brier score."""
    labels = np.asarray(labels, dtype=np.float64)
    share = labels / np.maximum(labels.sum(axis=1, keepdims=True), 1e-12)
    return 0.5 * ((np.asarray(prob, dtype=np.float64) - share) ** 2).sum(axis=-1)


def draws(preds: np.ndarray, labels: np.ndarray, groups: np.ndarray, probs: np.ndarray | None = None, n: int = 1000,
          seed: int = 0) -> dict[str, np.ndarray]:
    """Each metric of a curve under `n` resamples of the test groups with replacement (an item's rows and its group's
    items stay together): preds (B, rows) classes at B budgets, labels (rows, K) counts, groups (rows,), probs
    (B, rows, K) the beliefs behind them (adds "brier"). Returns {metric: (n, B)}. The same groups and seed give the
    same resamples, so the draws of two methods on the same rows are paired."""
    labels = np.asarray(labels, dtype=np.float64)
    _, g = np.unique(groups, return_inverse=True)
    G, K = int(g.max()) + 1, labels.shape[1]
    weight = np.random.default_rng(seed).multinomial(G, np.full(G, 1.0 / G), size=n).astype(np.float64)   # (n, G)
    per_group = lambda v: np.bincount(g, weights=v, minlength=G)  # noqa: E731
    share = labels / np.maximum(labels.sum(axis=1, keepdims=True), 1e-12)
    rows = weight @ per_group(np.ones(len(g)))
    total = weight @ np.stack([per_group(labels[:, k]) for k in range(K)], axis=1)               # (n, K)
    present = total > 0
    preds = np.asarray(preds)
    out = {m: np.empty((n, len(preds))) for m in METRICS + (("brier",) if probs is not None else ())}
    for b, p in enumerate(preds):
        out["accuracy"][:, b] = weight @ per_group(share[np.arange(len(p)), p]) / rows
        agree = weight @ np.stack([per_group(labels[:, k] * (p == k)) for k in range(K)], axis=1)
        out["balanced_accuracy"][:, b] = np.where(present, agree / np.maximum(total, 1e-12), 0).sum(1) / present.sum(1)
        if probs is not None:
            out["brier"][:, b] = weight @ per_group(brier_rows(probs[b], labels)) / rows
    return out


def interval(d: dict[str, np.ndarray], area: np.ndarray) -> dict[str, dict]:
    """Per metric of draws (n, B): lo, hi (B,), the 2.5 and 97.5 percentiles, and the frontier area's draws (n,), the
    mean over the budgets `area` (B,) bool."""
    return {m: {"lo": np.percentile(v, 2.5, axis=0), "hi": np.percentile(v, 97.5, axis=0), "area": v[:, area].mean(axis=1)}
            for m, v in d.items()}


def bootstrap(preds: np.ndarray, labels: np.ndarray, groups: np.ndarray, area: np.ndarray, draws_: int = 1000,
              seed: int = 0, probs: np.ndarray | None = None) -> dict[str, dict]:
    """The 95% interval of a curve and its area from resampling test groups (draws, then interval)."""
    return interval(draws(preds, labels, groups, probs, draws_, seed), area)


def over_seeds(per_seed: list[dict[str, np.ndarray]], seed: int = 1) -> dict[str, np.ndarray]:
    """Draws of the mean over training seeds: draw d averages, over seeds resampled with replacement, each seed's draw d
    (the test groups resampled alike for every seed). The band then holds both the test sample's and the training
    seeds' variation; with one seed it is the plain bootstrap."""
    S, n = len(per_seed), next(iter(per_seed[0].values())).shape[0]
    pick = np.random.default_rng(seed).integers(0, S, size=(n, S)) if S > 1 else np.zeros((n, 1), dtype=int)
    return {m: np.stack([d[m] for d in per_seed])[pick, np.arange(n)[:, None]].mean(axis=1) for m in per_seed[0]}
