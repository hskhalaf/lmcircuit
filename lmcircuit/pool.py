"""The frontier-coverage pool: the judges whose per-task cost-accuracy frontier comes closest to that of all judges.

For each task, a judge is a point (mean planning cost, accuracy) on the given items; the task's frontier is the best
accuracy reachable at each cost by one judge. A pool's regret on a task is the mean gap between the full frontier and
the pool's over a log-spaced cost grid. The pool minimises the mean regret over tasks (then the largest): greedy
additions, then single swaps until none improves. Model-free: no belief model is fitted. The mean, not the largest:
as good on OLL BBH and Decision Index, and the more stable pool across folds in our experiments.
"""
from __future__ import annotations

import itertools

import numpy as np

from .data import Dataset


def coverage_pool(ds: Dataset, items, size: int, plan_cost: np.ndarray) -> tuple[list[int], dict]:
    items = np.asarray(items)
    M = ds.n_judges
    if M <= size:
        return list(range(M)), {"note": "every judge"}
    label = ds.labels[items].argmax(axis=1)
    a0 = ds.votes()[items, :, 0]                                                  # the class each first call votes for
    right = np.where(a0 >= 0, a0 == label[:, None], np.nan)
    tasks = sorted(set(ds.task[items]))
    of = ds.task[items]
    chance = 1.0 / ds.n_classes
    with np.errstate(invalid="ignore"):
        A = np.asarray([np.nanmean(right[of == t], axis=0) for t in tasks])        # (T, M)
    A = np.nan_to_num(A, nan=chance)                                                # no valid call: chance
    C = np.asarray([plan_cost[items][of == t].mean(axis=0) for t in tasks])         # (T, M)
    positive = C[C > 0]
    lo = positive.min() * 0.999 if len(positive) else 1e-12                          # free judges sit at the left edge
    grid = np.geomspace(lo, max(C.max(), lo) * 1.001, 400)

    def front(k, S):
        f = np.full(len(grid), chance)
        for j in S:
            f = np.where(grid >= C[k, j], np.maximum(f, A[k, j]), f)
        return f
    full = [front(k, range(M)) for k in range(len(tasks))]

    def regret(S):
        return 100 * np.asarray([(full[k] - front(k, S)).mean() for k in range(len(tasks))])

    def score(S):
        r = regret(S)
        return r.mean(), r.max()
    pool: list[int] = []
    while len(pool) < size:
        pool.append(min((j for j in range(M) if j not in pool), key=lambda j: score(pool + [j])))
    improved = True
    while improved:
        improved = False
        for a, b in itertools.product(list(pool), range(M)):
            if b not in pool and score([b if x == a else x for x in pool]) < score(pool):
                pool, improved = [b if x == a else x for x in pool], True
    pool.sort(key=lambda j: C[:, j].mean())
    r = regret(pool)
    return pool, {"tasks": tasks, "regret_points": dict(zip(tasks, map(float, r))), "mean_regret": float(r.mean()),
                  "max_regret": float(r.max())}
