"""CARROT (Somerstep et al., 2025, "CARROT: A Cost Aware Rate Optimal Router").

A predictive router over every pool judge, one call per item. For an item x it estimates each judge's probability of a
correct answer P_m(x) and its cost C_m(x), and calls the judge minimising (1 - mu)(1 - P_m(x)) + mu C_m(x); sweeping
mu in [0, 1] traces the curve (0: the judge predicted most accurate; 1: the judge predicted cheapest). The estimators
are the paper's CARROT (KNN): a multi-output k-nearest-neighbours regressor on the item's features, k = 40 by cosine
similarity (their RouterBench setting), fitted on the training items. No validation data in the fit.

Substitutions (made in our experiments):
* the item's prior features (lmcircuit.embed) instead of OpenAI's text-embedding-3-small;
* a judge's correctness and cost on a training item are means over its recorded calls (correctness: the share of
  annotations agreeing with the call); training items without a call of some judge are left out (no mean to regress);
* costs in the circuit's unit, mu swept as lam / (1 + lam) over 0, a log grid of lam and infinity: rescaling cost by a
  constant only re-indexes mu, so these are the routers of a mu grid in dollars.
"""
from __future__ import annotations

import numpy as np

from .common import FoldData, Point, Result

SUBSTITUTIONS = [
    "item features: the dataset's prior features instead of OpenAI text-embedding-3-small",
    "a judge's correctness and cost on a training item are means over its recorded calls (correctness: the share of "
    "annotations agreeing with the call)",
    "mu swept as lam / (1 + lam) over a log grid of lam, costs in the circuit's unit (the same routers as a mu grid in "
    "dollars)",
]


def run(fd: FoldData, p: dict, cfg) -> Result:
    from sklearn.neighbors import KNeighborsRegressor
    with fd.stage("fit"):
        train = fd.full_train()
        correct = fd.mean_over_calls(fd.quality(train), train)                                # (n, M)
        cost = fd.mean_over_calls(fd.cost[train, :, :fd.worlds], train)
        knn = KNeighborsRegressor(n_neighbors=min(int(p["neighbors"]), len(train)), metric="cosine",
                                  n_jobs=max(1, cfg.workers))
        knn.fit(fd.features[train], np.hstack([correct, cost]))
    lams = np.concatenate([[0.0], np.geomspace(float(p["lam_lo"]), float(p["lam_hi"]), int(p["lambdas"])), [np.inf]])
    points = []
    with fd.stage("decide"):
        est = {name: knn.predict(fd.features[items]) for name, items in (("val", fd.val), ("test", fd.test))}
        for lam in lams:
            out = {}
            for name, items in (("val", fd.val), ("test", fd.test)):
                pc, c = est[name][:, :fd.M], est[name][:, fd.M:]
                choice = np.argmin(c if np.isinf(lam) else (1 - pc) + lam * c, axis=1)
                out[name] = fd.single(items, choice)
            points.append(Point({"mu": 1.0 if np.isinf(lam) else float(lam / (1 + lam)), "lam": float(lam)},
                                out["val"], out["test"]))
    return Result(points, {"reference": "Somerstep et al., CARROT: A Cost Aware Rate Optimal Router (2025), CARROT (KNN)",
                           "substitutions": SUBSTITUTIONS, "training_items": int(len(train))})
