"""Cascade routing (Dekoninck, Baader & Vechev, ICML 2025, "A Unified Approach to Routing and Cascading for LLMs";
eth-sri/cascade-routing).

The library's classification setting (vendored in _vendor/cascade_routing): ClassificationQualityComputer, learned
per-observed-subset LogisticRegression estimators of each judge's correctness from the answers seen so far (their
class probabilities, agreement and entropy features), fitted on training worlds; ClassificationCostComputer with
constant per-judge costs (each judge's mean planning cost per call, circuit unit); and CascadeRouter with
force_order=False, the paper's cascade routing (at every step the next judge, or stopping, is chosen by the
quality-cost trade-off; the answer is the judge estimated best among those asked). For each of 12 budgets
(log-spaced from 0.5 x the cheapest judge to the sum of all judges) the router's lambdas are fitted on the training
worlds with the library's ConstantStrategy (20 bisection steps; the router mixes two lambdas with probability gamma
to meet the budget), then the policy is replayed on the validation and the test worlds. Training worlds: 2000 drawn
at random (seeded), as in our experiments (each budget's lambda search replays every training world).

Substitutions (made in our experiments): a judge's "answer" is the call's class probabilities at the verdict
(Dataset.probs), its one-hot verdict where not recorded; costs are constant per judge (planning cost).
Seeds: the router's random mixing draws np.random; it is seeded with seed + b for budget b's fit, seed + 100003
(b + 1) for the test replay and seed + 200003 (b + 1) for the validation replay.
Deviation from the library: its quality computer draws a scrambled Sobol sequence (for the expected maximum of the
quality estimates) from fresh entropy; the vendored copy seeds it, so a run is reproducible.
"""
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

from .common import FoldData, Point, Result

SUBSTITUTIONS = [
    "judge answers are each call's class probabilities at the verdict token; the one-hot verdict where not recorded",
    "quality/cost computers are the library's learned per-observed-subset LogisticRegression estimators fit on "
    "training worlds; costs are constant per judge (planning cost)",
]
_SHARED: dict = {}


def _inputs(fd: FoldData, worlds: list[tuple[int, int]]):
    """Library inputs per world: question key, each judge's class probabilities, correctness, realised cost."""
    probs, onehot = fd.ds.probs, np.eye(fd.K)

    def vec(i, j, r):
        p = None if probs is None else probs[i, j, r].astype(np.float64)
        return p if p is not None and np.isfinite(p).all() else onehot[fd.answers[i, j, r]]
    questions = [f"{i}#{r}" for i, r in worlds]
    answers = [[vec(i, j, r) for j in range(fd.M)] for i, r in worlds]
    quality = np.asarray([(fd.answers[i, :, r] == fd.y[i]).astype(np.float64) for i, r in worlds])
    cost = np.asarray([fd.cost[i, :, r] for i, r in worlds], dtype=np.float64)
    return questions, answers, quality, cost


def _replay(router, data, seed: int, fd: FoldData, items):
    q, a, u, c = data
    np.random.seed(seed)
    out = router._execute(router.lambdas, q, a, ground_truth_qualities=u, ground_truth_costs=c)
    n, W = len(items), fd.worlds
    sel = np.asarray(out["per_question_selected"]).reshape(n, W)
    pred = np.take_along_axis(fd.answers[items, :, :W], sel[:, None, :], axis=1)[:, 0]
    queried = np.zeros((n * W, fd.M), dtype=bool)
    for k, run in enumerate(out["per_question_models_run"]):
        queried[k, run] = True
    return fd.charged(items, pred, queried.reshape(n, W, fd.M).transpose(0, 2, 1))


def _budget(task: tuple[int, float]) -> Point:
    from ._vendor.cascade_routing import CascadeRouter, ConstantStrategy
    b, budget = task
    sh = _SHARED
    fd: FoldData = sh["fd"]
    started = time.perf_counter()
    np.random.seed(fd.seed + b)
    router = CascadeRouter(sh["quality"], sh["cost"], models=[{"name": j} for j in fd.judges],
                           max_expected_cost=float(budget),
                           strategies=[ConstantStrategy(max(budget, 1e-9), n_iterations=sh["lambda_iterations"])],
                           force_order=False, cascade=False, max_depth=None)
    q, a, u, _ = sh["train"]
    router.fit(q, a, ground_truth_qualities=u, ground_truth_costs=sh["train_plan"])
    fitted = time.perf_counter()
    test = _replay(router, sh["test"], fd.seed + 100_003 * (b + 1), fd, fd.test)
    val = _replay(router, sh["val"], fd.seed + 200_003 * (b + 1), fd, fd.val)
    return Point({"budget": float(budget), "lambdas": [None if v is None else float(v) for v in router.lambdas],
                  "gamma": float(router.gamma), "fit_seconds": fitted - started,
                  "decide_seconds": time.perf_counter() - fitted}, val, test, budget=float(budget))


def run(fd: FoldData, p: dict, cfg) -> Result:
    from sklearn.linear_model import LogisticRegression

    from ._vendor.cascade_routing import ClassificationCostComputer, ClassificationQualityComputer
    started = time.perf_counter()
    worlds = fd.train_worlds(None if p.get("train_cap") is None else int(p["train_cap"]))
    train = _inputs(fd, worlds)
    train_plan = np.tile(fd.plan[None, :], (len(worlds), 1))
    quality = ClassificationQualityComputer(model_class=LogisticRegression, n_highest_include=1,
                                            question_indicator="___NOQ___", answer_indicator="___NOA___",
                                            remove_options=None, max_depth=None, n_samples=100, store_all=True)
    quality.fit(train[0], train[1], train[2])
    cost = ClassificationCostComputer(input_costs=[0.0] * fd.M, output_costs=[0.0] * fd.M, tokenize=False,
                                      constant_cost=True)
    cost.fit(train[0], train[1], train_plan)
    fit_seconds = time.perf_counter() - started
    budgets = np.geomspace(fd.plan.min() * 0.5, fd.plan.sum(), int(p["budgets"]))
    _SHARED.update(fd=fd, quality=quality, cost=cost, lambda_iterations=int(p["lambda_iterations"]), train=train,
                   train_plan=train_plan,
                   val=_inputs(fd, [(int(i), r) for i in fd.val for r in range(fd.worlds)]),
                   test=_inputs(fd, [(int(i), r) for i in fd.test for r in range(fd.worlds)]))
    tasks = list(enumerate(budgets.tolist()))
    prep = time.perf_counter() - started                     # everything before the per-budget optimisation
    workers = max(1, min(int(cfg.workers), len(tasks)))
    try:
        if workers == 1:
            points = [_budget(t) for t in tasks]
        else:          # fork shares the fitted estimators copy-on-write
            with mp.get_context("fork").Pool(workers) as pool:
                points = pool.map(_budget, tasks, chunksize=1)
    finally:
        _SHARED.clear()
    fd.add("fit", prep + sum(pt.settings["fit_seconds"] for pt in points))  # per budget, in worker processes:
    fd.add("decide", sum(pt.settings["decide_seconds"] for pt in points))        # their sum
    return Result(points, {"reference": "Dekoninck et al., cascade routing (eth-sri/cascade-routing), vendored; "
                                        "force_order=False", "substitutions": SUBSTITUTIONS,
                           "budgets": budgets.tolist(), "train_worlds": len(worlds), "estimator_fit_seconds": fit_seconds})
