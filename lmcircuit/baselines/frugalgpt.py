"""FrugalGPT (Chen, Zaharia & Zou, TMLR 2024, "FrugalGPT: How to Use Large Language Models While Reducing Cost and
Improving Performance"; stanford-futuredata/FrugalGPT), the LLM cascade.

The library's own LLMChain and optimizer (vendored in _vendor/frugalgpt, unmodified algorithm): every ordering of
`chain_length` judges, and for each a score threshold per cascade step brute-forced (scipy.optimize.brute, 40 quantile
points per step, then fmin) to maximise accuracy on the training worlds within a budget on the expected cost; the
best ordering wins. At use, a judge's answer is accepted when its score is above the step's threshold, else the next
judge is asked; the last judge always answers.

Scoring function: the released one (FrugalGPT/scoring.py), fine-tuned per fold on its training items by
scorers.frugalgpt_scores: one DistilBERT per judge reading "query answer", the probability the answer is correct,
trained on one response per training query with its own 55% hold-out. A call answering v on item i is scored
S[i, j, v]. As in the release (LLMCascade.train), the cascade is optimised on the training worlds with those scores;
the validation items are not used.

Chain length 3, the longest cascade of the paper (the library enumerates every ordering of L_max judges and
brute-forces L_max - 1 thresholds; L_max = 8 is out of reach). Budgets: 12 log-spaced from 1.15 x the cheapest judge
(a cascade must afford its first call; LLMChain has no free fallback) to the cost of the three most expensive judges,
in the circuit's unit with each judge's mean planning cost per call. Training worlds: 2000 drawn at random (seeded),
since the vendored objective is a Python loop over worlds. A budget with no feasible chain gives no point. LLMChain
writes to relative temp/ and strategy/ folders, so every optimisation runs in its own temporary working directory.

Substitutions (made in our experiments): the scorer is trained on the judges' verdicts (quality: agreement with
the annotations), one response per training query = the item's first recorded call; chain_length bounds L_max.
"""
from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from ..config import resolve_device
from . import scorers
from .common import FoldData, Point, Result

SUBSTITUTIONS = [
    "scoring function trained on the judges' verdicts: quality = agreement with the annotations, one response per "
    "training query = the item's first recorded call",
    "chain_length bounds LLMChain L_max (paper: 3)",
]
_SHARED: dict = {}


@contextlib.contextmanager
def _sandbox():
    """A fresh cwd with the relative temp/ and strategy/ folders LLMChain writes to; the library's per-ordering log
    lines (at CRITICAL) silenced."""
    previous, disabled = os.getcwd(), logging.root.manager.disable
    with tempfile.TemporaryDirectory(prefix="frugalgpt_") as tmp:
        (Path(tmp) / "temp").mkdir()
        (Path(tmp) / "strategy").mkdir()
        os.chdir(tmp)
        logging.disable(logging.CRITICAL)
        try:
            yield
        finally:
            logging.disable(disabled)
            os.chdir(previous)


def _responses(fd: FoldData, worlds: list[tuple[int, int]], cost) -> dict:
    keys = [f"{i}#{r}" for i, r in worlds]
    return {judge: {"answer": {k: str(int(fd.answers[i, j, r])) for k, (i, r) in zip(keys, worlds)},
                    "cost": {k: float(cost(i, j, r)) for k, (i, r) in zip(keys, worlds)}, "sp": {}, "logprobs": {}}
            for j, judge in enumerate(fd.judges)}


def _scores(fd: FoldData, S: np.ndarray, worlds: list[tuple[int, int]]) -> dict:
    return {judge: {f"{i}#{r}": float(S[i, j, fd.answers[i, j, r]]) for i, r in worlds} for j, judge in enumerate(fd.judges)}


def _eval_worlds(fd: FoldData, items) -> list[tuple[int, int]]:
    return [(int(i), r) for i in items for r in range(fd.worlds)]


def _budget(task: tuple[int, float]):
    from ._vendor.frugalgpt import LLMChain
    b, budget = task
    sh = _SHARED
    fd: FoldData = sh["fd"]
    started = time.perf_counter()
    with _sandbox():
        chain = LLMChain(metric="em", L_max=sh["chain_length"])
        chain.setbudget(budget=float(budget))
        chain.train(sh["responses_train"], sh["labels_train"], sh["scores_train"])
        if not chain.model_ids:
            return None
        fitted = time.perf_counter()
        preds = {name: chain.predict(sh[f"responses_{name}"], sh[f"scores_{name}"]) for name in ("val", "test")}
    index = {judge: j for j, judge in enumerate(fd.judges)}
    out = {}
    for name, items in (("val", fd.val), ("test", fd.test)):
        pred = np.zeros((len(items), fd.worlds), dtype=np.int64)
        queried = np.zeros((len(items), fd.M, fd.worlds), dtype=bool)
        for k, (i, r) in enumerate(_eval_worlds(fd, items)):
            key = f"{i}#{r}"
            pred[k // fd.worlds, r] = int(preds[name]["answer"][key])
            for judge in preds[name]["query_apis"][key]:
                queried[k // fd.worlds, index[judge], r] = True
        out[name] = fd.charged(items, pred, queried)
    return Point({"budget": float(budget), "chain": list(chain.model_ids), "thresholds": [float(t) for t in chain.thres],
                  "fit_seconds": fitted - started, "decide_seconds": time.perf_counter() - fitted},
                 out["val"], out["test"], budget=float(budget))


def run(fd: FoldData, p: dict, cfg) -> Result:
    with fd.stage("fit"):          # the scorer fine-tuned, then its scores on every call (inference counted here)
        S, report = scorers.frugalgpt_scores(fd, cfg.scorer("frugalgpt"), resolve_device(cfg.device), cfg.text_parts)
    started = time.perf_counter()
    chain_length = min(int(p["chain_length"]), fd.M)
    train = fd.train_worlds(None if p.get("train_cap") is None else int(p["train_cap"]))
    top = np.sort(fd.plan)[-chain_length:].sum()
    budgets = np.geomspace(fd.plan.min() * 1.15, top, int(p["budgets"]))
    _SHARED.update(fd=fd, chain_length=chain_length,
                   responses_train=_responses(fd, train, lambda i, j, r: fd.plan[j]),
                   labels_train=[{"_id": f"{i}#{r}", "answer": str(int(fd.y[i]))} for i, r in train],
                   scores_train=_scores(fd, S, train))
    for name, items in (("val", fd.val), ("test", fd.test)):
        worlds = _eval_worlds(fd, items)
        _SHARED[f"responses_{name}"] = _responses(fd, worlds, lambda i, j, r: fd.cost[i, j, r])
        _SHARED[f"scores_{name}"] = _scores(fd, S, worlds)
    tasks = list(enumerate(budgets.tolist()))
    prep = time.perf_counter() - started                     # everything before the per-budget optimisation
    workers = max(1, min(int(cfg.workers), len(tasks)))
    try:
        if workers == 1:
            results = [_budget(t) for t in tasks]
        else:
            with mp.get_context("fork").Pool(workers) as pool:
                results = pool.map(_budget, tasks, chunksize=1)
    finally:
        _SHARED.clear()
    points = [r for r in results if r is not None]
    fd.add("fit", prep + sum(r.settings["fit_seconds"] for r in points))  # per budget, in worker processes:
    fd.add("decide", sum(r.settings["decide_seconds"] for r in points))        # their sum
    return Result(points, {"reference": "Chen, Zaharia & Zou, FrugalGPT (stanford-futuredata/FrugalGPT) LLMChain + "
                                        "optimizer, vendored", "substitutions": SUBSTITUTIONS,
                           "chain_length": chain_length, "budgets": budgets.tolist(), "train_worlds": len(train),
                           "skipped_budgets": len(results) - len(points), "scorer": report,
                           "scorer_seconds": report["seconds"]})
