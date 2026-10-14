"""What every baseline shares: one fold's data, the label rules, worlds, and the record of an operating point.

A fold (`FoldData.build`) keeps the circuit's pool of judges, ordered cheapest first by their mean planning cost on the
training items (the order the cascades follow and the pairwise methods' "weak" judge come from), and the circuit's
items: labelled training items; validation and test items that are labelled and have a recorded call of every pool
judge in each of the first `worlds` worlds (evaluation.evaluate_fold's rule). A test row is one (item, world), item
major, as in Dataset.by_world: the rows the circuit is scored on.

Training reads only recorded calls: positions below `worlds` of a training item that hold a call (MISSING slots are
never read). Costs are in the circuit's unit: USD / Dataset.scale (the priciest judge's mean call = 1).

Labels (generalised to annotation counts): a method that needs a hard label
trains and calibrates on the annotators' majority class (`y`; ties go to the highest class index, so for two classes
an exact tie counts as class 1, as eta >= 0.5 would); a method defined with a response's quality uses
`quality`, the share of annotations agreeing with the call (1 or 0 under one annotation). Every operating point is
scored by lmcircuit.metrics.scores against the annotation counts (accuracy: agreement with a random annotator).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import combinations

import numpy as np

from ..data import MISSING, Dataset


class NotApplicable(ValueError):
    """The method cannot run on this dataset (too few recorded calls, no item text, a missing dependency)."""


@dataclass
class Outcome:
    """One operating point on one split's rows (item x world, item major): prediction, cost, calls per judge."""
    pred: np.ndarray                   # (rows,) class
    cost: np.ndarray                   # (rows,) charged cost, circuit unit
    calls: np.ndarray                  # (M,) calls per row of each judge (a repeated-call method may exceed 1)


@dataclass
class Point:
    settings: dict                     # the knob values (and anything the method reports per point)
    val: Outcome
    test: Outcome
    budget: float | None = None        # the optimisation budget, circuit unit (methods selected by their budget)


@dataclass
class Result:
    points: list[Point]
    info: dict = field(default_factory=dict)


@dataclass
class FoldData:
    ds: Dataset                        # the pool's judges, cheapest first
    train: np.ndarray                  # labelled training items
    val: np.ndarray                    # labelled validation items, every call of the first `worlds` worlds recorded
    test: np.ndarray                   # the same for test items
    worlds: int
    seed: int
    y: np.ndarray                      # (N,) majority annotation (MISSING: unlabelled)
    scale: float                       # USD per circuit cost unit
    plan: np.ndarray                   # (M,) mean planning cost per call on the training items, circuit unit
    seconds: dict = field(default_factory=dict)   # wall-clock seconds per stage of the method running: "fit" (training
                                       # and setting it up on train and validation), "decide" (its decisions on the
                                       # validation and test rows at every operating point)

    @classmethod
    def build(cls, ds: Dataset, train, val, test, pool: list[str], worlds: int, seed: int) -> "FoldData":
        train, val, test = (np.asarray(a) for a in (train, val, test))
        train = train[ds.labelled[train]]
        mean_plan = ds.plan_cost_for(train)[train].mean(axis=0)
        cols = sorted((ds.judge_names.index(n) for n in pool), key=lambda j: mean_plan[j])   # stable: pool order on ties
        sub = ds.with_judges([ds.judge_names[j] for j in cols])
        if sub.answer_classes is not None:            # judges answering in their own alphabet: the methods read votes
            sub = replace(sub, answers=sub.votes(), answer_classes=None)
        done = sub.valid[:, :, :worlds].all(axis=(1, 2)) & sub.labelled
        scale = ds.scale()
        labels = ds.labels if ds.labels is not None else np.zeros((ds.n_items, ds.n_classes))
        K = ds.n_classes
        y = np.where(labels.sum(axis=1) > 0, K - 1 - labels[:, ::-1].argmax(axis=1), MISSING)
        return cls(ds=sub, train=train, val=val[done[val]], test=test[done[test]], worlds=int(worlds), seed=int(seed),
                   y=y, scale=scale, plan=mean_plan[cols] / scale)

    # ------------------------------------------------------------------ timing
    @contextmanager
    def stage(self, name: str):
        """Time a stage of the method running (the GPU's queued work finished at both ends)."""
        _sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            self.add(name, time.perf_counter() - started)

    def add(self, name: str, seconds: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + float(seconds)

    # ------------------------------------------------------------------ shape
    @property
    def M(self) -> int:
        return self.ds.n_judges

    @property
    def K(self) -> int:
        return self.ds.n_classes

    @property
    def judges(self) -> list[str]:
        return self.ds.judge_names

    @property
    def answers(self) -> np.ndarray:
        return self.ds.answers

    @property
    def valid(self) -> np.ndarray:
        return self.ds.valid

    @cached_property
    def cost(self) -> np.ndarray:
        """(N, M, R) each call's cost, circuit unit."""
        return self.ds.cost / self.scale

    @property
    def features(self) -> np.ndarray:
        if self.ds.features is None:
            raise NotApplicable("this method reads the prior features: run lmcircuit.embed (or set ds.features)")
        return self.ds.features

    def utility(self, items) -> np.ndarray:
        """(n, M, worlds) 1 where the call agrees with the majority annotation, 0 otherwise or without a call."""
        a = self.answers[np.asarray(items), :, :self.worlds]
        return ((a == self.y[np.asarray(items), None, None]) & (a != MISSING)).astype(np.float64)

    def quality(self, items) -> np.ndarray:
        """(n, M, worlds) the share of an item's annotations agreeing with each call (NaN without a call)."""
        items = np.asarray(items)
        a = self.answers[items, :, :self.worlds]
        share = self.ds.labels[items] / np.maximum(self.ds.labels[items].sum(axis=1, keepdims=True), 1e-12)
        q = np.take_along_axis(share[:, None, None, :], np.maximum(a, 0)[..., None], axis=-1)[..., 0]
        return np.where(a != MISSING, q, np.nan)

    def recorded(self, items) -> np.ndarray:
        """(n, M, worlds) which of the first `worlds` positions hold a call."""
        return self.valid[np.asarray(items), :, :self.worlds]

    def judge_accuracy(self, split: str = "train") -> np.ndarray:
        """(M,) each judge's agreement with the majority annotation over its recorded calls on `split`."""
        items = self.train if split == "train" else self.val
        u, ok = self.utility(items), self.recorded(items)
        return (u * ok).sum(axis=(0, 2)) / np.maximum(ok.sum(axis=(0, 2)), 1)

    def weak_strong(self) -> tuple[int, int]:
        """The pairwise methods' pair: the cheapest judge and the costlier judge most accurate on the training calls."""
        if self.M < 2:
            raise NotApplicable("a pairwise method needs at least two judges")
        acc = self.judge_accuracy("train")
        return 0, int(max(range(1, self.M), key=lambda j: acc[j]))

    def valid_pairs(self) -> list[tuple[int, int]]:
        """(cheap, expensive): lower planning cost and lower training accuracy for the cheap judge."""
        acc, c = self.judge_accuracy("train"), self.plan
        pairs = {(a, b) for a, b in combinations(range(self.M), 2) if c[a] < c[b] and acc[b] > acc[a]}
        return sorted(pairs, key=lambda p: (c[p[0]], c[p[1]]))

    def pairs(self, main: tuple[int, int], which: str) -> list[tuple[int, int]]:
        """The (weak, strong) pairs a pairwise method sweeps: its own `main` pair alone ("main"), or that pair and every
        valid pair ("all"); at each budget, validation then picks among all their operating points."""
        if which not in ("main", "all"):
            raise ValueError(f"pairs is 'main' or 'all', not {which!r}")
        return [main] + ([pr for pr in self.valid_pairs() if pr != main] if which == "all" else [])

    def train_worlds(self, cap: int | None = None) -> list[tuple[int, int]]:
        """Training (item, world) pairs with a recorded call of every judge, item major; `cap` of them drawn at random
        without replacement (seeded) when set."""
        ok = self.recorded(self.train).all(axis=1)                                     # (n, worlds)
        pairs = [(int(self.train[k]), int(r)) for k, r in zip(*np.nonzero(ok))]
        if cap is not None and cap < len(pairs):
            keep = np.sort(np.random.default_rng(self.seed).permutation(len(pairs))[:cap])
            pairs = [pairs[k] for k in keep]
        return pairs

    def full_train(self) -> np.ndarray:
        """Training items with at least one recorded call of every judge (where per-item means over calls exist)."""
        return self.train[self.recorded(self.train).any(axis=2).all(axis=1)]

    def mean_over_calls(self, values: np.ndarray, items) -> np.ndarray:
        """(n, M) the mean of (n, M, worlds) `values` over each judge's recorded calls."""
        ok = self.recorded(items)
        return np.where(ok, values, 0.0).sum(axis=2) / np.maximum(ok.sum(axis=2), 1)

    def repeats_recorded(self, judge: int) -> int:
        """Leading call positions of `judge` recorded on every validation and test item."""
        full = self.valid[np.concatenate([self.val, self.test]), judge].all(axis=0)
        return int(np.argmin(np.append(full, False)))

    # ------------------------------------------------------------------ outcomes
    def charged(self, items, pred: np.ndarray, queried: np.ndarray) -> Outcome:
        """pred (n, worlds); queried (n, M, worlds): the judges whose call of that world is paid."""
        items = np.asarray(items)
        cost = (queried * self.cost[items, :, :self.worlds]).sum(axis=1)
        return Outcome(pred.reshape(-1).astype(np.int8), cost.reshape(-1).astype(np.float32),
                       queried.sum(axis=(0, 2)) / max(queried.shape[0] * queried.shape[2], 1))

    def single(self, items, judge: np.ndarray) -> Outcome:
        """A single-call router: judge (n,) or (n, worlds) answers the item, its call of that world paid."""
        items = np.asarray(items)
        judge = np.broadcast_to(judge[:, None] if np.ndim(judge) == 1 else judge, (len(items), self.worlds))
        pred = np.take_along_axis(self.answers[items, :, :self.worlds], judge[:, None, :], axis=1)[:, 0]
        return self.charged(items, pred, np.arange(self.M)[None, :, None] == judge[:, None, :])

    def repeated(self, items, judge: int, n: int, repeats: int) -> tuple[np.ndarray, np.ndarray]:
        """The calls a method sampling one judge n times reads in world r: positions r, r + 1, ... (cyclically) of the
        judge's first `repeats` recorded calls. Returns answers and costs, each (len(items), worlds, n)."""
        items = np.asarray(items)
        pos = (np.arange(self.worlds)[:, None] + np.arange(n)[None, :]) % repeats              # (worlds, n)
        return self.answers[items[:, None, None], judge, pos[None]], self.cost[items[:, None, None], judge, pos[None]]

    def has_probs(self, judge: int) -> bool:
        """Whether some of `judge`'s calls recorded class probabilities that are not the answer itself."""
        if self.ds.probs is None:
            return False
        p = self.ds.probs[:, judge]
        p = p[np.isfinite(p).all(axis=-1)]
        return bool(len(p)) and bool(((p > 1e-9) & (p < 1 - 1e-9)).any())

    def sampled(self, items, judge: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        """n calls of `judge` in world r when fewer are recorded: the world's recorded call first, then n - 1 answers
        drawn from that call's class probabilities (Dataset.probs; the answer itself where none were recorded), each paid
        as the recorded call. Draws are seeded by the fold's seed, the judge and the item. Returns answers and costs,
        each (len(items), worlds, n)."""
        items = np.asarray(items)
        a = self.answers[items, judge, :self.worlds]                                          # (n_i, W)
        onehot = np.eye(self.K)[np.maximum(a, 0)]
        p = onehot if self.ds.probs is None else self.ds.probs[items, judge, :self.worlds].astype(np.float64)
        p = np.where(np.isfinite(p).all(axis=-1, keepdims=True), p, onehot)
        cdf = np.cumsum(p / p.sum(axis=-1, keepdims=True), axis=-1)
        u = np.stack([np.random.default_rng([self.seed, judge, int(i)]).random((self.worlds, n - 1)) for i in items])
        draws = np.minimum((u[..., None] > cdf[:, :, None, :]).sum(axis=-1), self.K - 1)       # inverse CDF
        cost = self.cost[items, judge, :self.worlds]
        return np.concatenate([a[..., None], draws], axis=-1), np.repeat(cost[..., None], n, axis=-1)


def _sync() -> None:
    import sys
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def majority(votes: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """votes (..., n): the most frequent class (ties: the first answer, in reading order, among the tied classes) and its
    share of the n answers. For two classes, a tie goes to the first answer."""
    counts = (votes[..., None] == np.arange(K)).sum(axis=-2)                                  # (..., K)
    top = counts.max(axis=-1)
    tied = np.take_along_axis(counts, votes, axis=-1) == top[..., None]                       # (..., n)
    first = np.take_along_axis(votes, np.argmax(tied, axis=-1)[..., None], axis=-1)[..., 0]
    return first, top / votes.shape[-1]
