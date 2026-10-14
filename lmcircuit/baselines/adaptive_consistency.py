"""Adaptive-Consistency (Aggarwal et al., EMNLP 2023, "Let's Sample Step by Step: Adaptive-Consistency for Efficient
Reasoning and Coding with LLMs").

One model is sampled one answer at a time, and sampling stops as soon as the majority answer is stable: after each
sample, stop if P(p1 > p2 | v) > C, where v1 >= v2 are the counts of the two most frequent answers and, under the
paper's Beta criterion (its default), P = P(p2 < 1/2) for p2 ~ Beta(v2 + 1, v1 + 1). The answer is the majority. C is
swept over [0.5, 1) as in the paper's analysis (its default is 0.95). No training or validation data in the fit.

Substitutions (made in our experiments):
* at most min(8, the judge's recorded calls) samples (the paper allows 40; our recorded datasets have 8);
* the model: the pool judge most accurate on the training calls (the method uses one model; settings `judge` names
  another);
* world r reads the judge's calls from position r onward (cyclically over the calls recorded on every validation and
  test item); a tie at the cap goes to the first answer among the tied classes.
Needs at least 2 recorded calls of the judge (one call cannot stop adaptively). With settings `simulate`, a judge with
one recorded call but recorded class probabilities is sampled instead: its recorded call first, the other samples drawn
from that call's probabilities (FoldData.sampled), up to max_samples; otherwise raises NotApplicable.
"""
from __future__ import annotations

import numpy as np

from .common import FoldData, NotApplicable, Outcome, Point, Result, majority

SUBSTITUTIONS = [
    "at most min(8, recorded calls) samples instead of 40",
    "one judge: the judge most accurate on the training calls",
    "with one recorded call (settings `simulate`): the further samples drawn from the call's class probabilities",
]
MIN_REPEATS = 2


def stability(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """The Beta criterion: P(p2 < 1/2) for p2 ~ Beta(v2 + 1, v1 + 1), v1 >= v2 the counts of the top two answers."""
    from scipy.stats import beta
    return beta.cdf(0.5, v2 + 1, v1 + 1)


def stop_times(votes: np.ndarray, threshold: float, K: int) -> tuple[np.ndarray, np.ndarray]:
    """votes (..., n) read in order: the number of samples drawn (1..n) and the majority answer when sampling stops."""
    counts = np.cumsum(votes[..., None] == np.arange(K), axis=-2)                           # (..., n, K) after t + 1
    top2 = -np.sort(-counts, axis=-1)[..., :2]                                              # v1 >= v2
    stop = stability(top2[..., 0], top2[..., 1]) > threshold
    stop[..., -1] = True                                                                     # the cap
    drawn = np.argmax(stop, axis=-1) + 1
    answer = np.empty(drawn.shape, dtype=np.int64)
    for t in np.unique(drawn):
        at = drawn == t
        answer[at] = majority(votes[at][..., :t], K)[0]
    return drawn, answer


def run(fd: FoldData, p: dict, cfg) -> Result:
    name = p.get("judge") or "best_train"
    with fd.stage("fit"):          # the judge, from the training calls (nothing else is learned)
        j = int(np.argmax(fd.judge_accuracy("train"))) if name == "best_train" else fd.judges.index(name)
    repeats = fd.repeats_recorded(j)
    simulated = repeats < MIN_REPEATS
    if simulated and not (p.get("simulate") and fd.has_probs(j)):
        raise NotApplicable(f"adaptive_consistency needs {MIN_REPEATS} recorded calls of {fd.judges[j]} on every "
                            f"evaluated item, or its class probabilities to sample from ({repeats})")
    cap = int(p["max_samples"]) if simulated else min(int(p["max_samples"]), repeats)
    points = []
    with fd.stage("decide"):
        data = {s: (fd.sampled(items, j, cap) if simulated else fd.repeated(items, j, cap, repeats))
                for s, items in (("val", fd.val), ("test", fd.test))}
        for c in p["thresholds"]:
            out = {}
            for s, (votes, cost) in data.items():
                drawn, answer = stop_times(votes, float(c), fd.K)
                calls = np.zeros(fd.M)
                calls[j] = drawn.mean()
                paid = (cost * (np.arange(cap) < drawn[..., None])).sum(axis=-1)
                out[s] = Outcome(answer.reshape(-1).astype(np.int8), paid.reshape(-1).astype(np.float32), calls)
            points.append(Point({"threshold": float(c), "mean_samples": float(out["test"].calls[j])},
                                out["val"], out["test"]))
    return Result(points, {"reference": "Aggarwal et al., Let's Sample Step by Step: Adaptive-Consistency for Efficient "
                                        "Reasoning and Coding with LLMs (EMNLP 2023), Beta stopping criterion",
                           "substitutions": SUBSTITUTIONS, "judge": fd.judges[j], "max_samples": cap,
                           "simulated_repeats": bool(simulated)})
