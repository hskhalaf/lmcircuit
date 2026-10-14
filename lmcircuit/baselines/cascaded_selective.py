"""Cascaded Selective Evaluation (Jung et al., ICLR 2025, "Trust or Escalate: LLM Judges with Provable Guarantees for
Human Agreement"; jaehunjung1/cascaded-selective-evaluation).

The paper's conformal "Learn then Test" calibration (SelectiveClassificationUtil in cascaded_evaluation/util.py:
candidate lambdas, a Clopper-Pearson selective-risk upper bound via binom.cdf + brentq, the monotone lambda search of
lambda_hat) and its sequential per-model calibration (CascadedClassifier.calibrate / apply_decision_rule), ported
verbatim from the authors' code. Judges are asked cheapest first in a fixed order, as in the paper's
Mistral-7B -> GPT-3.5 -> GPT-4 cascade; a judge's verdict is accepted when its confidence clears its threshold. The
risk tolerance alpha is swept (linspace(0.02, 0.45, 20), delta = 0.1); thresholds are calibrated on the validation
worlds (the paper's calibration set, every world one calibration example) against the majority annotation.

Substitutions (made in our experiments):
* confidence: the judge's probability of its own verdict at the verdict token (Dataset.probs at the answer; the
  paper's predictive-probability estimate). The paper's main estimate, simulated annotators, needs several differently
  prompted calls per judge. A call whose probability was not recorded has confidence 1;
* no human escalation: an item no judge accepts is answered by the last (most expensive) judge.
Guards the paper code lacks (added in our experiments): a judge with fewer than 30 unclaimed calibration worlds
and no cheaper threshold gets 1.000001 (never accept); when every accepted world is wrong the risk bound is 1.
"""
from __future__ import annotations

import numpy as np

from .common import FoldData, Point, Result

SUBSTITUTIONS = [
    "phat: the judge's own probability of its verdict at the verdict token (the paper's predictive-probability "
    "confidence); 1 where not recorded",
    "no human escalation: an item unresolved at the most expensive judge is answered by it",
]
NEVER_ACCEPT = 1.000001


class SelectiveClassificationUtil:
    """Verbatim port of cascaded_evaluation/util.py (numpy instead of torch)."""

    def __init__(self, cal_phats, cal_yhats, cal_labels, delta):
        self.cal_phats = cal_phats
        self.cal_yhats = cal_yhats
        self.cal_labels = cal_labels
        self.delta = delta
        self.lambdas = np.array([lam for lam in np.linspace(0, 1, 5000) if self.n_lambda(lam) >= 30])

    def n_lambda(self, lam):
        return int((self.cal_phats >= lam).sum())

    def selective_risk(self, lam):
        mask = self.cal_phats >= lam
        return (self.cal_yhats[mask] != self.cal_labels[mask]).sum() / mask.sum()

    def selective_risk_upper_bound(self, lam):
        from scipy.optimize import brentq
        from scipy.stats import binom

        def condition(r, lam):
            return binom.cdf(self.selective_risk(lam) * self.n_lambda(lam), self.n_lambda(lam), r) - self.delta
        try:
            return brentq(condition, 0, 0.9999, args=(lam,))
        except ValueError:      # every accepted world is wrong: no sign change, the bound is trivially 1
            return 1.0

    def lambda_hat(self, alpha):
        if len(self.lambdas) == 0:
            return 0.
        if self.selective_risk_upper_bound(self.lambdas[-1]) > alpha:
            return self.lambdas[-1]
        for lam in self.lambdas[::-1]:
            next_lam = lam - 1 / self.lambdas.size
            if self.selective_risk_upper_bound(next_lam) > alpha:
                return lam
        return self.lambdas[0]


def calibrate(phat, yhat, labels, alpha: float, delta: float) -> list[float]:
    """CascadedClassifier.calibrate: per judge in order, on the calibration worlds no cheaper judge has claimed."""
    lambda_hats, claimed = [], np.zeros(len(labels), dtype=bool)
    for j in range(phat.shape[1]):
        util = SelectiveClassificationUtil(phat[~claimed, j], yhat[~claimed, j], labels[~claimed], delta)
        lam = (lambda_hats[-1] if lambda_hats else NEVER_ACCEPT) if len(util.lambdas) == 0 else util.lambda_hat(alpha)
        lambda_hats.append(float(lam))
        claimed |= phat[:, j] >= lam
    return lambda_hats


def apply_decision_rule(phat, yhat, lambda_hats):
    """CascadedClassifier.apply_decision_rule, the last judge answering instead of abstaining."""
    n, M = phat.shape
    pred, evaluator = np.full(n, -1, dtype=np.int64), np.full(n, -1)
    for j in range(M):
        accept = (phat[:, j] >= lambda_hats[j]) & (evaluator < 0)
        evaluator[accept], pred[accept] = j, yhat[accept, j]
    rest = evaluator < 0
    evaluator[rest], pred[rest] = M - 1, yhat[rest, M - 1]
    return pred, evaluator


def worlds(fd: FoldData, items):
    """phat, yhat (n * worlds, M) and labels (n * worlds,), item major."""
    W = fd.worlds
    yhat = fd.answers[items, :, :W]                                                            # (n, M, W)
    p = np.full(yhat.shape, np.nan) if fd.ds.probs is None else \
        np.take_along_axis(fd.ds.probs[items, :, :W], yhat[..., None], axis=-1)[..., 0]
    phat = np.where(np.isfinite(p), p, 1.0)
    flat = lambda a: a.transpose(0, 2, 1).reshape(len(items) * W, fd.M)  # noqa: E731
    return flat(phat), flat(yhat), np.repeat(fd.y[items], W)


def run(fd: FoldData, p: dict, cfg) -> Result:
    with fd.stage("fit"):
        cal = worlds(fd, fd.val)
    with fd.stage("decide"):
        data = {"val": cal, "test": worlds(fd, fd.test)}
    points = []
    for alpha in np.linspace(0.02, 0.45, int(p["alphas"])):
        with fd.stage("fit"):
            lambda_hats = calibrate(*cal, float(alpha), float(p["delta"]))
        out = {}
        with fd.stage("decide"):
            for name, items in (("val", fd.val), ("test", fd.test)):
                pred, evaluator = apply_decision_rule(data[name][0], data[name][1], lambda_hats)
                evaluator = evaluator.reshape(len(items), fd.worlds)
                out[name] = fd.charged(items, pred.reshape(len(items), fd.worlds),
                                       np.arange(fd.M)[None, :, None] <= evaluator[:, None, :])
        points.append(Point({"alpha": float(alpha), "lambda_hats": lambda_hats}, out["val"], out["test"]))
    recorded = 0.0 if fd.ds.probs is None else float(np.isfinite(fd.ds.probs[fd.val, :, :fd.worlds, 0]).mean())
    return Result(points, {"reference": "Jung et al., Trust or Escalate (jaehunjung1/cascaded-selective-evaluation), "
                                        "SelectiveClassificationUtil + CascadedClassifier.calibrate/apply_decision_rule",
                           "substitutions": SUBSTITUTIONS, "order": fd.judges, "calibration": "validation worlds",
                           "share_of_calls_with_probabilities": recorded})
