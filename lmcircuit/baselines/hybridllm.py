"""Hybrid LLM (Ding et al., ICLR 2024, "Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing").

A pairwise, single-call router: each item goes to the small (weak) or the large (strong) model, never both. The router
is the paper's probabilistic one, r_prob: DeBERTa-v3-large reads the query and predicts Pr[q(weak) >= q(strong)],
trained with binary cross-entropy on labels estimated from repeated responses, 5 epochs, the checkpoint with the
lowest validation loss (scorers.hybridllm_router, fine-tuned on the fold's training items). An item goes to the weak
model when that probability is at least a threshold. The paper sets the threshold for a target "cost advantage" (the
share of queries sent to the weak model) from 500 validation queries; the curve is those targets from 0% to 100%
(200 of them), each threshold applied to the validation and test items.

The pair: weak = the cheapest pool judge, strong = the costlier judge most accurate on the training calls (the paper
studies fixed pairs of a small and a large model; one DeBERTa fine-tune per fold, so other pairs are not swept).

Substitutions (made in our experiments): the quality q of a response is the verdict's agreement with the
annotations (1 or 0 under one annotation, else the share agreeing), not a BART score of generated text, so
Pr[q(weak) >= q(strong)] is the share of (weak call, strong call) pairs in which the weak call is at least as good; the
repeated responses are the recorded calls of the first `worlds` worlds instead of 10 sampled generations.
"""
from __future__ import annotations

import numpy as np

from . import scorers
from ..config import resolve_device
from .common import FoldData, Point, Result

SUBSTITUTIONS = [
    "quality of a response = the verdict's agreement with the annotations instead of a BART score; the probabilistic "
    "label is the share of recorded (weak, strong) call pairs with q_weak >= q_strong",
    "repeated responses = the recorded calls instead of 10 sampled generations",
    "pair: the cheapest judge and the costlier judge most accurate on the training calls",
]


def run(fd: FoldData, p: dict, cfg) -> Result:
    with fd.stage("fit"):          # the router fine-tuned, then its scores on every item (inference counted here)
        weak, strong = fd.weak_strong()
        prob, report = scorers.hybridllm_router(fd, weak, strong, cfg.scorer("hybridllm"), resolve_device(cfg.device),
                                                cfg.text_parts)
        rng = np.random.default_rng(fd.seed)
        calibration = rng.choice(fd.val, size=min(int(p["calibration_items"]), len(fd.val)), replace=False)
        targets = np.linspace(0.0, 1.0, int(p["thresholds"]))      # cost advantage: the share of queries sent to weak
        taus = [np.inf if t == 0 else float(np.quantile(prob[calibration], 1 - t, method="higher")) for t in targets]
    points = []
    with fd.stage("decide"):
        for target, tau in zip(targets, taus):
            out = {name: fd.single(items, np.where(prob[items] >= tau, weak, strong))
                   for name, items in (("val", fd.val), ("test", fd.test))}
            points.append(Point({"cost_advantage_target": float(target), "tau": float(tau),
                                 "weak_route_rate": float((prob[fd.test] >= tau).mean())}, out["val"], out["test"]))
    return Result(points, {"reference": "Ding et al., Hybrid LLM (ICLR 2024): r_prob router, DeBERTa-v3-large, "
                                        "thresholds calibrated on 500 validation queries per target cost advantage",
                           "substitutions": SUBSTITUTIONS, "pair": [fd.judges[weak], fd.judges[strong]],
                           "calibration_items": int(len(calibration)), "router": report,
                           "scorer_seconds": report["seconds"]})
