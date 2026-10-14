"""The answer-consistency cascade of Yue et al. (ICLR 2024, "Large Language Model Cascades with Mixture of Thoughts
Representations for Cost-efficient Reasoning"), vote-based decision with one prompt (their CoT-1D-Vote).

The weak model is sampled K times; its majority answer A is accepted when the agreement score s = (number of the K
answers equal to A) / K is at least a threshold tau; otherwise the item goes to the strong model, answered by
self-consistency over 3 samples (their GPT-4 setting). tau is swept from 0.4 to 1, the paper's range. No training or
validation data in the fit (the pair is chosen on the training calls).

Substitutions (made in our experiments):
* K = min(8, the weak judge's recorded calls) samples (the paper samples K = 20; our recorded datasets have 8);
* the pair: the cheapest pool judge (weak) and the costlier judge most accurate on the training calls (strong), as for
  HybridLLM (the paper uses GPT-3.5 and GPT-4); with pairs "all" (the default) every valid (cheap, expensive) pair with
  enough recorded calls too, and validation picks the pair at each budget as it picks the threshold;
* the mixtures of thought representations and demonstration sets do not apply to a one-word verdict from one prompt;
* world r reads the weak judge's calls from position r onward and the strong judge's three calls from position r
  (cyclically over the calls recorded on every validation and test item); every call is paid. A tie goes to the first
  answer among the tied classes.
Needs at least 3 recorded calls per judge (the strong judge's three samples). With settings `simulate`, a judge with
fewer recorded calls but recorded class probabilities is sampled instead: its recorded call first, the other samples
drawn from that call's probabilities (FoldData.sampled), K = weak_samples; otherwise raises NotApplicable.
"""
from __future__ import annotations

import numpy as np

from .common import FoldData, NotApplicable, Outcome, Point, Result, majority

SUBSTITUTIONS = [
    "K = min(8, recorded calls) samples of the weak judge instead of 20",
    "weak = the cheapest judge, strong = the costlier judge most accurate on the training calls (paper: GPT-3.5, GPT-4)",
    "one prompt and one representation (CoT-1D-Vote); the thought-representation mixtures do not apply to a verdict",
    "with one recorded call (settings `simulate`): the further samples drawn from the call's class probabilities",
]
MIN_REPEATS = 3


def run(fd: FoldData, p: dict, cfg) -> Result:
    with fd.stage("fit"):          # the pair, from the training calls (nothing else is learned)
        weak, strong = fd.weak_strong()
        pairs = fd.pairs((weak, strong), p.get("pairs", "main"))
    k_strong, need = int(p["strong_samples"]), max(MIN_REPEATS, int(p["strong_samples"]))
    usable = lambda j: fd.repeats_recorded(j) >= need or (p.get("simulate") and fd.has_probs(j))  # noqa: E731
    if not (usable(weak) and usable(strong)):
        raise NotApplicable(f"consistency_cascade needs {need} recorded calls of the weak and the "
                            f"strong judge on every evaluated item, or their class probabilities to sample from "
                            f"({fd.judges[weak]}: {fd.repeats_recorded(weak)}, "
                            f"{fd.judges[strong]}: {fd.repeats_recorded(strong)})")
    pairs = [(w, s) for w, s in pairs if usable(w) and usable(s)]
    simulated = sorted({fd.judges[j] for pair in pairs for j in pair if fd.repeats_recorded(j) < need})
    points = []

    def draw(items, j, n):
        r = fd.repeats_recorded(j)
        return fd.repeated(items, j, n, r) if r >= n else fd.sampled(items, j, n)

    with fd.stage("decide"):
        for w, s in pairs:
            r_w = fd.repeats_recorded(w)
            k_weak, data = (min(int(p["weak_samples"]), r_w) if r_w >= need else int(p["weak_samples"])), {}
            for name, items in (("val", fd.val), ("test", fd.test)):
                wv, wc = draw(items, w, k_weak)                                            # (n, W, k)
                sv, sc = draw(items, s, k_strong)
                (wa, agree), (sa, _) = majority(wv, fd.K), majority(sv, fd.K)
                data[name] = (wa, agree, sa, wc.sum(axis=2), sc.sum(axis=2))
            for tau in np.linspace(float(p["tau_lo"]), 1.0, int(p["thresholds"])):
                out = {}
                for name, (wa, agree, sa, wcost, scost) in data.items():
                    accept = agree >= tau - 1e-12
                    calls = np.zeros(fd.M)
                    calls[w] += k_weak
                    calls[s] += k_strong * float((~accept).mean())
                    out[name] = Outcome(np.where(accept, wa, sa).reshape(-1).astype(np.int8),
                                        (wcost + np.where(accept, 0.0, scost)).reshape(-1).astype(np.float32), calls)
                points.append(Point({"pair": [fd.judges[w], fd.judges[s]], "tau": float(tau)}, out["val"], out["test"]))
    return Result(points, {"reference": "Yue et al., Large Language Model Cascades with Mixture of Thoughts "
                                        "Representations for Cost-efficient Reasoning (ICLR 2024), CoT-1D-Vote",
                           "substitutions": SUBSTITUTIONS, "pair": [fd.judges[weak], fd.judges[strong]],
                           "pairs": [[fd.judges[w], fd.judges[s]] for w, s in pairs],
                           "weak_samples": int(p["weak_samples"]), "strong_samples": k_strong,
                           "simulated_repeats": simulated})
