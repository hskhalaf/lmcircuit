"""Every choice of the method in one place, with the evidence behind each default (docs/method.md, section 4)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BeliefConfig:
    types: tuple[int, ...] = (1, 2, 4, 8)   # numbers of latent item types tried; the one with the lowest validation loss
                                     # is kept (4 types raised the frontier area over 1 on OLL BBH 70.4 -> 74.1,
                                     # Decision Index 77.3 -> 78.0, PolitiCAUSE 78.3 -> 79.1, 5/5 folds each)
    d_model: int = 64                 # width of the text projection and of every (judge, answer) token
    d_hidden: int = 128               # width of the label trunk and of the type and judge heads
    lr: float = 2e-3                  # AdamW, cosine schedule over `epochs`
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 300                 # cap; early stopping on the validation label loss
    patience: int = 30
    alpha: float = 1.0                # weight of the judges' loss against the label's
    calibration_states: int = 10      # validation states per item the two temperatures are fitted on


@dataclass(frozen=True)
class CircuitConfig:
    pool_size: int = 8                # the DP is exact over (K+1)^pool states ...
    max_states: int = 6561            # ... so the pool is also capped at (K+1)^pool <= max_states (3^8: binary keeps 8
                                      # judges, 3 classes 6, 4 classes 5)
    lambdas: int = 201                # geometric grid of cost prices (20 per decade), plus 0 and infinity. Error is in
    lambda_lo: float = 1e-8           # [0, 1] and one call removes at most 1 - 1/K of it, so a judge costing c (the
    lambda_hi: float = 100.0          # priciest's mean call = 1) is never asked above lambda = (1 - 1/K) / c: 100
                                      # reaches the cheapest pool judges (c >= 0.005); below 1e-8 a call's price is
                                      # negligible against any change in the error
    target: str = "accuracy"          # the objective a fitted circuit starts with (lmcircuit.objectives)
    max_cells: float | None = None    # rows x lambdas x states per batched DP call; None: from the free GPU memory
                                      # (shared by the folds that run at once)
    device: str = "auto"              # "auto": cuda when available
    belief: BeliefConfig = field(default_factory=BeliefConfig)

    def pool_for(self, n_answers: int) -> int:
        """The pool size: pool_size, cut until the (A+1)^pool states of judges answering one of A values fit."""
        size = self.pool_size
        while size > 1 and (n_answers + 1) ** size > self.max_states:
            size -= 1
        return size


@dataclass(frozen=True)
class EvalConfig:
    folds: int = 5                    # grouped cross-validation: part f test, the other groups train and validation ...
    val_share: float = 0.2            # ... a random share of them validation
    shuffle_seed: int = 0
    worlds: int | None = None         # recorded calls per judge used for every item (default: min(5, recorded))
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)   # training seeds of the belief model (the folds stay fixed); curves
                                      # are the mean over them
    area_lo: float = 0.003            # frontier area: the metric at budget 0 and `area_count` log-spaced budgets
    area_hi: float = 1.0
    area_count: int = 50
    n_jobs: int = 1                   # (fold, seed) fits run at once, as processes sharing the GPU (1: one after another,
                                      # in this process; the recorded times are then each method's alone)


REVISIONS = {"Qwen/Qwen3-Embedding-0.6B": "a579a21d7aff542145eebef8d60ed73ec281a0b4"}   # pinned embedding models


@dataclass(frozen=True)
class EmbedConfig:
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    revision: str | None = None       # None: the pinned revision of a known model (REVISIONS), else the latest
    max_seq_length: int = 4096        # tokens of an item's text (its task description, then the item); at 512, the
                                      # planned calls of 89% of AgentJudgeBench items were cut off (the judged part)
    batch_size: int = 32              # texts per batch at most ...
    batch_tokens: int = 32768         # ... and at most this many padded tokens (long texts go in small batches)
    instruction: str = ("Given an item of a classification task, represent it for predicting its correct class and "
                        "which language models classify it correctly")

    def resolved_revision(self) -> str | None:
        return self.revision or REVISIONS.get(self.model)


def memory_cells(max_cells: float | None, device: str) -> float:
    """The DP's cell budget: max_cells, else half the free memory of the GPU at about 24 bytes a cell (values,
    actions and their temporaries), or 5e7 on the CPU."""
    if max_cells:
        return float(max_cells)
    import torch
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.mem_get_info(torch.device(device))[0] / 2 / 24
    return 5e7


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
