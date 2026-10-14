"""The published baselines' settings: the values of our experiments, each method's own knob and sweep, and how an
operating point is chosen at a budget."""
from __future__ import annotations

from dataclasses import dataclass, field

METHODS = ("routellm", "hybridllm", "frugalgpt", "cascaded_selective", "cascade_routing", "carrot", "consistency_cascade",
           "adaptive_consistency")

SETTINGS: dict[str, dict] = {
    # pairs (the two methods that route between two judges cheaply): "all" sweeps every valid (cheap, expensive) pair
    # besides their own and lets validation pick the pair at each budget, as it picks every other setting
    "routellm": {"dimension": 16, "thresholds": 200, "epochs": 30, "batch_size": 4096, "pairs": "all"},
    "hybridllm": {"thresholds": 200, "calibration_items": 500},
    "cascaded_selective": {"alphas": 20, "delta": 0.1},
    # the vendored objective is a Python loop over training worlds: 2000 of them (our experiments' cap)
    "frugalgpt": {"budgets": 12, "train_cap": 2000, "chain_length": 3},
    "cascade_routing": {"budgets": 12, "train_cap": 2000, "lambda_iterations": 20},
    "carrot": {"neighbors": 40, "lambdas": 200, "lam_lo": 1e-3, "lam_hi": 1e3},     # CARROT (KNN), RouterBench's k
    "consistency_cascade": {"weak_samples": 8, "strong_samples": 3, "tau_lo": 0.4, "thresholds": 13,   # paper: K = 20
                            "pairs": "all", "simulate": True},   # simulate: sample a judge with too few recorded
    "adaptive_consistency": {"judge": "best_train", "max_samples": 8,  # calls from its class probabilities (both methods)
                             "thresholds": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999],  # paper: up to 40
                             "simulate": True},
}

# The fine-tuned scorers (scorers.py), trained per fold on its training items. `model`: a local checkpoint directory;
# None resolves model_id at `revision` in the Hugging Face cache (no download).
SCORERS: dict[str, dict] = {
    "frugalgpt": {"model": None, "model_id": "distilbert/distilbert-base-uncased",
                  "revision": "12040accade4e8a0f71eabdb258fecc2e7e948be", "num_labels": 2, "epochs": 8, "batch_size": 8,
                  "lr": 5e-5, "warmup_steps": 500, "weight_decay": 0.01, "holdout": 0.55, "max_length": 512,
                  "seed": 2024},                     # the released scorer's settings (lr: the HF Trainer default)
    "hybridllm": {"model": None, "model_id": "microsoft/deberta-v3-large",
                  "revision": "64a8c8eab3e352a784c658aef62be1662607476f", "num_labels": 1, "epochs": 5, "batch_size": 16,
                  "lr": 1e-5, "warmup_ratio": 0.1, "weight_decay": 0.01, "max_length": 512,
                  "seed": 0},                        # paper: DeBERTa-v3-large, 5 epochs, best validation checkpoint;
                                                     # learning rate, batch and warm-up not reported (ours)
}

# How the operating point at a budget is chosen, per fold and metric (aggregate): "cost", the point with the best
# validation value of the metric among those whose validation cost is within the budget, as for the circuit;
# "budget", the point optimised for the largest budget within it (the method's own contract: FrugalGPT's
# LLMChain.setbudget, cascade routing's max_expected_cost, both bounding the expected planning cost on the training data).
RULES = {m: "budget" if m in ("frugalgpt", "cascade_routing") else "cost" for m in METHODS}


@dataclass(frozen=True)
class BaselineConfig:
    params: dict = field(default_factory=dict)       # method -> {setting: value}, over SETTINGS
    scorers: dict = field(default_factory=dict)      # "frugalgpt" | "hybridllm" -> {setting: value}, over SCORERS
    seed: int = 42                                   # our experiments' seed (the same in every fold; the baselines
                                                     # run with this one seed)
    workers: int = 1                                 # forked processes for the budget sweeps (frugalgpt, cascade_routing)
    device: str = "auto"                             # the scorers' device ("auto": cuda when available)
    text_parts: tuple[str, ...] = ("task_description", "item")   # prior_text parts the scorers read, in this order

    def of(self, method: str) -> dict:
        return {**SETTINGS[method], **self.params.get(method, {})}

    def scorer(self, kind: str) -> dict:
        return {**SCORERS[kind], **self.scorers.get(kind, {})}
