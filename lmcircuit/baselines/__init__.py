"""Eight published routing and cascade methods, on exactly the circuit's folds, pools and test rows.

    report = lc.evaluate(ds, baselines="all")                         # the circuit and the baselines
    report = lc.baselines.evaluate(ds, "all", report=report)          # or added to an existing report
    lc.plot.figures(report, "figures/")                               # baselines drawn on accuracy

    routellm              Ong et al. (ICLR 2025), RouteLLM matrix-factorisation router; one pair, one call
    hybridllm             Ding et al. (ICLR 2024), Hybrid LLM r_prob router (fine-tuned DeBERTa-v3-large); one pair
    cascaded_selective    Jung et al. (ICLR 2025), Trust or Escalate: conformal cheapest-first cascade
    frugalgpt             Chen et al. (TMLR 2024), FrugalGPT LLM cascade (vendored library, fine-tuned DistilBERT scorers)
    cascade_routing       Dekoninck et al. (ICML 2025), cascade routing (vendored library)
    carrot                Somerstep et al. (2025), CARROT (KNN) router over every judge; one call
    consistency_cascade   Yue et al. (ICLR 2024), answer-consistency cascade (CoT-1D-Vote); needs >= 3 recorded calls
    adaptive_consistency  Aggarwal et al. (EMNLP 2023), Adaptive-Consistency, Beta criterion; needs >= 2 recorded calls

Implemented as in our experiments, with their settings (config.SETTINGS, config.SCORERS) and substitutions (each
module's docstring). Every method handles K classes. The fine-tuned scorers need the `baselines` extra
(transformers) and a GPU in practice; scipy and scikit-learn are needed by cascaded_selective, adaptive_consistency,
carrot and cascade_routing.
"""
from .common import FoldData, NotApplicable
from .config import METHODS, RULES, SCORERS, SETTINGS, BaselineConfig
from .evaluation import aggregate, evaluate, evaluate_fold, load_folds

__all__ = ["BaselineConfig", "FoldData", "METHODS", "NotApplicable", "RULES", "SCORERS", "SETTINGS", "aggregate",
           "evaluate", "evaluate_fold", "load_folds"]
