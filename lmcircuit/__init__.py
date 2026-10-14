"""lmcircuit: ask language-model judges one at a time, and stop when the answer is worth its cost.

    import lmcircuit as lc
    ds = lc.embed(lc.datasets.load("oll_bbh", raw_dir))            # or lc.Dataset.from_records(...)
    report = lc.evaluate(ds, baselines="all", out="runs/oll_bbh")  # cross-fitted frontiers, two objectives, saved
    lc.plot.figures(report, "figures/oll_bbh", "OLL BBH")

examples/lmcircuit.ipynb: your own dataset (lc.collect), and using one fitted lc.Circuit.
"""
from importlib.metadata import PackageNotFoundError, version

from . import baselines, datasets, plot
from .circuit import Circuit, Decision, Policy
from .collection import JudgeSpec, collect
from .config import BeliefConfig, CircuitConfig, EmbedConfig, EvalConfig
from .data import MISSING, Dataset, Judge
from .embedding import embed
from .evaluation import aggregate, evaluate, evaluate_fold, load_folds, save_report
from .labelling import label
from .llm import Backend, Prompt
from .metrics import scores

try:
    __version__ = version("lmcircuit")
except PackageNotFoundError:          # running from a source tree
    __version__ = "0.0.0+local"

__all__ = ["Backend", "BeliefConfig", "Circuit", "CircuitConfig", "Dataset", "Decision", "EmbedConfig", "EvalConfig",
           "Judge", "JudgeSpec", "MISSING", "Policy", "Prompt", "aggregate", "baselines", "collect", "datasets", "embed",
           "evaluate", "evaluate_fold", "label", "load_folds", "plot", "save_report", "scores"]
