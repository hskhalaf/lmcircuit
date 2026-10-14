# lmcircuit

Ask language-model judges one at a time, and stop when the answer is worth its cost.

You have items to classify and several judges (LLMs under a prompt, reward models, classifiers) that answer them at
different costs. `lmcircuit` learns a belief model of the label and the judges' answers and solves an exact dynamic
program: for each item, it decides which judge to ask next, or to stop and answer. The same belief model serves
two objectives (accuracy and balanced accuracy); each circuit's cost frontier is scored on every metric, against
each judge alone, majority voting and eight published routing and cascade methods, on held-out items.

```python
import lmcircuit as lc

lc.datasets.fetch("oll_bbh", "data/oll_bbh")                  # 46 open models' answers on eight BBH tasks
ds = lc.embed(lc.datasets.load("oll_bbh", "data/oll_bbh"))     # prior features of each item
report = lc.evaluate(ds, baselines="all", out="runs/oll_bbh")   # 5 grouped folds x 5 seeds, every item tested once
lc.plot.figures(report, "figures/oll_bbh", "OLL BBH")            # accuracy.pdf, balanced_accuracy.pdf, calibration.pdf
```

## Install

```bash
pip install "lmcircuit[all]"        # or the extras you need: datasets, llm, embed, plot, baselines
```

The method needs only numpy and torch; a GPU makes the dynamic program much faster.

## Example

[examples/lmcircuit.ipynb](examples/lmcircuit.ipynb): pick a registered dataset and inspect it (its judges, how much
data, examples), or bring your own questions with ground truth and the Hugging Face models to choose between (each
under a shared or its own system prompt, sampled as often as you like); then evaluate both objectives, draw the
figures, and use one circuit item by item.

## What is where

```
lmcircuit/
  data.py          Dataset, Judge: items, judges, recorded answers and costs, labels, grouped folds, save / load
  llm.py           Backend (a vLLM server or the OpenAI API), Prompt, calling and parsing
  labelling.py     lc.label: labels from grader models
  collection.py    lc.collect: judges' answers, costs and verdict probabilities
  embedding.py     lc.embed: prior features
  belief.py        the belief model: the BeliefNet within latent item types (orders and the A/B swap for pairs)
  dp.py            the exact dynamic program over the answers seen so far
  objectives.py    accuracy, balanced accuracy: class weights for the DP, and how an operating point is chosen
  pool.py          which judges a circuit may ask
  circuit.py       lc.Circuit: fit, choose an objective and a budget, decide (on recorded answers or live), save / load
  evaluation.py    lc.evaluate: cross-fitting over seeds, references, calibration, frontier areas, times, saved folds
  metrics.py       accuracy, balanced accuracy and Brier score against annotation counts; resampled intervals
  plot.py          the figures of reports (the frontier figure for any set of datasets, calibration)
  config.py        every default of the method, with the evidence behind it
  baselines/       eight published methods on the same folds and test items (one file each)
  datasets/        registered benchmarks (one file each): lc.datasets.fetch / load
examples/          the notebook above
docs/method.md     the method, its defaults, and why
```

## Datasets

`lc.datasets.REGISTRY` holds the paper's seven datasets. Three come with their judges' recorded answers: Decision
Index, OLL BBH and AgentJudgeBench. For the other four (PolitiCAUSE, BeaverTails, LMArena, HealthBench), `load` gives
the items, and each module's `recipe` reruns the paper's judges with `lc.collect` against your own vLLM servers:

```python
lc.datasets.fetch("politicause", "data/politicause")
ds = lc.datasets.load("politicause", "data/politicause")                  # the items, no judges yet
specs = lc.datasets.politicause.recipe({"Qwen3-8B": "http://localhost:8000/v1", ...})   # one server per model
ds = lc.collect(ds, specs, repeats=8)                                     # answers, costs, verdict probabilities
```

Each loader's docstring gives its source, licence, items, judges and costs.

## Baselines

RouteLLM, Hybrid LLM, Trust or Escalate, FrugalGPT, cascade routing, CARROT, the answer-consistency cascade and
Adaptive-Consistency: `lc.evaluate(ds, baselines="all")`. Each runs on the circuit's folds, pools and test items, with
the settings of its paper (`lc.baselines`; each module's docstring lists what we had to substitute). The two
repeated-call methods sample further calls from the recorded verdict probabilities when a judge has one recorded
call.

## License

MIT (`LICENSE`). The vendored FrugalGPT and cascade-routing code under
`lmcircuit/baselines/_vendor/` keeps its own licences. Each dataset's licence is in its loader's docstring.
