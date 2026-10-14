# Running lmcircuit: notes for coding agents

`lmcircuit` decides, item by item, which language-model judges to ask and when to stop. You give it items with
labels and recorded judge answers; it fits a belief model and solves an exact dynamic program, then evaluates the
resulting circuits against single judges, majority voting and eight published baselines. The method is described in
`docs/method.md`; every public function has a docstring that states its inputs and outputs.

## Setup

```bash
pip install "lmcircuit[all]"      # or the extras you need: datasets, llm, embed, plot, baselines
```

- The method itself needs only numpy and torch. A GPU makes the dynamic program much faster; everything also runs on
  CPU.
- `lc.embed` downloads Qwen3-Embedding-0.6B from the Hugging Face Hub on first use.
- `lc.collect` and `lc.label` call judges over HTTP: a vLLM server (`vllm serve <model>`) or the OpenAI API (key in
  `OPENAI_API_KEY` or `~/.config/openai/key`).

## The pipeline

```python
import lmcircuit as lc

ds = ...                                   # 1. a Dataset: items, labels, judges' recorded answers and costs
ds = lc.embed(ds)                          # 2. prior features of each item (frozen encoder)
report = lc.evaluate(ds, out="runs/x")     # 3. cross-fitted evaluation; add baselines="all" for the eight baselines
lc.plot.figures(report, "figures/x", "x")  # 4. accuracy.pdf, balanced_accuracy.pdf, calibration.pdf
```

### 1. Getting a Dataset

- **A registered dataset** (`lc.datasets.REGISTRY`): `lc.datasets.fetch(name, raw_dir)` downloads the pinned public
  source once, and `lc.datasets.load(name, raw_dir)` builds the Dataset.
  - `decision_index`, `oll_bbh`, `agentjudgebench` include their judges' recorded answers: ready to evaluate.
  - `politicause`, `beavertails`, `arena`, `healthbench` load the items only (`ds.n_judges == 0`). Rerun the paper's
    judges with `lc.collect(ds, lc.datasets.<name>.recipe({model: base_url, ...}), repeats=8)`, which needs one
    vLLM server per model (the recipe's docstring names them).
- **Your own data with recorded answers**: `lc.Dataset.from_records(items, judges, answers, cost, labels=...)`.
  `items` are dicts with `id`, and optionally `task`, `group` (items sharing a group never straddle folds) and the
  texts to embed (`task_description`, `item`). `answers` is (N, M) or (N, M, R) class indices, `lc.MISSING` where no
  call was recorded; `cost` has the same shape, in USD.
- **Your own questions, judges not yet run**: build items with `from_records(items, labels=...)`, then
  `lc.collect(ds, [lc.JudgeSpec(name, lc.Backend.vllm(url, model, price_in=..., price_out=...), lc.Prompt(...)), ...],
  repeats=k)`. Calls are cached in `lmcircuit_cache.sqlite`, so a rerun never pays twice.
- **No labels**: `lc.label(ds, prompt, graders)` labels the items with grader models.

`ds.save(dir)` and `lc.Dataset.load(dir)` store a Dataset (embed and collect once, then reload).

### 2. Evaluating

`lc.evaluate(ds, baselines=None, out=dir)` runs 5 grouped folds x 5 training seeds (`lc.EvalConfig`), so every item is
tested once. With `out`, every (fold, seed) is saved as it finishes, along with `report.json`.

- **Splitting the work**: run `lc.evaluate_fold(ds, fold, seed, out=dir)` per job (each is independent), then
  `report = lc.aggregate(ds, lc.load_folds(dir))` and `lc.save_report(report, path)`.
- **Baselines**: `baselines="all"` or a list of names from `lc.baselines.METHODS`. FrugalGPT and HybridLLM fine-tune
  a scorer per fold and need a GPU and hours on large datasets. Consistency Cascade and Adaptive-Consistency need
  repeated calls, or recorded verdict probabilities to sample from; otherwise they are recorded as skipped.
- **Reading a report**: `report["objectives"][objective]["circuit"]["area"]` holds the frontier areas, and `["curve"]`
  holds one row per budget (accuracy, balanced accuracy, Brier score, cost, calls per judge, 95% intervals).
  `report["baselines"][method]` has the same layout, and `report["references"]` holds each judge alone and majority
  voting.

Folds need at least 5 groups and enough labelled items per fold; a few hundred items is a sensible minimum.

### 3. Using one fitted circuit

```python
train, val, test = next(ds.folds(5))
circuit = lc.Circuit().fit(ds, train, val)          # objective: accuracy; circuit.objective("balanced_accuracy")
point = circuit.at_budget(0.05)                     # budget: expected cost per item / the priciest judge's mean call
circuit.save("circuit")                             # lc.Circuit.load("circuit") later

policy, answers = circuit.policy(ds.features[i], point), {}
while (d := policy.decide(answers)).ask is not None:
    answers[d.ask] = call_the_judge(d.ask, item)     # your code: the judge's class index
print(d.predict, d.belief)
```

Pairwise datasets (two candidates, e.g. `arena`) also need `features_swap=ds.features_swap[i]` in `circuit.policy`.

## Configuration

Defaults live in `lc.CircuitConfig` (pool size, price grid, device), `lc.BeliefConfig` (types, training),
`lc.EvalConfig` (folds, seeds, budgets) and `lc.baselines.BaselineConfig`; each field is commented with its reason.
Pass `lc.CircuitConfig(device="cpu")` to force CPU. Lower `EvalConfig(seeds=(42,))` for a quick run.

## Layout

```
lmcircuit/       the package (see README.md, "What is where")
examples/        lmcircuit.ipynb: a registered dataset or your own task, end to end
docs/method.md   the method, its defaults and the evaluation protocol
```
