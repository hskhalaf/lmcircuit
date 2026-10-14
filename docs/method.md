# lmcircuit: the method

`lmcircuit` decides, item by item, which language-model judges to ask and when to stop, so that a classification is
as accurate as possible for its cost. This document is the design of record: the data model, the pipeline, the method
and its defaults, the evaluation and the baselines.

## 1. Terms

- **Item**: one thing to classify (a question, a response under a criterion, a pair of responses, a trajectory).
- **Label**: one of K classes, 2 <= K <= 4. An item may carry several annotations; labels are counts (N, K), and a
  hard label is a count of one.
- **Judge**: anything that answers an item with one of the K classes at a cost: an LLM under a prompt, a reward model,
  a classifier, a model at a reasoning-effort level. Judges answer in the label space.
- **Call**: one judge's answer on one item, with its cost. A judge may have several recorded calls per item
  (repeats); evaluation treats each repeat as one world.
- **Prior features**: what is in hand about an item before any judge is paid: its task and the item as the judges
  see it, the judged object included (a response, a trajectory). Learned on the training labels, the prior is then a
  free judge; every method compared reads the same features.
- **Pool**: the judges a circuit may call: at most 8, and at most (K+1)^M <= 6561 states (binary 8 judges, 3
  classes 6, 4 classes 5), since the DP is exact over every state.
- **Circuit**: the policy. At each state (the answers seen so far) it either stops and predicts, or pays for one
  more judge.
- **Budget**: expected cost per item, rescaled by the most expensive judge's mean cost per call.

## 2. Data model

`Dataset` holds, for N items, M judges, R recorded calls per judge and K classes:

| field | shape | meaning |
|---|---|---|
| `ids`, `task`, `group` | (N,) | item ids; task name; fold unit (items of one source share a group and never straddle train and test) |
| `answers` | (N, M, R) int | each call's class, `MISSING` (-1) where no call was recorded |
| `cost` | (N, M, R) float | each call's cost in USD (recorded, or tokens x price) |
| `orders` | (N, M, R) int | pairwise tasks: the order each call presented the two candidates in (0 as given, 1 exchanged); answers stay in the given order. None: one order |
| `judges` | list of `Judge` | name, model, prices (USD per million tokens), total and active parameters, metadata |
| `labels` | (N, K) float | annotation counts (a hard label is a count of one; an unlabelled item is all zeros) |
| `classes` | K names | what each class index means |
| `prior_text` | part -> (N,) str | the texts `embed` reads, and the fields the judge prompts read |
| `features` | (N, d) | prior features (filled by `embed`, or set directly) |
| `features_swap` | (N, d) | pairwise tasks: the features with the two candidates exchanged |
| `plan_cost` | (N, M) float | USD the circuit plans with, known before a call; None: each judge's mean training cost |
| `probs` | (N, M, R, K) float | each call's class probabilities at its verdict token (NaN: not recorded); confidence-based baselines read them |

`Dataset.save(dir)` writes `dataset.npz` (arrays), `prior_text.json.gz` and `manifest.json` (judges, classes,
provenance in `meta`). `Dataset.check()` reports unlabelled items, tasks dominated by one class, items that are all
their own group, judges with missing calls, and invalid costs.

## 3. Pipeline

1. **Build** a `Dataset` from records (`Dataset.from_records`) or a registered loader (`lc.datasets.load(name, raw)`,
   after `lc.datasets.fetch(name, raw)`).
2. **Label** (`lc.label`), only when there is no gold: grader models (a vLLM server or the OpenAI API) answer every
   item, `samples` times each; labels are the counts. Warns when a grader shares a model family with a judge.
3. **Collect** (`lc.collect`), only when judges' answers are not recorded: each judge (`lc.JudgeSpec`: backend,
   prompt) answers every item `repeats` times; answers, USD cost and a planning cost (input tokens x price + the
   judge's 95th-percentile output length x price) are recorded, and, when the backend reports token log-probabilities,
   each class's probability at the verdict token. A prompt with `permute` shows candidates (A/B) in a per-call random
   order (recorded in `orders`) and maps answers back. Calls run concurrently and are cached in SQLite by a hash of
   (endpoint, model, messages, max_tokens, seed), so a rerun never pays twice.
4. **Embed** (`lc.embed`): a frozen Qwen3-Embedding-0.6B in its query format, up to 4096 tokens per text, in the
   layout the dataset declares: `item` (the task description and the item, one text), `criterion` (a criterion and
   its context embedded apart: [c, x, c * x]) or `pair` (two candidates embedded apart: [(a + b)/2, a - b, |a - b|,
   a * b, lengths, task], and the same with the candidates exchanged). Precomputed features may be set instead.
5. **Fit** (`lc.Circuit().fit(ds, train, val, seed=s)`): the pool, the belief model, and every operating point's
   validation cost and metric for an objective; `circuit.objective(name)` switches objective without refitting.
6. **Use**: `circuit.at_budget(b)` fixes the operating point; `circuit.decide(ds, items, points)` decides from
   recorded answers; `circuit.policy(features, point, orders=..., features_swap=...).decide(answers)` decides live;
   `circuit.save` / `load`.
7. **Evaluate** (`lc.evaluate(ds, baselines="all", out=dir)`, or `lc.evaluate_fold(ds, f, seed, out=dir)` per job,
   then `lc.aggregate(ds, lc.load_folds(dir))`): grouped cross-fitting over every training seed; references, the
   published baselines and calibration; `lc.save_report`; `lc.plot.frontiers` and `lc.plot.figures`.

## 4. Method and defaults

Every default below was settled on the paper's seven datasets. All live in `CircuitConfig` and `BeliefConfig`.

**Belief model** (`belief.py`): a belief network within latent item types. For features x and an
unobserved type t in 1..T:
- pi_t(x), the type prior; q_jt(x, o), judge j's answer distribution in type t under presentation order o; both
  two-layer heads on a linear projection h of x;
- eta_t(x, s), the label distribution in type t at the state s: each answer seen is a token
  g(e_judge + e_answer [+ e_order] + e_type); the tokens are summed (the order of the answers does not matter, an
  unasked judge adds nothing) and [h + e_type, the sum, the share of judges asked] goes through a two-layer trunk.

At a state, the type posterior w_t(s) is proportional to pi_t(x) times the q_jt of the answers seen, and
P(Y | x, s) = sum_t w_t(s) eta_t(x, s), P(X_j | x, s, o) = sum_t w_t(s) q_jt(x, o). An answer counts differently,
and an unasked judge's predicted answer moves with the answers seen, through the type; with one type, the judges'
predicted answers depend on the text alone.
On a pairwise task, both outputs average the log-probabilities of the item and of its candidate-exchanged view
mapped back, so the model is exactly equivariant to exchanging the candidates. The DP's tables over every state are
computed at once: the trunk's first layer is linear, so its input splits into a text part per (item, type) and a
token part per (judge, answer).
- training: per batch, the label's cross-entropy at one sampled state per item (a number of judges uniform on 0..M,
  which judges uniformly, one recorded call of each) plus the cross-entropy of every unasked judge's prediction
  against all its recorded answers on the item, at another sampled state (weight 1); AdamW (learning rate 2e-3,
  weight decay 1e-4, batch 256), cosine schedule over 300 epochs, early stopping on the label loss at fixed
  validation states (patience 30);
- types: T in {1, 2, 4, 8}, the one with the lowest validation loss (label plus judges) at those states. Four types
  raised the frontier area over one on OLL BBH (70.4 to 74.1), Decision Index (77.3 to 78.0) and PolitiCAUSE (78.3
  to 79.1), in every fold of each;
- calibration: one temperature for the label and one for the judges' answers, each minimising the validation
  negative log-likelihood at 10 sampled states per item;
- no refit on train + validation: the model that chooses the operating point on validation is the model tested.

**Pool**: when a dataset has more than 8 judges, the frontier-coverage pool: the 8 judges minimising the mean, over
tasks, of the gap between the pool's per-task cost-accuracy frontier and the full one (model-free, train + validation;
the largest gap breaks ties). The mean rather than the largest: as good on OLL BBH and Decision Index, and the pool is
more stable across folds.

**DP**: exact backward induction over (K+1)^M states for all lambdas at once, on GPU. Stop loss: the smallest
weighted risk over the K predictions, with class weights w from the objective. Costs per item. Lambda grid: 0, 201
geometric values in [1e-8, 100] (20 per decade), and infinity (never ask). One call removes at most 1 - 1/K of the
error, so a judge costing c is never asked above lambda = (1 - 1/K) / c: 100 reaches judges at 0.5% of the priciest
judge's cost; below 1e-8 the price of a call is negligible against any change in the error.

**Objectives** (`objectives.py`). Every objective linear in the confusion matrix is the DP with class weights.
Accuracy: w = 1. Balanced accuracy: w_y = 1 / (K pi_y), exact (two classes: predict 1 when P(1) > pi_1). The
guarantees hold for the belief model's probabilities.

**Operating point**: at budget b, the lambda with the best validation value of the objective among those whose
validation cost is within b (the cheapest on ties): the constrained problem, estimated on validation. Every method
compared uses this rule.

**Costs**: recorded USD where available; else tokens x price; open models without a price use the compute proxy
0.01 x active parameters (billions) per million tokens (mixtures of experts: active parameters).

## 5. Evaluation

**Cross-fitting**: 5 folds by group (part f test; of the other groups a random 20% validation and the rest
training), the same in every seed, so every item is tested exactly once. Each fold is fitted with every training seed
(`EvalConfig.seeds`, default 42 to 46); the folds' test rows are pooled per seed, and curves and areas are the mean
over the seeds. Rows: each test item in `worlds` replays (min(5, recorded calls)), world r being every judge's r-th call.

**Reported**: at every budget and for each objective, accuracy, balanced accuracy, the Brier score of the belief
where the circuit stops, the mean cost and the calls per item to each pool judge (and each seed's values); 95%
intervals of every curve and frontier area from 1000 resamples of the test groups combined with resampled seeds
(each draw averages the seeds drawn, every seed resampled on the same groups); references: each judge alone,
majority voting of the pool (with its interval), always the majority class. Calibration, on the test rows: the belief
before any judge and after every pool judge, the belief where the circuit stops at budgets 0.01, 0.03, 0.1, 0.3 and
1, and majority voting's vote shares. Frontier area: a metric averaged over budget 0 and 50 log-spaced budgets in
[0.003, 1].

**Saved** (`out`): per (fold, seed), the fitted circuit (`circuit/`: belief model, temperatures, validation curve),
and per objective the operating points chosen at every budget with, per test row, the prediction, the cost, the
judges asked and the belief where the circuit stopped (`decisions.npz`), and everything else (`result.json`); the
baselines' folds (`baselines/fold{f}/`); `report.json`.

**Times** (`record["seconds"]`, wall clock with the GPU's queued work finished before each reading): the pool, the
belief model (each number of types tried, and calibration), the operating points (the validation pass of every
objective at every price) and the test pass; the device and how many fits shared it are recorded with them. With
`EvalConfig.n_jobs` = 1 (the default) every time is the method's alone.

## 6. Parallelism

- (fold, seed) fits: independent processes (`EvalConfig.n_jobs`) or one job each (`lc.evaluate_fold`, then
  `lc.aggregate`).
- DP: all lambdas and items batched on the GPU, chunked by a memory budget.
- Judge and grader calls: asyncio with bounded concurrency per backend; vLLM servers batch internally.

## 7. Published baselines

`lc.baselines` implements eight published methods with the settings and substitutions of our experiments (each
module's docstring): RouteLLM, Hybrid LLM, Trust or Escalate (cascaded
selective evaluation), FrugalGPT, cascade routing, CARROT, the answer-consistency cascade and Adaptive-Consistency.
They run on exactly the circuit's folds, pools and test rows
(`lc.evaluate(ds, baselines="all")`, or `lc.baselines.evaluate(ds, "all", report=report)` on an existing report), with
one seed, are fitted on each fold's training items (validation items only where the method uses them), and are
reported like the circuit: at each budget and for each metric, the operating point with the best validation value of
that metric within the budget (FrugalGPT and cascade routing: the point optimised for the largest budget within it),
pooled over folds; a fold with no point within the budget answers with the training majority class at no cost (not
drawn). Times per method and fold: "fit" (training and setting up on train and validation, scorers' inference
included) and "decide" (the decisions on the validation and test rows at every operating point; FrugalGPT's and
cascade routing's per-budget work is timed in its worker processes and summed). FrugalGPT's and cascade routing's
libraries are vendored (Apache-2.0, `baselines/_vendor/`, provenance and pinned commits there). The fine-tuned
scorers (FrugalGPT's DistilBERT, Hybrid LLM's DeBERTa-v3-large) need the `baselines` extra and a GPU. The
repeated-call methods need recorded repeats (consistency cascade 3, Adaptive-Consistency 2); with fewer, they draw the
further calls from the recorded call's class probabilities (settings `simulate`, the recorded call first), and are
skipped when no probabilities were recorded.
