"""Cross-fitted evaluation: every item tested once; for each objective, the circuit at every budget, scored on every
metric; references, the published baselines and the belief model's calibration.

    report = lc.evaluate(ds, baselines="all", out="runs/oll_bbh")   # every (fold, seed), the baselines, then aggregate
    lc.evaluate_fold(ds, f, seed, out="runs/oll_bbh")               # or one (fold, seed) per job, ...
    report = lc.aggregate(ds, lc.load_folds("runs/oll_bbh"))        # ... then aggregate what they saved

Per fold (docs/method.md): grouped k-fold parts (part f test; of the other groups a random share validation, the rest
train), the same in every seed; per training seed, one circuit fitted on train and validation; for each objective, at
each budget, the operating point with the best validation value of that objective within the budget
(objectives.pick); its decisions on the test items x worlds: prediction, charged cost, the judges asked and the belief
where the circuit stops, per test row. With `out`, each (fold, seed) is saved in out/fold{f}_seed{s}/: the fitted
circuit (circuit/), the decisions (decisions.npz) and the rest (result.json).
Aggregation, per seed, pools the folds' test rows at each budget; curves are the mean over the seeds. 95% intervals of
every curve and frontier area resample the test groups and the training seeds (metrics.over_seeds). Frontier area: a
metric averaged over budget 0 and log-spaced budgets. Calibration: the belief before any judge and after every pool
judge, the belief where each circuit stops at the budgets CALIBRATION (every seed's rows pooled), and majority voting's
vote shares. Times (record["seconds"], wall clock with the GPU's queued work finished): pool, belief (every number of
types tried, and calibration), operating points (the validation pass of every objective at every price) and test
(the test pass); the saved circuit's validation curve (its default target) comes from the same validation pass.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .circuit import Circuit, Clock
from .config import CircuitConfig, EvalConfig, memory_cells, resolve_device
from .data import Dataset
from .metrics import METRICS, brier_rows, draws, interval, over_seeds, scores
from .objectives import OBJECTIVES, pick
from .objectives import weights as objective_weights

CALIBRATION = (0.01, 0.03, 0.1, 0.3, 1.0)        # budgets at which the stopping belief's calibration is reported


def budgets(ecfg: EvalConfig) -> tuple[np.ndarray, np.ndarray]:
    """The budgets every curve is drawn at, and which of them the frontier area averages over."""
    area = np.concatenate([[0.0], np.geomspace(ecfg.area_lo, ecfg.area_hi, ecfg.area_count)])
    grid = np.unique(np.concatenate([area, np.geomspace(1e-5, 3.0, 160), CALIBRATION]))
    return grid, np.isin(grid, area)


def vote_shares(ds: Dataset, items, worlds: int, pool: list[str]) -> np.ndarray:
    """Per row of items x worlds, the share of the pool's answers in each class: majority voting's belief."""
    cols = [ds.judge_names.index(n) for n in pool]
    ans = ds.votes()[np.asarray(items)][:, cols, :worlds].transpose(0, 2, 1).reshape(-1, len(cols))
    votes = np.stack([(ans == k).sum(axis=1) for k in range(ds.n_classes)], axis=1).astype(np.float64)
    return votes / np.maximum(votes.sum(axis=1, keepdims=True), 1)


def references(ds: Dataset, train_idx, test_idx, worlds: int, pool: list[str]) -> dict:
    """On the test items x worlds: each judge alone, majority vote of the pool, the training majority class."""
    test_idx = np.asarray(test_idx)
    flat = lambda a: a[test_idx, :, :worlds].transpose(0, 2, 1).reshape(len(test_idx) * worlds, -1)  # noqa: E731
    lab = np.repeat(ds.labels[test_idx], worlds, axis=0)
    cost, ans = flat(ds.cost) / ds.scale(), flat(ds.votes())
    as_floats = lambda s: {k: float(v) for k, v in s.items()}  # noqa: E731
    out = {"judges": {}}
    for j, name in enumerate(ds.judge_names):
        ok = ans[:, j] >= 0
        out["judges"][name] = {"cost": float(cost[ok, j].mean()), **as_floats(scores(ans[ok, j], lab[ok]))}
    cols = [ds.judge_names.index(n) for n in pool]
    votes = vote_shares(ds, test_idx, worlds, pool)
    out["majority_vote"] = {"cost": float(cost[:, cols].sum(axis=1).mean()), **as_floats(scores(votes.argmax(axis=1), lab))}
    majority = int(ds.labels[np.asarray(train_idx)].sum(axis=0).argmax())
    out["majority_class"] = {"cost": 0.0, **as_floats(scores(np.full(len(lab), majority), lab))}
    return out


def device_name(device: str) -> str:
    import torch
    return torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "cpu"


def evaluate_fold(ds: Dataset, fold: int, seed: int = 0, config: CircuitConfig | None = None,
                  eval_config: EvalConfig | None = None, objectives=None, out=None, concurrent: int = 1) -> dict:
    """One (fold, seed): the circuit fitted on the fold's training and validation parts with this training seed, its
    decisions on the test part. out: a directory to save it in (out/fold{f}_seed{s}/). concurrent: how many fits shared
    the device meanwhile (recorded with the times)."""
    ecfg = eval_config or EvalConfig()
    objectives = tuple(objectives or OBJECTIVES)
    train, val, test = list(ds.folds(ecfg.folds, ecfg.shuffle_seed, ecfg.val_share))[fold]
    worlds = int(ecfg.worlds or min(5, ds.answers.shape[2]))
    circuit = Circuit(config).fit(ds, train, val, worlds, seed, curve=False)     # the curves come from the pass below
    clock = Clock(circuit.device)
    done = ds.with_judges(circuit.pool).complete(worlds) & ds.labelled          # every pool judge's call recorded
    dropped, test, val = int((~done[test]).sum()), test[done[test]], val[done[val]]
    grid, _ = budgets(ecfg)
    val_labels = np.repeat(ds.labels[val], worlds, axis=0)
    weights = {name: objective_weights(name, circuit.train_labels) for name in objectives}
    offset = dict(zip(objectives, np.cumsum([0] + [len(w) for w in weights.values()])[:-1]))
    stacked = np.concatenate(list(weights.values()))
    with clock("operating_points"):
        on_val = circuit.run(ds, val, stacked, asked=False)                    # every objective in one pass
        chosen, keep = {}, [set() for _ in stacked]
        for name in objectives:
            g = slice(offset[name], offset[name] + len(weights[name]))
            cost = np.stack([c.mean(axis=1) for c in on_val["cost"][g]])                      # (G, L)
            metric = scores(np.stack(on_val["pred"][g]), val_labels)[name]
            chosen[name] = [pick(cost, metric, b) for b in grid]
            for k, l in set(chosen[name]):
                keep[offset[name] + k].add(l)
            if name == circuit.cfg.target:
                circuit.set_curve(name, weights[name], cost, metric)
        keep = [sorted(k) for k in keep]
    with clock("test"):
        on_test = circuit.run(ds, test, stacked, keep, beliefs=True)
    record = {**circuit.record, "device": device_name(circuit.device), "concurrent": int(concurrent)}
    record["seconds"] = {**record["seconds"], **clock.seconds}
    result = {"fold": fold, "seed": int(seed), "worlds": worlds, "pool": circuit.pool, "record": record,
              "n_test_items": int(len(test)), "n_test_dropped": dropped, "test_items": np.repeat(test, worlds),
              "budgets": grid, "references": references(ds, train, test, worlds, circuit.pool),
              "calibration": {"prior": on_test["prior"].astype(np.float32), "full": on_test["full"].astype(np.float32),
                              "majority_vote": vote_shares(ds, test, worlds, circuit.pool).astype(np.float32)},
              "objectives": {}}
    for name in objectives:
        points = sorted(set(chosen[name]))
        at = lambda k, g, l: on_test[k][offset[name] + g][keep[offset[name] + g].index(l)]  # noqa: E731
        result["objectives"][name] = {
            "at": np.asarray([points.index(p) for p in chosen[name]]),             # budget -> point
            "lambda": np.asarray([circuit.lambdas[l] for _, l in points]),
            "val_cost": np.asarray([np.mean(on_val["cost"][offset[name] + g][l]) for g, l in points]),
            "pred": np.stack([at("pred", g, l) for g, l in points]),               # (points, rows)
            "cost": np.stack([at("cost", g, l) for g, l in points]),               # (points, rows), rescaled
            "asked": np.stack([at("asked", g, l) for g, l in points]),             # (points, rows, pool)
            "belief": np.stack([at("belief", g, l) for g, l in points]).astype(np.float16)}   # (points, rows, K)
    if out is not None:
        where = Path(out) / f"fold{fold}_seed{seed}"
        circuit.save(where / "circuit")
        save_tree(result, where)
    return result


# ---------------------------------------------------------------------------------------------- storage
def save_tree(result: dict, where) -> Path:
    """A nested dict as decisions.npz (its arrays, keys joined by '/') and result.json (everything else)."""
    where = Path(where)
    where.mkdir(parents=True, exist_ok=True)
    arrays, rest = {}, {}

    def split(node, path, into):
        for k, v in node.items():
            key = f"{path}/{k}" if path else str(k)
            if isinstance(v, dict):
                split(v, key, into.setdefault(k, {}))
            elif isinstance(v, np.ndarray) and v.ndim > 0:
                arrays[key] = v
            else:
                into[k] = v
    split(result, "", rest)
    np.savez_compressed(where / "decisions.npz", **arrays)
    (where / "result.json").write_text(json.dumps(rest, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return where


def load_tree(where) -> dict:
    where = Path(where)
    out = json.loads((where / "result.json").read_text())
    with np.load(where / "decisions.npz", allow_pickle=False) as a:
        for key in a.files:
            node, *path, leaf = [out] + key.split("/")
            for k in path:
                node = node.setdefault(k, {})
            node[leaf] = a[key]
    return out


def load_folds(out) -> list[dict]:
    """Every (fold, seed) saved under `out` (evaluate_fold's `out`)."""
    return [load_tree(p.parent) for p in sorted(Path(out).glob("fold*_seed*/result.json"))]


# ---------------------------------------------------------------------------------------------- aggregation
def calibration(prob: np.ndarray, labels: np.ndarray, bins: int = 10) -> dict:
    """Reliability of P(class 1) (two classes) or of the most probable class: per equal-width bin, the mean predicted
    probability, the observed share (of annotations), the rows; the expected calibration error; the log loss
    (cross-entropy against each row's annotation shares) and the Brier score."""
    prob = np.asarray(prob, dtype=np.float64)
    share = labels / np.maximum(labels.sum(axis=1, keepdims=True), 1e-12)
    log_loss = float(-(share * np.log(np.clip(prob, 1e-12, 1.0))).sum(axis=1).mean())
    if prob.shape[1] == 2:
        p, hit = prob[:, 1], share[:, 1]
    else:
        top = prob.argmax(axis=1)
        p, hit = prob[np.arange(len(prob)), top], share[np.arange(len(prob)), top]
    which = np.minimum((p * bins).astype(int), bins - 1)
    rows = np.bincount(which, minlength=bins)
    mean = lambda v: np.bincount(which, weights=v, minlength=bins) / np.maximum(rows, 1)  # noqa: E731
    predicted, observed = mean(p), mean(hit)
    return {"predicted": predicted.tolist(), "observed": observed.tolist(), "rows": rows.tolist(),
            "ece": float((rows * np.abs(predicted - observed)).sum() / max(rows.sum(), 1)), "log_loss": log_loss,
            "brier": float(brier_rows(prob, labels).mean())}


def _stage_seconds(records: list[dict]) -> dict:
    """Each stage's seconds: mean and total over the (fold, seed) fits."""
    flat = []
    for r in records:
        s = dict(r["seconds"])
        stages = s.pop("belief_stages", {})
        flat.append({**{k: float(v) for k, v in s.items()}, **{f"belief_{k}": float(v) for k, v in stages.items()}})
    keys = sorted({k for f in flat for k in f})
    return {k: {"mean": float(np.mean([f.get(k, 0.0) for f in flat])), "total": float(sum(f.get(k, 0.0) for f in flat))}
            for k in keys}


def aggregate(ds: Dataset, results: list[dict], eval_config: EvalConfig | None = None) -> dict:
    """Curves (every metric and the Brier score at every budget: the mean over seeds of the folds' pooled test rows),
    frontier areas, intervals, references, calibration and times."""
    ecfg = eval_config or EvalConfig()
    grid, area = budgets(ecfg)
    seeds = sorted({int(r["seed"]) for r in results})
    folds = sorted({int(r["fold"]) for r in results})
    by = {(int(r["fold"]), int(r["seed"])): r for r in results}
    if len(by) != len(folds) * len(seeds):
        raise ValueError(f"every fold needs every seed: have {sorted(by)}")
    first = [by[f, seeds[0]] for f in folds]
    items = np.concatenate([np.asarray(r["test_items"]) for r in first])
    labels, groups = ds.labels[items], ds.group[items]
    w = np.asarray([len(r["test_items"]) for r in first], dtype=float) / len(items)
    mean = lambda get: float(sum(wi * get(r["references"]) for wi, r in zip(w, first)))  # noqa: E731
    refs0 = first[0]["references"]
    shares = np.concatenate([np.asarray(r["calibration"]["majority_vote"], dtype=np.float64) for r in first])
    mv = draws(shares.argmax(axis=1)[None], labels, groups, shares[None])
    report = {"n_items": ds.n_items, "n_tested": int(sum(r["n_test_items"] for r in first)),
              "n_dropped": int(sum(r["n_test_dropped"] for r in first)), "folds": len(folds), "seeds": seeds,
              "pools": [r["pool"] for r in first], "budgets": grid.tolist(),
              "records": [{"fold": f, "seed": s, **by[f, s]["record"]} for f in folds for s in seeds],
              "seconds": _stage_seconds([by[f, s]["record"] for f in folds for s in seeds]),
              "references": {"judges": {j: {k: mean(lambda r, j=j, k=k: r["judges"][j][k]) for k in refs0["judges"][j]}
                                        for j in refs0["judges"]},
                             **{n: {k: mean(lambda r, n=n, k=k: r[n][k]) for k in refs0[n]}
                                for n in ("majority_vote", "majority_class")}},
              "calibration": {s_: calibration(np.concatenate([np.asarray(by[f, s]["calibration"][s_]) for s in seeds
                                                              for f in folds]), np.tile(labels, (len(seeds), 1)))
                              for s_ in ("prior", "full", "majority_vote")},
              "objectives": {}}
    report["references"]["majority_vote"]["interval"] = {m: np.percentile(v[:, 0], [2.5, 97.5]).tolist()
                                                         for m, v in mv.items()}
    stops = [int(np.flatnonzero(np.isclose(grid, b))[0]) for b in CALIBRATION]
    for name in first[0]["objectives"]:
        values, per_seed, calls, stopped = [], [], [], {b: [] for b in CALIBRATION}
        for s in seeds:
            blocks = [by[f, s]["objectives"][name] for f in folds]
            take = lambda key, b: np.concatenate([np.asarray(r[key])[np.asarray(r["at"])[b]] for r in blocks])  # noqa: E731
            pred = np.stack([take("pred", b) for b in range(len(grid))])
            prob = np.stack([take("belief", b).astype(np.float32) for b in range(len(grid))])
            cost = np.asarray([take("cost", b).astype(np.float64).mean() for b in range(len(grid))])
            v = {**{m: np.asarray(x) for m, x in scores(pred, labels).items()}, "brier": brier_rows(prob, labels).mean(axis=1),
                 "cost": cost}
            values.append(v)
            per_seed.append(draws(pred, labels, groups, prob))
            per_point = [np.asarray(r["asked"]).mean(axis=1) for r in blocks]                # (points, pool) per fold
            seed_calls = []
            for b in range(len(grid)):
                c = {}
                for f, r, q in zip(folds, blocks, per_point):
                    share = len(by[f, s]["test_items"]) / len(items)
                    for judge, x in zip(by[f, s]["pool"], q[np.asarray(r["at"])[b]]):
                        c[judge] = c.get(judge, 0.0) + float(x) * share
                seed_calls.append(c)
            calls.append(seed_calls)
            for b, g in zip(CALIBRATION, stops):
                stopped[b].append(prob[g])
        spread = interval(over_seeds(per_seed), area)
        rows = []
        for b in range(len(grid)):
            judges = sorted({j for c in calls for j in c[b]})
            rows.append({"budget": float(grid[b]), "cost": float(np.mean([v["cost"][b] for v in values])),
                         **{m: float(np.mean([v[m][b] for v in values])) for m in METRICS + ("brier",)},
                         "by_seed": {m: [float(v[m][b]) for v in values] for m in METRICS + ("brier", "cost")},
                         "calls_per_item": {j: float(np.mean([c[b].get(j, 0.0) for c in calls])) for j in judges},
                         "interval": {m: [float(sp["lo"][b]), float(sp["hi"][b])] for m, sp in spread.items()}})
        report["objectives"][name] = {"circuit": {
            "curve": rows,
            "area": {m: float(np.mean([r[m] for r, a in zip(rows, area) if a])) for m in spread},
            "area_by_seed": {m: [float(np.mean(v[m][area])) for v in values] for m in spread},
            "area_interval": {m: np.percentile(sp["area"], [2.5, 97.5]).tolist() for m, sp in spread.items()},
            "area_draws": {m: sp["area"].round(6).tolist() for m, sp in spread.items()},
            "calibration": {f"{b:g}": calibration(np.concatenate(stopped[b]), np.tile(labels, (len(seeds), 1)))
                            for b in CALIBRATION}}}
    return report


# ---------------------------------------------------------------------------------------------- running
def _threads(n: int) -> None:
    import torch
    torch.set_num_threads(n)


def run_folds(fn, args: list[tuple], n_jobs: int) -> list:
    """fn(*a) for every a: in up to n_jobs spawned processes, which share the GPU and split the CPU cores, or in this
    process when n_jobs is 1."""
    n = min(int(n_jobs), len(args))
    if n <= 1:
        return [fn(*a) for a in args]
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or n)
    with cf.ProcessPoolExecutor(n, mp_context=multiprocessing.get_context("spawn"), initializer=_threads,
                                initargs=(max(1, cores // n),)) as ex:
        return list(ex.map(fn, *zip(*args)))


def evaluate(ds: Dataset, config: CircuitConfig | None = None, eval_config: EvalConfig | None = None,
             objectives=None, baselines=None, baseline_config=None, out=None) -> dict:
    """Every (fold, seed) (eval_config.n_jobs at a time), then the aggregate report. baselines: published methods
    ("all" or names of lmcircuit.baselines.METHODS) on the same folds, pools and test rows, one seed, under
    report["baselines"]. out: a directory for every (fold, seed)'s results, the baselines' folds and report.json."""
    ecfg = eval_config or EvalConfig()
    objectives = tuple(objectives or OBJECTIVES)
    config = config or CircuitConfig()
    jobs = [(f, s) for f in range(ecfg.folds) for s in ecfg.seeds]
    n_jobs = min(ecfg.n_jobs, len(jobs))
    budget = memory_cells(config.max_cells, resolve_device(config.device))                # one GPU, shared by the fits
    shared = replace(config, max_cells=budget / max(1, n_jobs))
    results = run_folds(evaluate_fold, [(ds, f, s, shared, ecfg, objectives, out, n_jobs) for f, s in jobs], n_jobs)
    report = aggregate(ds, results, ecfg)
    report["config"] = {"circuit": asdict(config), "eval": asdict(ecfg)}
    if baselines:
        from .baselines import evaluate as evaluate_baselines
        evaluate_baselines(ds, baselines, baseline_config, ecfg, report=report, circuit_config=config,
                           out=None if out is None else Path(out) / "baselines")
    if out is not None:
        save_report(report, Path(out) / "report.json")
    return report


def save_report(report: dict, path) -> Path:
    """The report as JSON (arrays as lists)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return path
