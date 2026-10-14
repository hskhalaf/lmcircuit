"""The published baselines on the circuit's folds, pools and test rows.

    folds = [lc.baselines.evaluate_fold(ds, f, "all") for f in range(5)]; lc.baselines.aggregate(ds, folds)
    report = lc.baselines.evaluate(ds, "all", report=report)       # added to an lc.evaluate report (its pools)
    report = lc.evaluate(ds, baselines="all")                       # the same, in one call

Per fold: grouped parts as in lmcircuit.evaluation (part f test; of the other groups a random share validation, the
rest training); the circuit's pool (frontier-coverage on the labelled training and validation items, or the pools of
an existing report); the circuit's rows (test items x worlds with every pool judge's call recorded). Each method is
fitted on the training items (and the validation items where it uses them: cascaded_selective calibrates on them,
hybridllm calibrates its thresholds and keeps the router checkpoint with them), and every operating point is replayed
on the validation and the test rows: prediction and cost per row, calls per judge, the method's run time (fine-tuning
included).
Aggregation, at each budget (in the circuit's unit), per metric and per fold: the operating point by the method's rule
(config.RULES): "cost", the one with the best validation value of that metric within the budget (the circuit's rule),
or "budget", the one optimised for the largest budget within it (FrugalGPT, cascade routing); pooled over the folds'
test rows. A fold where no point is within the budget answers every row with the training majority class at no cost
(the reference every method has for free; such budgets are marked, fallback_folds, and not drawn), so the frontier
area (the metric at budget 0 and log-spaced budgets, as for the circuit) is defined for every method. No random
mixtures of operating points (the circuit gets none either). Times per method and fold (seconds): "fit" and "decide"
(FoldData.stage), and "total".
"""
from __future__ import annotations

import importlib
import time
from pathlib import Path

import numpy as np

from ..config import CircuitConfig, EvalConfig, resolve_device
from ..data import Dataset
from ..evaluation import budgets, device_name, run_folds, save_tree
from ..metrics import METRICS, draws, interval, scores
from ..objectives import pick
from ..pool import coverage_pool
from .common import FoldData, NotApplicable
from .config import METHODS, RULES, BaselineConfig


def _methods(methods) -> list[str]:
    """The method names ("all" or None: every method)."""
    if methods in ("all", None):
        return list(METHODS)
    names = [methods] if isinstance(methods, str) else list(methods)
    unknown = sorted(set(names) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown baselines {unknown}; known: {list(METHODS)}")
    return names


def evaluate_fold(ds: Dataset, fold: int, methods="all", config: BaselineConfig | None = None,
                  eval_config: EvalConfig | None = None, pool: list[str] | None = None,
                  circuit_config: CircuitConfig | None = None, where=None) -> dict:
    """One fold of every method. pool: the circuit's pool for this fold (default: chosen as Circuit.fit does, with
    circuit_config's pool size). where: a directory to save it in (where/fold{f}/)."""
    ecfg, cfg = eval_config or EvalConfig(), config or BaselineConfig()
    names = _methods(methods)
    train, val, test = list(ds.folds(ecfg.folds, ecfg.shuffle_seed, ecfg.val_share))[fold]
    worlds = int(ecfg.worlds or min(5, ds.answers.shape[2]))
    if pool is None:
        tr, va = train[ds.labelled[train]], val[ds.labelled[val]]
        cols, _ = coverage_pool(ds, np.concatenate([tr, va]), (circuit_config or CircuitConfig()).pool_for(ds.n_classes),
                                ds.plan_cost_for(tr))
        pool = [ds.judge_names[j] for j in cols]
    fd = FoldData.build(ds, train, val, test, pool, worlds, cfg.seed)
    if not len(fd.val) or not len(fd.test):
        raise ValueError(f"fold {fold}: no validation or test item has a recorded call of every pool judge")
    val_lab = np.repeat(ds.labels[fd.val], worlds, axis=0)
    out = {"fold": fold, "worlds": worlds, "pool": list(pool), "judges": fd.judges, "test_items": np.repeat(fd.test, worlds),
           "n_test_items": int(len(fd.test)), "n_test_rows": int(len(fd.test) * worlds),
           "majority_class": int(ds.labels[fd.train].sum(axis=0).argmax()), "methods": {}}
    for m in names:
        fd.seconds = {}
        started = time.perf_counter()
        try:
            res = importlib.import_module(f".{m}", __package__).run(fd, cfg.of(m), cfg)
        except NotApplicable as e:                       # e.g. a repeated-call method without recorded repeats
            out["methods"][m] = {"skipped": str(e)}
            continue
        if not res.points:
            raise ValueError(f"{m} produced no operating points")
        P = res.points
        val_pred = np.stack([p.val.pred for p in P])
        out["methods"][m] = {
            "rule": RULES[m], "seconds": {**fd.seconds, "total": time.perf_counter() - started},
            "info": res.info, "settings": [p.settings for p in P],
            "budget": np.asarray([np.nan if p.budget is None else p.budget for p in P]),
            "val_cost": np.asarray([p.val.cost.mean() for p in P]),
            "val_scores": {k: np.asarray(v) for k, v in scores(val_pred, val_lab).items()},
            "test_pred": np.stack([p.test.pred for p in P]).astype(np.int8),
            "test_cost": np.stack([p.test.cost for p in P]).astype(np.float32),
            "test_calls": np.stack([p.test.calls for p in P])}
    out["device"], out["seed"] = device_name(resolve_device(cfg.device)), cfg.seed
    if where is not None:
        save_tree(out, Path(where) / f"fold{fold}")
    return out


def _choose(r: dict, budget: float, metric: str) -> int | None:
    """The operating point at `budget` for `metric` by the method's rule (config.RULES); None when no point is within
    it."""
    key = np.asarray(r["budget"] if r["rule"] == "budget" else r["val_cost"], dtype=np.float64)
    if not (key <= budget + 1e-12).any():
        return None
    if r["rule"] == "budget":                       # the point optimised for the largest budget within this one
        ok = np.flatnonzero(key <= budget + 1e-12)
        return int(ok[np.argmax(key[ok])])
    return pick(np.where(np.isfinite(key), key, np.inf), np.asarray(r["val_scores"][metric]), budget)[0]


def aggregate(ds: Dataset, folds: list[dict], eval_config: EvalConfig | None = None, report: dict | None = None) -> dict:
    """Pooled curves (every metric at every budget, the operating points chosen per metric), frontier areas, intervals
    and times of every method, in report["baselines"] (a new report of these folds when none is given). A method
    missing from a fold (its job did not finish) is reported as skipped. With a report holding the circuit's frontier
    area draws, each method's paired difference from the circuit: versus_circuit, the circuit's area minus the
    method's, with its 95% interval over the same resamples of the test groups."""
    ecfg = eval_config or EvalConfig()
    grid, area = budgets(ecfg)
    tested = int(sum(f["n_test_items"] for f in folds))
    if report is not None and report.get("n_tested") not in (None, tested):
        raise ValueError(f"the baselines' test items ({tested}) differ from the report's ({report['n_tested']})")
    out = report if report is not None else {"n_items": ds.n_items, "n_tested": tested, "folds": len(folds),
                                             "pools": [f["pool"] for f in folds], "budgets": grid.tolist()}
    block = out.setdefault("baselines", {})
    items = np.concatenate([np.asarray(f["test_items"]) for f in folds])
    labels = ds.labels[items]
    for m in [m for m in METHODS if any(m in f["methods"] for f in folds)]:
        if missing := [f["fold"] for f in folds if m not in f["methods"]]:
            block[m] = {"skipped": f"no result for folds {missing}"}
            continue
        if skipped := [f["methods"][m]["skipped"] for f in folds if "skipped" in f["methods"][m]]:
            block[m] = {"skipped": skipped[0]}
            continue

        def pooled(b, metric, m=m):
            preds, costs, calls, fallback = [], [], {}, 0
            for f in folds:
                r, rows = f["methods"][m], len(f["test_items"])
                if (i := _choose(r, b, metric)) is None:      # nothing within the budget: the training majority class
                    preds.append(np.full(rows, f["majority_class"])); costs.append(np.zeros(rows))
                    fallback += 1
                    continue
                preds.append(np.asarray(r["test_pred"])[i]); costs.append(np.asarray(r["test_cost"])[i])
                for name, q in zip(f["judges"], np.asarray(r["test_calls"])[i]):
                    calls[name] = calls.get(name, 0.0) + float(q) * rows / len(items)
            pred = np.concatenate(preds)
            return pred, {"budget": float(b), "cost": float(np.concatenate(costs).mean()),
                          **{k: float(v) for k, v in scores(pred, labels).items()},
                          "calls_per_item": calls, "fallback_folds": fallback}
        seconds = [f["methods"][m]["seconds"] for f in folds]
        block[m] = {"rule": RULES[m], "objectives": {},
                    "seconds": {k: {"mean": float(np.mean([s_.get(k, 0.0) for s_ in seconds])),
                                    "total": float(sum(s_.get(k, 0.0) for s_ in seconds)),
                                    "by_fold": [float(s_.get(k, 0.0)) for s_ in seconds]}
                                for k in sorted({k for s_ in seconds for k in s_})},
                    "points": [int(len(f["methods"][m]["val_cost"])) for f in folds],
                    "info": [f["methods"][m]["info"] for f in folds]}
        for metric in METRICS:
            preds, curve = zip(*[pooled(b, metric) for b in grid])
            spread = interval(draws(np.stack(preds), labels, ds.group[items]), area)
            for b, row in enumerate(curve):
                row["interval"] = {k: [float(s_["lo"][b]), float(s_["hi"][b])] for k, s_ in spread.items()}
            block[m]["objectives"][metric] = {
                "curve": list(curve),
                "area": {k: float(np.mean([r[k] for r, a in zip(curve, area) if a])) for k in METRICS},
                "area_interval": {k: np.percentile(s_["area"], [2.5, 97.5]).tolist() for k, s_ in spread.items()},
                "area_draws": {k: s_["area"].round(6).tolist() for k, s_ in spread.items()}}
            ours = out.get("objectives", {}).get(metric, {}).get("circuit", {})
            if "area_draws" in ours:
                diff = np.asarray(ours["area_draws"][metric]) - spread[metric]["area"]
                block[m]["objectives"][metric]["versus_circuit"] = {
                    "area_difference": ours["area"][metric] - block[m]["objectives"][metric]["area"][metric],
                    "interval": np.percentile(diff, [2.5, 97.5]).tolist(), "share_circuit_better": float((diff > 0).mean())}
    return out


def evaluate(ds: Dataset, methods="all", config: BaselineConfig | None = None, eval_config: EvalConfig | None = None,
             report: dict | None = None, circuit_config: CircuitConfig | None = None, out=None) -> dict:
    """Every fold (eval_config.n_jobs at a time), then aggregate. With `report` (lc.evaluate's, possibly loaded from
    JSON), its folds' pools and evaluation settings are used and the baselines are added to it. out: a directory for
    each fold's results (out/fold{f}/)."""
    if report is not None:
        ecfg = eval_config or EvalConfig(**{k: tuple(v) if isinstance(v, list) else v
                                            for k, v in report.get("config", {}).get("eval", {}).items()})
        pools = report["pools"]
    else:
        ecfg = eval_config or EvalConfig()
        pools = [None] * ecfg.folds
    args = [(ds, f, methods, config, ecfg, pools[f], circuit_config, out) for f in range(ecfg.folds)]
    return aggregate(ds, run_folds(evaluate_fold, args, ecfg.n_jobs), ecfg, report)


def load_folds(*outs) -> list[dict]:
    """Every fold saved under the directories `outs` (evaluate_fold's `where`), the methods of each fold merged across
    them (e.g. the expensive methods run in one job and the cheap ones in another)."""
    from ..evaluation import load_tree
    folds = {}
    for out in outs:
        for p in sorted(Path(out).glob("fold*/result.json")):
            f = load_tree(p.parent)
            if f["fold"] not in folds:
                folds[f["fold"]] = f
            elif not np.array_equal(folds[f["fold"]]["test_items"], f["test_items"]):
                raise ValueError(f"fold {f['fold']}: {p.parent} has other test rows than the folds before it")
            else:
                folds[f["fold"]]["methods"].update(f["methods"])
    return [folds[k] for k in sorted(folds)]
