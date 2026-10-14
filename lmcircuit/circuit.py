"""The circuit: a pool of judges, a belief model over them, and the DP's policies for an objective and every cost price.

    circuit = Circuit(config).fit(ds, train_idx, val_idx)        # pool, belief model, validation curve
    circuit.objective("balanced_accuracy")                        # another objective: same belief model, new curve
    point = circuit.at_budget(0.1)                                # (weight, price) chosen on validation
    pred, cost, asked = circuit.decide(ds, items, [point])        # decisions on recorded answers
    decision = circuit.policy(features, point).decide({"judge-a": 1})   # live: ask whom next, or predict
    circuit.save(path); Circuit.load(path)
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from . import belief as belief_mod
from . import dp, objectives
from .belief import BeliefNet
from .config import BeliefConfig, CircuitConfig, memory_cells, resolve_device
from .data import MISSING, Dataset
from .metrics import scores
from .pool import coverage_pool


@dataclass(frozen=True)
class Decision:
    """The next judge to ask (`ask`), or the prediction (`predict`, with the belief over classes) when stopping."""
    ask: str | None = None
    predict: int | None = None
    belief: np.ndarray | None = None


class Policy:
    """One item's policy at one operating point, solved once: `decide(answers)` for any answers seen so far."""

    def __init__(self, pool: list[str], weights: np.ndarray, post: np.ndarray, actions: np.ndarray, space: dp.StateSpace):
        self.pool, self.weights, self.post, self.actions, self.space = pool, weights, post, actions, space

    def decide(self, answers: dict[str, int]) -> Decision:
        if set(answers) - set(self.pool) or any(not 0 <= int(v) < self.space.A for v in answers.values()):
            raise ValueError(f"answers must be values in [0, {self.space.A}) from pool judges {self.pool}")
        a = np.asarray([int(answers.get(n, 0)) for n in self.pool])
        state = int(self.space.state_of(a[None], np.asarray([[n in answers for n in self.pool]]))[0])
        if (act := int(self.actions[state])) >= 0:
            return Decision(ask=self.pool[act])
        p = self.post[state]
        return Decision(predict=int(np.argmax(p * self.weights)), belief=p)


class Clock:
    """Wall-clock seconds per stage, the device's queued work finished before each reading."""

    def __init__(self, device):
        self.device, self.seconds = torch.device(device), {}

    def __call__(self, stage: str):
        clock = self

        class _Stage:
            def __enter__(self):
                belief_mod._sync(clock.device)
                self.started = time.perf_counter()

            def __exit__(self, *exc):
                belief_mod._sync(clock.device)
                clock.seconds[stage] = clock.seconds.get(stage, 0.0) + time.perf_counter() - self.started
        return _Stage()


class Circuit:
    def __init__(self, config: CircuitConfig | None = None):
        self.cfg = config or CircuitConfig()
        self.lambdas = dp.lambda_grid(self.cfg.lambdas, self.cfg.lambda_lo, self.cfg.lambda_hi)

    # ---------------------------------------------------------------- fitting
    def fit(self, ds: Dataset, train_idx, val_idx, worlds: int | None = None, seed: int = 0,
            curve: bool = True) -> "Circuit":
        """The pool (chosen on train + validation), the belief model on the labelled training items (their labels and
        every recorded call of the pool judges; early stopping, the number of types and the temperatures on
        validation), and the validation curve of config.target. `worlds`: the recorded calls per judge an item is
        decided on (default min(5, recorded)); `seed`: the belief model's training seed; `curve`: False leaves the
        validation curve to the caller (set_curve; lc.evaluate computes it for every objective in one pass).
        record["seconds"]: each stage's wall-clock time."""
        if ds.labels is None or ds.features is None:
            raise ValueError("fit needs labels (lmcircuit.label) and features (lmcircuit.embed)")
        cfg = self.cfg
        self.device = resolve_device(cfg.device)
        clock = Clock(self.device)
        self.worlds, self.seed = int(worlds or min(5, ds.answers.shape[2])), int(seed)
        train_idx = np.asarray(train_idx)[ds.labelled[np.asarray(train_idx)]]
        val_idx = np.asarray(val_idx)[ds.labelled[np.asarray(val_idx)]]
        self.scale, self.n_classes, self.classes, self.n_answers = ds.scale(), ds.n_classes, ds.classes, ds.n_answers
        with clock("pool"):
            plan = ds.plan_cost_for(train_idx)
            both = np.concatenate([train_idx, val_idx])
            cols, self.pool_info = coverage_pool(ds, both, cfg.pool_for(ds.n_answers), plan)
            self.pool = [ds.judge_names[j] for j in cols]
            self.mean_plan = plan[both][:, cols].mean(axis=0)                  # USD per call, per pool judge
        calls = ds.with_judges(self.pool)
        data = {"x": ds.features, "x_swap": ds.features_swap, "answers": calls.answers, "orders": calls.call_orders(),
                "labels": ds.labels}
        model_kw = dict(d_features=ds.features.shape[1], n_judges=len(self.pool), n_classes=ds.n_classes,
                        n_answers=ds.n_answers, n_orders=ds.n_orders, swap=ds.features_swap is not None, d_model=cfg.belief.d_model,
                        d_hidden=cfg.belief.d_hidden)
        with clock("belief"):
            self.belief, self.record = belief_mod.fit(data, train_idx, val_idx, model_kw, cfg.belief, self.device, seed)
        self.train_labels = ds.labels[train_idx]
        self._val = (ds, val_idx[calls.complete(self.worlds)[val_idx]])        # replay needs every call
        if not len(self._val[1]):
            raise ValueError("no validation item has a recorded call of every pool judge")
        if curve:
            with clock("validation"):
                self.objective(cfg.target)
        self.record["seconds"] = {**clock.seconds, "belief_stages": self.record.pop("seconds")}
        return self

    def objective(self, name: str) -> "Circuit":
        """Optimise `name` (lmcircuit.objectives): its class weights, and every operating point's validation cost and
        metric, val_cost and val_metric (G weights, L prices)."""
        self.target, self.weights = name, objectives.weights(name, self.train_labels)
        ds, val = self._val
        out = self.run(ds, val, self.weights, asked=False)
        self.val_cost = np.stack([c.mean(axis=1) for c in out["cost"]])
        self.val_metric = scores(np.stack(out["pred"]), np.repeat(ds.labels[val], self.worlds, axis=0))[name]
        return self

    def set_curve(self, name: str, weights: np.ndarray, val_cost: np.ndarray, val_metric: np.ndarray) -> "Circuit":
        """The validation curve of objective `name` computed elsewhere (weights (G, K), val_cost and val_metric (G, L))."""
        self.target, self.weights, self.val_cost, self.val_metric = name, weights, val_cost, val_metric
        return self

    def at_budget(self, budget: float) -> tuple[int, int]:
        """The operating point (weight index, price index) for a budget (rescaled cost per item), on validation."""
        return objectives.pick(self.val_cost, self.val_metric, budget)

    # ---------------------------------------------------------------- decisions on recorded answers
    def decide(self, ds: Dataset, items, points: list[tuple[int, int]]):
        """At each operating point, on `items` x worlds of `ds` from their recorded answers (labels not needed):
        predictions (P, rows), charged cost rescaled (P, rows), asked (P, rows, M)."""
        keep = [sorted({l for g_, l in points if g_ == g}) for g in range(len(self.weights))]
        out = self.run(ds, items, self.weights, keep)
        at = lambda k, g, l: out[k][g][keep[g].index(l)]  # noqa: E731
        return tuple(np.stack([at(k, g, l) for g, l in points]) for k in ("pred", "cost", "asked"))

    @torch.no_grad()
    def run(self, ds: Dataset, items, weights: np.ndarray, keep: list | None = None, asked: bool = True,
            beliefs: bool = False) -> dict:
        """One pass over `items` x worlds of `ds`, in chunks of rows sized by the memory budget: the belief model's
        tables once per chunk, then the DP for every weight vector at once (weights (G, K)). Returns, per g, the
        decisions at the price indices keep[g] (None: every price): pred [g] (L', rows), cost [g] (L', rows), asked [g]
        (L', rows, M); with `beliefs`, the belief where each policy stops, belief [g] (L', rows, K), and P(Y | x)
        before any judge and after every pool judge: prior, full (rows, K). A judge without a recorded call on a row
        is never asked there."""
        items = np.asarray(items)
        rows, data, space = self._rows(ds, items)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1, self.n_classes)
        G, L, M, A = len(weights), len(self.lambdas), space.M, space.A
        keep = [list(range(L)) if k is None else list(k) for k in (keep if keep is not None else [None] * G)]
        cols = [g * L + l for g in range(G) for l in keep[g]]
        plan = self._plan(ds)[np.repeat(items, self.worlds)] / self.scale
        call_cost = rows.cost[:, :, 0] / self.scale
        out = {k: [[] for _ in range(G)] for k in ("pred", "cost", "asked", "belief")}
        full = space.state_of(np.maximum(data["answers"], 0), data["asked"])
        prior, after = [], []
        per_row = (G * L if cols else 1) * space.n_states + space.n_states * M * A
        for sl, post, nu in self._tables(data, space, per_row):
            p = post.cpu().numpy()
            a = np.maximum(data["answers"][sl], 0)
            if cols:
                pen = np.where(data["asked"][sl][:, None, :], dp.penalties(self.lambdas, plan[sl]), np.inf)
                _, A = dp.solve(space, post, nu, pen, weights)
                A = A[:, cols].cpu().numpy()
                start = 0
                for g in range(G):
                    if not keep[g]:
                        continue
                    took, pred, final = dp.replay(space, A[:, start:start + len(keep[g])], p, a, weights[g])
                    start += len(keep[g])
                    out["pred"][g].append(pred.T.astype(np.int8))
                    out["cost"][g].append((took * call_cost[sl][:, None, :]).sum(axis=2).T.astype(np.float32))
                    if asked:
                        out["asked"][g].append(took.transpose(1, 0, 2))
                    if beliefs:
                        out["belief"][g].append(final.transpose(1, 0, 2).astype(np.float32))
            if beliefs:
                prior.append(p[:, space.root]); after.append(p[np.arange(len(p)), full[sl]])
        for k in ("pred", "cost", "asked", "belief"):
            out[k] = [np.concatenate(v, axis=1) if v else None for v in out[k]]
        if beliefs:
            out["prior"], out["full"] = np.concatenate(prior), np.concatenate(after)
        return out

    # ---------------------------------------------------------------- live use
    @torch.no_grad()
    def policy(self, features: np.ndarray, point: tuple[int, int], cost: dict[str, float] | None = None,
               orders: dict[str, int] | None = None, features_swap: np.ndarray | None = None) -> Policy:
        """One item's policy at an operating point. `cost`: its planning cost per pool judge (USD), default the
        training means; `orders`: the presentation order each pool judge's call will use (pairwise tasks, default 0);
        `features_swap`: the item's features with its candidates exchanged (needed when the circuit has a swap)."""
        g, l = point
        space = dp.StateSpace(len(self.pool), self.n_answers, self.device)
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32)[None], device=self.device)  # noqa: E731
        o = torch.as_tensor([[int((orders or {}).get(n, 0)) for n in self.pool]], device=self.device)
        if self.belief.swap and features_swap is None:
            raise ValueError("this circuit's belief model is swap-equivariant: give the item's features_swap")
        post, nu = self.belief.tables(t(features), None if features_swap is None else t(features_swap), o, space,
                                      memory_cells(self.cfg.max_cells, self.device))
        c = np.asarray([(cost or {}).get(n, m) for n, m in zip(self.pool, self.mean_plan)], dtype=np.float64)
        _, A = dp.solve(space, post, nu, dp.penalties(self.lambdas[[l]], c[None] / self.scale), self.weights[g])
        return Policy(self.pool, self.weights[g], post[0].cpu().numpy(), A[0, 0].cpu().numpy(), space)

    # ---------------------------------------------------------------- internals
    def _data(self, rows: Dataset) -> dict:
        return {"x": rows.features.astype(np.float32),
                "x_swap": None if rows.features_swap is None else rows.features_swap.astype(np.float32),
                "orders": rows.call_orders()[:, :, 0], "answers": rows.answers[:, :, 0],
                "asked": rows.answers[:, :, 0] != MISSING}

    def _rows(self, ds: Dataset, items):
        rows, _ = ds.with_judges(self.pool).take(np.asarray(items)).by_world(self.worlds)
        return rows, self._data(rows), dp.StateSpace(len(self.pool), self.n_answers, self.device)

    def _tables(self, data: dict, space: dp.StateSpace, per_row: float):
        """The belief model's tables, in chunks of rows sized by the memory budget: (slice, post, nu)."""
        cells = memory_cells(self.cfg.max_cells, self.device)
        per = max(1, int(cells // max(1.0, per_row)))
        t = lambda a, sl: None if a is None else torch.as_tensor(a[sl], device=self.device)  # noqa: E731
        for s in range(0, len(data["x"]), per):
            sl = slice(s, s + per)
            post, nu = self.belief.tables(t(data["x"], sl), t(data["x_swap"], sl), t(data["orders"], sl), space, cells)
            yield sl, post, nu

    def _plan(self, ds: Dataset) -> np.ndarray:
        """(N, pool) planning cost of `ds`'s items: its plan_cost, else the training means."""
        if ds.plan_cost is not None:
            return ds.plan_cost[:, [ds.judge_names.index(n) for n in self.pool]]
        return np.broadcast_to(self.mean_plan, (ds.n_items, len(self.pool))).copy()

    # ---------------------------------------------------------------- storage
    STATE = ("worlds", "seed", "scale", "n_classes", "classes", "n_answers", "pool", "pool_info", "record", "target")

    def save(self, path) -> Path:
        """The fitted circuit (belief model, validation curve, configuration) in one directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"kw": self.belief.kw, "belief": self.belief.state_dict()}, path / "belief.pt")
        np.savez(path / "arrays.npz", mean_plan=self.mean_plan, weights=self.weights, val_cost=self.val_cost,
                 val_metric=self.val_metric, train_labels=self.train_labels)
        (path / "circuit.json").write_text(json.dumps({"config": asdict(self.cfg),
                                                       **{k: getattr(self, k) for k in self.STATE}}, default=str, indent=1))
        return path

    @classmethod
    def load(cls, path, device: str = "auto") -> "Circuit":
        path = Path(path)
        meta = json.loads((path / "circuit.json").read_text())
        cfg = meta.pop("config")
        belief = {**cfg["belief"], "types": tuple(cfg["belief"]["types"])}
        circuit = cls(CircuitConfig(**{**cfg, "belief": BeliefConfig(**belief)}))
        for k in cls.STATE:
            setattr(circuit, k, meta.get(k))
        circuit.device = resolve_device(device)
        saved = torch.load(path / "belief.pt", map_location=circuit.device, weights_only=True)
        circuit.belief = BeliefNet(**saved["kw"]).to(circuit.device)
        circuit.belief.load_state_dict(saved["belief"])
        circuit.belief.eval()
        a = np.load(path / "arrays.npz")
        for k in ("mean_plan", "weights", "val_cost", "val_metric", "train_labels"):
            setattr(circuit, k, a[k])
        return circuit
