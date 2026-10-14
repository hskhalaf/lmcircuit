"""The data model: items, judges, recorded calls, labels and prior features (docs/method.md, section 2)."""
from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

MISSING = -1                       # an answer slot with no recorded call; as a 1-D label: no label


@dataclass(frozen=True)
class Judge:
    """One judge: a model under a prompt (or any classifier) answering in the label space at a price."""
    name: str
    model: str | None = None
    price_in: float | None = None          # USD per million input tokens
    price_out: float | None = None         # USD per million output tokens
    params_b: float | None = None          # total parameters (billions)
    active_params_b: float | None = None   # parameters active per token (a mixture of experts); None: all of them
    meta: dict = field(default_factory=dict)

    @property
    def compute_params_b(self) -> float | None:
        """The parameters one forward pass uses: what the compute proxy prices."""
        return self.active_params_b if self.active_params_b is not None else self.params_b


@dataclass
class Dataset:
    ids: np.ndarray                        # (N,) str
    task: np.ndarray                       # (N,) str
    group: np.ndarray                      # (N,) str: the fold unit (items of one source share a group)
    answers: np.ndarray                    # (N, M, R) int, MISSING where no call: a class, or one of the judges'
                                           # answer values (answer_classes)
    cost: np.ndarray                       # (N, M, R) float USD per call
    judges: list[Judge]
    n_classes: int = 2
    classes: list[str] | None = None       # class names, in label order
    answer_classes: list[int] | None = None   # judges answering in their own alphabet (e.g. a vote and a confidence):
                                           # the class each answer value votes for; None: answers are classes
    labels: np.ndarray | None = None       # (N, K) annotation counts
    prior_text: dict[str, np.ndarray] = field(default_factory=dict)   # part -> (N,) str, embedded for the prior
    features: np.ndarray | None = None     # (N, d) prior features
    features_swap: np.ndarray | None = None   # (N, d) a pairwise task's features with its two candidates exchanged;
                                           # the exchange reverses the classes (0: the first candidate) and the orders
    orders: np.ndarray | None = None       # (N, M, R) int, the order a pairwise call presented the candidates in (0 as
                                           # given, 1 exchanged; answers are always in the given order); None: one order
    plan_cost: np.ndarray | None = None    # (N, M) USD the circuit plans with, known before the call (e.g. input
                                           # tokens x price); None: each judge's mean recorded cost on training items
    probs: np.ndarray | None = None        # (N, M, R, K) the call's probability of each class at its verdict (NaN:
                                           # not recorded); what confidence-based baselines read
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_records(cls, items: list[dict], judges: list[Judge | str] = (), answers=None, cost=None,
                     n_classes: int = 2, labels=None, prior_parts=("task_description", "item"), classes=None,
                     plan_cost=None, probs=None, orders=None, answer_classes=None,
                     meta: dict | None = None) -> "Dataset":
        """items: dicts with `id`, `task` (default "all"), `group` (default the id) and the prior-text parts;
        answers (N, M) or (N, M, R) classes in [0, K) or MISSING; cost the same shape (USD); labels (N,) classes
        (MISSING: unlabelled) or (N, K) annotation counts; plan_cost (N, M) USD; probs answers' shape + (K,), each
        call's class probabilities at its verdict, NaN where not recorded; orders the answers' shape, each call's
        presentation order (pairwise tasks); answer_classes: when judges answer in their own alphabet, the class each
        answer value votes for (answers then lie in [0, len(answer_classes))). Without judges: items only, for
        lc.collect."""
        judges = [j if isinstance(j, Judge) else Judge(j) for j in judges]
        K = int(n_classes)
        answers = np.asarray(answers if answers is not None else np.zeros((len(items), 0)), dtype=np.int64)
        cost = np.asarray(cost if cost is not None else np.zeros(answers.shape), dtype=np.float64)
        if answers.ndim == 2:
            answers, cost = answers[..., None], cost[..., None]
        if answers.shape != cost.shape or answers.shape[:2] != (len(items), len(judges)):
            raise ValueError(f"answers {answers.shape} and cost {cost.shape} must be (items, judges[, calls])")
        A = K if answer_classes is None else len(answer_classes)
        if answer_classes is not None and not all(0 <= int(c) < K for c in answer_classes):
            raise ValueError(f"answer_classes must be classes in [0, {K})")
        if ((answers < MISSING) | (answers >= A)).any():
            raise ValueError(f"answers must be values in [0, {A}) or MISSING ({MISSING})")
        if labels is not None:
            labels = np.asarray(labels)
            if labels.ndim == 1:
                hard = labels.astype(np.int64)
                if ((hard < MISSING) | (hard >= K)).any():
                    raise ValueError(f"labels must be classes in [0, {K}) or MISSING ({MISSING})")
                labels = np.zeros((len(hard), K))
                labels[hard >= 0, hard[hard >= 0]] = 1.0
            labels = labels.astype(np.float64)
            if labels.shape != (len(items), K) or (labels < 0).any():
                raise ValueError(f"labels must be (items,) classes or ({len(items)}, {K}) non-negative counts")
        if classes is not None and len(classes) != K:
            raise ValueError(f"{len(classes)} class names for {K} classes")
        if probs is not None:
            probs = np.asarray(probs, dtype=np.float32)
            probs = probs[:, :, None] if probs.ndim == 3 else probs
            if probs.shape != (*answers.shape, K):
                raise ValueError(f"probs {probs.shape} must be {(*answers.shape, K)}")
        if orders is not None:
            orders = np.asarray(orders, dtype=np.int64)
            orders = orders[..., None] if orders.ndim == 2 else orders
            if orders.shape != answers.shape or (orders < 0).any():
                raise ValueError(f"orders {orders.shape} must be non-negative with the answers' shape {answers.shape}")
            orders = np.where(answers >= 0, orders, 0)
        return cls(ids=np.asarray([str(r["id"]) for r in items]), task=np.asarray([str(r.get("task", "all")) for r in items]),
                   group=np.asarray([str(r.get("group", r["id"])) for r in items]), answers=answers, cost=cost,
                   judges=judges, n_classes=K, classes=None if classes is None else list(classes), labels=labels,
                   prior_text={p: np.asarray([str(r.get(p, "")) for r in items]) for p in prior_parts},
                   plan_cost=None if plan_cost is None else np.asarray(plan_cost, dtype=np.float64),
                   probs=probs, orders=orders,
                   answer_classes=None if answer_classes is None else [int(c) for c in answer_classes],
                   meta=dict(meta or {}))

    # ---------------------------------------------------------------- shape
    @property
    def n_items(self) -> int:
        return len(self.ids)

    @property
    def n_judges(self) -> int:
        return len(self.judges)

    @property
    def judge_names(self) -> list[str]:
        return [j.name for j in self.judges]

    @property
    def n_answers(self) -> int:
        """The judges' answer values: the classes, or their own alphabet (answer_classes)."""
        return self.n_classes if self.answer_classes is None else len(self.answer_classes)

    def votes(self) -> np.ndarray:
        """(N, M, R) the class each call votes for (MISSING where no call): the answers themselves unless the judges
        answer in their own alphabet."""
        if self.answer_classes is None:
            return self.answers
        return np.where(self.answers >= 0, np.asarray(self.answer_classes)[np.maximum(self.answers, 0)], MISSING)

    @property
    def n_orders(self) -> int:
        return 1 if self.orders is None else int(self.orders.max(initial=0)) + 1

    def call_orders(self) -> np.ndarray:
        """(N, M, R) each call's presentation order (0 without orders)."""
        return np.zeros(self.answers.shape, dtype=np.int64) if self.orders is None else self.orders

    @property
    def valid(self) -> np.ndarray:
        return self.answers != MISSING

    @property
    def labelled(self) -> np.ndarray:
        """(N,) items with at least one annotation."""
        return np.zeros(self.n_items, dtype=bool) if self.labels is None else self.labels.sum(axis=1) > 0

    def scale(self) -> float:
        """Cost rescaling: the most expensive judge's mean cost per recorded call (1 if every call is free)."""
        c = np.where(self.valid, self.cost, np.nan)
        with np.errstate(invalid="ignore"):
            means = np.nanmean(c, axis=(0, 2)) if self.valid.any() else np.zeros(self.n_judges)
        top = np.nanmax(np.nan_to_num(means, nan=0.0)) if len(means) else 0.0
        return float(top) if top > 0 else 1.0

    def fields(self) -> list[dict]:
        """Per item, the fields a prompt is filled with: `id` and every prior-text part."""
        return [{"id": i, **{k: v[n] for k, v in self.prior_text.items()}} for n, i in enumerate(self.ids)]

    # ---------------------------------------------------------------- views
    def take(self, idx) -> "Dataset":
        idx = np.asarray(idx)
        return replace(self, ids=self.ids[idx], task=self.task[idx], group=self.group[idx], answers=self.answers[idx],
                       cost=self.cost[idx], labels=None if self.labels is None else self.labels[idx],
                       prior_text={k: v[idx] for k, v in self.prior_text.items()},
                       features=None if self.features is None else self.features[idx],
                       features_swap=None if self.features_swap is None else self.features_swap[idx],
                       orders=None if self.orders is None else self.orders[idx],
                       plan_cost=None if self.plan_cost is None else self.plan_cost[idx],
                       probs=None if self.probs is None else self.probs[idx])

    def with_judges(self, names) -> "Dataset":
        cols = [self.judge_names.index(n) for n in names]
        return replace(self, judges=[self.judges[c] for c in cols], answers=self.answers[:, cols],
                       cost=self.cost[:, cols], plan_cost=None if self.plan_cost is None else self.plan_cost[:, cols],
                       probs=None if self.probs is None else self.probs[:, cols],
                       orders=None if self.orders is None else self.orders[:, cols])

    def by_world(self, worlds: int) -> tuple["Dataset", np.ndarray]:
        """One row per (item, world): world r is every judge's r-th call. Returns the rows and each row's item."""
        item = np.repeat(np.arange(self.n_items), worlds)
        world = np.tile(np.arange(worlds), self.n_items)
        rows = self.take(item)
        rows = replace(rows, ids=np.asarray([f"{i}#{w}" for i, w in zip(rows.ids, world)]),
                       answers=self.answers[item, :, world][:, :, None], cost=self.cost[item, :, world][:, :, None],
                       probs=None if self.probs is None else self.probs[item, :, world][:, :, None],
                       orders=None if self.orders is None else self.orders[item, :, world][:, :, None])
        return rows, item

    def plan_cost_for(self, train_idx) -> np.ndarray:
        """(N, M) USD the circuit plans with: plan_cost, else each judge's mean recorded cost on the training items."""
        if self.plan_cost is not None:
            return self.plan_cost
        train_idx = np.asarray(train_idx)
        c = np.where(self.valid[train_idx], self.cost[train_idx], np.nan)
        with np.errstate(invalid="ignore"):
            mean = np.nan_to_num(np.nanmean(c, axis=(0, 2)), nan=0.0)
        return np.broadcast_to(mean, (self.n_items, self.n_judges)).copy()

    def complete(self, worlds: int) -> np.ndarray:
        """(N,) items with a recorded call of every judge in each of the first `worlds` worlds."""
        return self.valid[:, :, :worlds].all(axis=(1, 2))

    def folds(self, k: int, seed: int = 0, val_share: float = 0.2):
        """Grouped k-fold cross-validation: groups shuffled with the seed and dealt round-robin into k parts. Yields
        (train, val, test) item indices for fold f: part f is test; of the other groups, a random `val_share`
        (seeded, per fold) is validation, the rest training."""
        names = np.unique(self.group)
        if k < 2 or len(names) < k:
            raise ValueError(f"folds need k >= 2 and at least k groups (k = {k}, {len(names)} groups)")
        rng = np.random.default_rng(seed)
        names = names[rng.permutation(len(names))]
        part = dict(zip(names, np.arange(len(names)) % k))
        for f in range(k):
            rest = [g for g in names if part[g] != f]
            val = set(np.asarray(rest, dtype=object)[rng.permutation(len(rest))[:max(1, int(round(val_share * len(rest))))]])
            role = np.asarray([2 if part[g] == f else 1 if g in val else 0 for g in self.group])
            yield np.flatnonzero(role == 0), np.flatnonzero(role == 1), np.flatnonzero(role == 2)

    # ---------------------------------------------------------------- checks
    def check(self) -> list[str]:
        """Problems worth a look before fitting (an empty list means none found)."""
        out = []
        if self.labels is None:
            out.append("no labels: run lmcircuit.label first")
        else:
            if not self.labelled.all():
                out.append(f"{int((~self.labelled).sum())} items have no label")
            share = self.labels / np.maximum(self.labels.sum(axis=1, keepdims=True), 1e-12)
            for t in np.unique(self.task):
                s = share[(self.task == t) & self.labelled].mean(axis=0) if ((self.task == t) & self.labelled).any() else None
                if s is not None and s.max() > 0.9:
                    out.append(f"task {t}: {100 * s.max():.0f}% of labels are one class")
        if len(np.unique(self.group)) == self.n_items:
            out.append("every item is its own group: items sharing a source (a conversation, a problem) must share one")
        miss = (~self.valid[:, :, 0]).mean(axis=0)
        for n, m in zip(self.judge_names, miss):
            if m == 1:
                out.append(f"judge {n}: no recorded call at all")
            elif m > 0:
                out.append(f"judge {n}: no call on {100 * m:.1f}% of items")
        if np.any((self.cost < 0) | (self.valid & ~np.isfinite(self.cost))):
            out.append("negative or non-finite costs")
        return out

    # ---------------------------------------------------------------- storage
    def save(self, path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {"ids": self.ids, "task": self.task, "group": self.group, "answers": self.answers, "cost": self.cost}
        for name in ("labels", "features", "features_swap", "orders", "plan_cost", "probs"):
            if getattr(self, name) is not None:
                arrays[name] = getattr(self, name)
        np.savez_compressed(path / "dataset.npz", **arrays)
        with gzip.open(path / "prior_text.json.gz", "wt", encoding="utf-8") as f:   # variable-length text, no pickle
            json.dump({k: [str(x) for x in v] for k, v in self.prior_text.items()}, f, ensure_ascii=False)
        (path / "manifest.json").write_text(json.dumps(
            {"n_classes": self.n_classes, "classes": self.classes, "answer_classes": self.answer_classes,
             "judges": [asdict(j) for j in self.judges], "meta": self.meta}, indent=1, default=str))
        return path

    @classmethod
    def load(cls, path) -> "Dataset":
        path = Path(path)
        a = np.load(path / "dataset.npz", allow_pickle=False)
        m = json.loads((path / "manifest.json").read_text())
        get = lambda k: a[k] if k in a.files else None  # noqa: E731
        if (path / "prior_text.json.gz").exists():
            with gzip.open(path / "prior_text.json.gz", "rt", encoding="utf-8") as f:
                prior = {k: np.asarray(v, dtype=object) for k, v in json.load(f).items()}
        else:                                                # datasets saved before 0.3: text inside the npz
            prior = {k.split("/", 1)[1]: a[k] for k in a.files if k.startswith("prior_text/")}
        return cls(ids=a["ids"], task=a["task"], group=a["group"], answers=a["answers"], cost=a["cost"],
                   judges=[Judge(**j) for j in m["judges"]], n_classes=m["n_classes"], classes=m.get("classes"),
                   answer_classes=m.get("answer_classes"),
                   labels=get("labels"), prior_text=prior,
                   features=get("features"), features_swap=get("features_swap"), orders=get("orders"),
                   plan_cost=get("plan_cost"),
                   probs=get("probs"), meta=m.get("meta", {}))
