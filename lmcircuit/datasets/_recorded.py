"""Recorded judge calls: answers recorded once and stored apart from the items, which a loader rebuilds from the pinned
public source and joins to the calls by id. The calls of the four datasets we queried are not distributed: without
<raw>/recorded, a loader returns the items alone (no judges), and the module's `recipe` reruns the calls with
lc.collect.

Raw layout:
    <raw>/recorded/calls.npz     ids (N,) str; judges (M,) str; answers (N, M, R) int8, MISSING (-1) where no call was
                                 recorded or its answer did not parse; cost (N, M, R) float64 USD (0 without a call);
                                 tokens_in, tokens_out (N, M, R) int64 (0 without a call). Optional:
                                 tokens_reasoning (N, M, R) int64 (-1: not reported); probs (N, M, R, K) float32, each
                                 class's probability at the verdict (NaN: not recorded); probs_source (N, M, R) int8
                                 (what produced it; meaning in calls.json); orders (N, M, R) int8, the order a pairwise
                                 call presented the candidates in (0 canonical, 1 swapped; answers are always in the
                                 canonical label order); plan_cost (N, M) float64 USD; labels (N, K) float64 counts
                                 (labels recorded with the calls, e.g. a grader's)
    <raw>/recorded/calls.json    classes; judges (name, model, revision, prices, request fields, ...); prompt (file,
                                 sha256); parser; calls per item; what probs_source and labels mean; provenance (run
                                 root, git revision, export date)
    <raw>/source/                the public source files, written by the dataset's `fetch`
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data import Dataset
from ._common import make_judge


def write(out_dir, arrays: dict, meta: dict) -> Path:
    """Write <out_dir>/calls.npz and calls.json (the layout above)."""
    a = {k: np.asarray(v) for k, v in arrays.items()}
    if len(set(a["ids"].tolist())) != len(a["ids"]) or [j["name"] for j in meta["judges"]] != a["judges"].tolist():
        raise ValueError("recorded ids must be unique and calls.json must list the judges of calls.npz in order")
    a["ids"], a["judges"], a["answers"] = a["ids"].astype(str), a["judges"].astype(str), a["answers"].astype(np.int8)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "calls.npz", **a)
    (out / "calls.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    return out


@dataclass
class Recorded:
    arrays: dict            # name -> array (calls.npz)
    meta: dict              # calls.json

    @property
    def ids(self) -> list[str]:
        return self.arrays["ids"].astype(str).tolist()


def read(raw_dir) -> Recorded | None:
    """The recorded calls under <raw>/recorded, or None when there are none."""
    path = Path(raw_dir) / "recorded"
    if not (path / "calls.npz").exists():
        return None
    with np.load(path / "calls.npz", allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    return Recorded(arrays, json.loads((path / "calls.json").read_text()))


def pick(recorded: Recorded | None, rows: dict[str, dict], dataset: str) -> list[tuple[str, dict]]:
    """(id, row) of the items: the recorded ids in recorded order, or every source row without recorded calls."""
    return list(rows.items()) if recorded is None else list(zip(recorded.ids, join(recorded, rows, dataset)))


def join(recorded: Recorded, rows: dict[str, dict], dataset: str) -> list[dict]:
    """The source rows of the recorded ids, in recorded order; every recorded id must be found."""
    missing = [i for i in recorded.ids if i not in rows]
    if missing:
        raise ValueError(f"{dataset}: {len(missing)} of {len(recorded.ids)} recorded ids are not in the source "
                         f"(first {missing[:5]}); is <raw>/source the pinned revision?")
    return [rows[i] for i in recorded.ids]


def dataset(recorded: Recorded | None, items: list[dict], judges: dict, classes: list[str], labels, prior_parts,
            meta: dict) -> Dataset:
    """The Dataset of the recorded calls on the joined items (without recorded calls: the items alone, for lc.collect).
    judges: the loader's CONFIG entries, which must be the recorded judges, in order, at the recorded prices."""
    if recorded is None:
        ds = Dataset.from_records(items, n_classes=len(classes), classes=classes, labels=labels, prior_parts=(),
                                  meta=meta)
        ds.prior_text = {p: np.asarray([str(it.get(p, "")) for it in items], dtype=object) for p in prior_parts}
        return ds
    a = recorded.arrays
    names = a["judges"].astype(str).tolist()
    recorded_prices = [(j["price_in"], j["price_out"]) for j in recorded.meta["judges"]]
    if names != list(judges) or not np.allclose(recorded_prices, [judges[n]["price"] for n in names], rtol=0, atol=1e-12):
        raise ValueError(f"recorded judges or prices differ from the loader's CONFIG: {names}")
    ds = Dataset.from_records(
        items, [make_judge(n, judges[n]) for n in names], a["answers"].astype(np.int64), a["cost"],
        n_classes=len(classes), classes=classes, labels=labels, prior_parts=(), plan_cost=a.get("plan_cost"),
        probs=a.get("probs"), orders=a.get("orders"), meta={**meta, "recorded": {k: recorded.meta[k] for k in
                                                          ("calls", "probs_source", "labels", "provenance")
                                                          if k in recorded.meta}})
    # object arrays: texts run to 70k characters, and a fixed-width numpy string array takes 4 bytes x the longest
    # text for every item (arena: about 40 GB)
    ds.prior_text = {p: np.asarray([str(it.get(p, "")) for it in items], dtype=object) for p in prior_parts}
    return ds
