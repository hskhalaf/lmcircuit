"""Labels for items that have none: one or more grader models answer each item, possibly several times; the label is
the count of their answers (soft labels, like several human annotators).

    ds = lc.label(ds, prompt, [lc.Backend.vllm(url, "Qwen/Qwen3-235B-A22B")], samples=3)

Graders should not share a model family with the pool's judges: their labels would favour those judges (the labels
then measure agreement with the grader, not correctness). `label` warns when they do.
"""
from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np

from .data import Dataset
from .llm import Backend, Prompt, call_many, family, parse_choice


def label(ds: Dataset, prompt: Prompt, graders: list[Backend], samples: int = 1, max_tokens: int = 4096,
          fields=None, cache_path="lmcircuit_cache.sqlite") -> Dataset:
    """Replace the dataset's labels with the graders' answer counts. fields: per item, the dict that fills the prompt
    (default: Dataset.fields(), the id and every prior-text part)."""
    if len(prompt.classes) != ds.n_classes:
        raise ValueError(f"the prompt names {len(prompt.classes)} classes, the dataset has {ds.n_classes}")
    if ds.labels is not None and ds.labelled.any():
        warnings.warn("the dataset already has labels: they are replaced by the graders'")
    judge_families = {family(j.model or j.name) for j in ds.judges}
    clash = [g.model for g in graders if family(g.model) in judge_families]
    if clash:
        warnings.warn(f"graders {clash} share a model family with the judges: the labels would favour those judges")
    convs = [prompt.messages(f) for f in (fields or ds.fields())]
    counts = np.zeros((ds.n_items, ds.n_classes))
    runs = []
    for g in graders:
        for s in range(samples):
            out = call_many(g, convs, max_tokens, seeds=[s] * len(convs), cache_path=cache_path)
            parsed = np.asarray([parse_choice(o["text"], prompt.classes) for o in out])
            ok = parsed >= 0
            counts[np.flatnonzero(ok), parsed[ok]] += 1
            runs.append({"grader": g.model, "sample": s, "parsed": float(ok.mean()),
                         "errors": int(sum("error" in o for o in out))})
    got = counts.sum(axis=1) > 0
    if not got.all():
        warnings.warn(f"{int((~got).sum())} of {ds.n_items} items got no parsable grade")
    agreement = float((counts.max(axis=1)[got] / counts.sum(axis=1)[got]).mean()) if got.any() else float("nan")
    return replace(ds, labels=counts, classes=ds.classes or list(prompt.classes),
                   meta={**ds.meta, "labels": {"graders": [g.model for g in graders], "samples": samples, "runs": runs,
                                               "agreement_with_majority": agreement}})
