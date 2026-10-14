"""Judge answers for items whose calls are not recorded: each judge (a backend and a prompt) answers every item,
`repeats` times; answers, tokens and USD cost are recorded, and the planning cost is set from what is known before a
call (input tokens x input price + the judge's 95th-percentile output length x output price). A backend with
`logprobs` also records each call's class probabilities at its verdict (Dataset.probs); a prompt with `permute` shows
those fields in a per-call random order, and answers and probabilities are mapped back to the label order.

    judges = [lc.JudgeSpec("qwen3-8b", lc.Backend.vllm(url_a, "Qwen/Qwen3-8B", price_in=0.05, price_out=0.2), prompt),
              lc.JudgeSpec("gpt-5-mini", lc.Backend.openai("gpt-5-mini", price_in=0.25, price_out=2.0), prompt)]
    ds = lc.collect(ds, judges, repeats=3)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np

from .data import MISSING, Dataset, Judge
from .llm import Backend, Prompt, call_many, locate_choice, verdict_probs

OUTPUT_QUANTILE = 0.95        # the output length a judge is planned with: what one would set as its token limit


@dataclass
class JudgeSpec:
    name: str
    backend: Backend
    prompt: Prompt
    max_tokens: int = 4096


def collect(ds: Dataset, judges: list[JudgeSpec], repeats: int = 1, fields=None,
            cache_path="lmcircuit_cache.sqlite", max_failed: float = 0.2) -> Dataset:
    """Replace the dataset's judges and calls with the collected ones (MISSING where a call failed or its answer did
    not parse). Warns when a judge's failed or unparsed share exceeds `max_failed`."""
    for j in judges:
        if len(j.prompt.classes) != ds.n_classes:
            raise ValueError(f"judge {j.name}: the prompt names {len(j.prompt.classes)} classes, the dataset has "
                             f"{ds.n_classes}")
    fields = fields or ds.fields()
    N, M = ds.n_items, len(judges)
    answers = np.full((N, M, repeats), MISSING, dtype=np.int64)
    cost, tokens_in, tokens_out = (np.zeros((N, M, repeats)) for _ in range(3))
    K = ds.n_classes
    probs = np.full((N, M, repeats, K), np.nan, dtype=np.float32) if any(j.backend.logprobs for j in judges) else None
    for m, j in enumerate(judges):
        for r in range(repeats):
            orders = [j.prompt.order(str(f["id"]), j.name, r) for f in fields]
            convs = [j.prompt.messages(f, o) for f, o in zip(fields, orders)]
            out = call_many(j.backend, convs, j.max_tokens, seeds=[r] * N, cache_path=cache_path)
            for n, o in enumerate(out):
                if "error" not in o:
                    shown, offset = locate_choice(o["text"], j.prompt.classes)
                    answers[n, m, r] = orders[n][shown] if shown >= 0 else MISSING
                    if probs is not None and shown >= 0:
                        p = verdict_probs(o.get("logprobs") or [], o["text"], offset, j.prompt.classes)
                        if p is not None:
                            probs[n, m, r, orders[n]] = p
                u = o.get("usage", {})
                tokens_in[n, m, r], tokens_out[n, m, r] = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
                cost[n, m, r] = j.backend.cost(u) or 0.0
        failed = float((answers[:, m] == MISSING).mean())
        if failed > max_failed:
            warnings.warn(f"judge {j.name}: {100 * failed:.0f}% of calls failed or did not parse")
    p_in = np.asarray([j.backend.price_in or 0.0 for j in judges])
    p_out = np.asarray([j.backend.price_out or 0.0 for j in judges])
    valid = answers != MISSING
    cap = np.asarray([np.quantile(tokens_out[:, m][valid[:, m]], OUTPUT_QUANTILE) if valid[:, m].any() else 0.0
                      for m in range(M)])
    plan = (tokens_in.max(axis=2) * p_in + cap * p_out) / 1e6
    meta = {**ds.meta, "calls": {"judges": [j.name for j in judges], "repeats": repeats,
                                 "parsed": float(valid.mean()), "output_cap_tokens": cap.tolist(),
                                 "tokens_in_mean": tokens_in.mean(axis=(0, 2)).tolist(),
                                 "tokens_out_mean": tokens_out.mean(axis=(0, 2)).tolist()}}
    return replace(ds, answers=answers, cost=cost, plan_cost=plan, probs=probs, meta=meta,
                   judges=[Judge(j.name, model=j.backend.model, price_in=j.backend.price_in, price_out=j.backend.price_out)
                           for j in judges])
