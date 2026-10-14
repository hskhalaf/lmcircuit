"""Prior features: each item's text embedded by a frozen encoder (Qwen3-Embedding, in its query format).

    ds = lc.embed(ds)                            # the layout the dataset declares (ds.meta["prior_layout"]), else "item"
    ds = lc.embed(ds, layout={"kind": "pair", "a": "candidate_a", "b": "candidate_b"})

Layouts:
    item       one text, the task description then the item: features = its embedding.
    criterion  a criterion and the context it is judged on, embedded apart (HealthBench's rubric item and
               conversation): features = [c, x, c * x], so the model can relate what is judged to what it is judged on.
    pair       two candidates, each embedded with its context (Arena's two responses, each with the prompt): features
               [(a + b) / 2, a - b, |a - b|, a * b, s], s the standardised log lengths in characters of the `lengths`
               parts (e.g. the prompt), of each candidate (`length_a`, `length_b`: the parts to measure, default the
               embedded texts) and of their sum, and the task as indicators; features_swap the same with the
               candidates exchanged (the belief model is then exactly equivariant to the exchange). The two candidates'
               lengths share one standardisation, so the exchange is an exact coordinate swap.
The standardisation uses every item's text (no labels), so it is the same in every fold.

The text is the item as the judges see it, the judged object included (a response, a trajectory): everything in hand
before any judge is paid. Every method compared reads the same features. Fine-tuning the encoder on each fold's
training items is not offered: with the earlier belief model it left OLL BBH's frontier area unchanged (74.1 frozen,
74.0 fine-tuned, three seeds) and raised Decision Index's by 2.2 points (77.4 to 79.6), at the price of training the
encoder in every fold; it was not tried with the current belief model.
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from .config import EmbedConfig
from .data import Dataset


def query(instruction: str, text: str) -> str:
    """Qwen3-Embedding's query format (model card)."""
    return f"Instruct: {instruction}\nQuery:{text}"


def item_texts(ds: Dataset) -> list[str]:
    """Each item's text: its task description (if any), then the item."""
    task = ds.prior_text.get("task_description", np.full(ds.n_items, ""))
    return [f"{t}\n\n{i}" if t else str(i) for t, i in zip(task, ds.prior_text["item"])]


def load_encoder(cfg: EmbedConfig, device: str):
    from sentence_transformers import SentenceTransformer
    import torch
    model = SentenceTransformer(cfg.model, revision=cfg.resolved_revision(), device=device,
                                model_kwargs={"dtype": torch.bfloat16 if device.startswith("cuda") else torch.float32})
    model.max_seq_length = cfg.max_seq_length
    return model


def encode(model, texts: list[str], cfg: EmbedConfig) -> np.ndarray:
    """L2-normalised embeddings (n, d) of texts in the query format, longest first, in batches of at most
    cfg.batch_tokens padded tokens."""
    items = [query(cfg.instruction, t) for t in texts]
    lengths = np.asarray([len(i) for i in model.tokenizer(items, truncation=True, max_length=cfg.max_seq_length)["input_ids"]])
    order, out, start = np.argsort(-lengths, kind="stable"), None, 0
    while start < len(order):
        size = int(max(1, min(cfg.batch_size, cfg.batch_tokens // max(1, lengths[order[start]]))))
        rows = order[start:start + size]
        emb = model.encode([items[i] for i in rows], batch_size=size, normalize_embeddings=True, show_progress_bar=False)
        out = np.zeros((len(items), emb.shape[1]), dtype=np.float32) if out is None else out
        out[rows] = emb
        start += size
    return out


def pair_features(a: np.ndarray, b: np.ndarray, scalars: np.ndarray, swap_cols: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Pair features and their exchange: a, b (N, d) the candidates' embeddings; scalars (N, s) raw item scalars, of
    which the columns swap_cols are the two candidates' (standardised together, exchanged by the swap)."""
    scalars = np.asarray(scalars, dtype=np.float64)
    mean, std = scalars.mean(axis=0), scalars.std(axis=0)
    i, j = swap_cols
    both = scalars[:, [i, j]].reshape(-1)
    mean[[i, j]], std[[i, j]] = both.mean(), both.std()
    std[std < 1e-6] = 1.0
    swapped = scalars.copy()
    swapped[:, [i, j]] = swapped[:, [j, i]]

    def assemble(left, right, raw):
        return np.concatenate([(left + right) / 2, left - right, np.abs(left - right), left * right,
                               (raw - mean) / std], axis=1).astype(np.float32)
    return assemble(a, b, scalars), assemble(b, a, swapped)


def embed(ds: Dataset, cfg: EmbedConfig | None = None, layout: dict | None = None, device: str = "auto") -> Dataset:
    """The dataset with `features` (and, for a pair layout, `features_swap`): its texts embedded by the frozen encoder in
    `layout` (default ds.meta["prior_layout"], else {"kind": "item"}; see the module docstring)."""
    from .config import resolve_device
    cfg = cfg or EmbedConfig()
    layout = dict(layout or ds.meta.get("prior_layout") or {"kind": "item"})
    started = time.time()
    model = load_encoder(cfg, resolve_device(device))
    text = lambda part: [str(t) for t in ds.prior_text[part]]  # noqa: E731
    swap = None
    if layout["kind"] == "item":
        features = encode(model, item_texts(ds), cfg)
    elif layout["kind"] == "criterion":
        c, x = encode(model, text(layout["criterion"]), cfg), encode(model, text(layout["context"]), cfg)
        features = np.concatenate([c, x, c * x], axis=1)
    elif layout["kind"] == "pair":
        a, b = encode(model, text(layout["a"]), cfg), encode(model, text(layout["b"]), cfg)
        chars = lambda part: np.asarray([len(t) for t in text(part)], dtype=np.float64)  # noqa: E731
        ca, cb = chars(layout.get("length_a", layout["a"])), chars(layout.get("length_b", layout["b"]))
        others = [chars(p) for p in layout.get("lengths", [])]
        tasks = sorted(set(ds.task))
        scalars = np.stack([np.log1p(c) for c in (*others, ca, cb, ca + cb)]
                           + [(ds.task == t).astype(np.float64) for t in tasks[1:]], axis=1)
        features, swap = pair_features(a, b, scalars, (len(others), len(others) + 1))
    else:
        raise ValueError(f"unknown layout {layout['kind']!r}: item, criterion or pair")
    meta = {"model": cfg.model, "revision": cfg.resolved_revision(), "max_seq_length": cfg.max_seq_length,
            "layout": layout, "seconds": time.time() - started}
    return replace(ds, features=features, features_swap=swap, meta={**ds.meta, "embedding": meta})
