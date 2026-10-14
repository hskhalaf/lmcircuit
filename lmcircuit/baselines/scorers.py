"""The fine-tuned models two baselines are defined with, trained per fold on its training items (a GPU; the `baselines`
extra: transformers).

frugalgpt  FrugalGPT's scoring function (Chen et al., 2023; FrugalGPT/scoring.py, llmcascade.py as released): one
           DistilBERT per judge reads "query answer" and classifies the answer as correct or not (two classes,
           cross-entropy); the score is the softmax probability of "correct". Examples: one response per query, as in
           the release (each training item's first recorded call of the judge), target = the answer's quality (the
           share of annotations agreeing with it: 1 or 0 under one annotation, a soft target otherwise). Settings of
           the release: distilbert-base-uncased, 8 epochs, batch 8, 500 warm-up steps, weight decay 0.01, learning
           rate 5e-5 (the Trainer default), the epoch with the lowest loss on a 55% hold-out of those examples (its
           own split, not the fold's validation items). Output: the score of every class as the answer, (N, M, K),
           so a call answering v on item i is scored S[i, j, v].
hybridllm  HybridLLM's router (Ding et al., ICLR 2024): DeBERTa-v3-large reads the query and predicts
           Pr[q(weak) >= q(strong)] (one logit, binary cross-entropy on the probabilistic labels estimated from
           repeated responses: the share of pairs of the weak and the strong judge's recorded calls in which the weak
           call is at least as good), 5 epochs, the checkpoint with the lowest loss on the fold's validation items.
           Output: the router's probability on every item, (N,).
Query text: the item's prior-text parts (BaselineConfig.text_parts, "part: text" joined by blank lines), cut to ~512
tokens in total with every part kept (fit_fields), then truncated by the tokenizer. An answer is written as its class
name (Dataset.classes; the class index when unnamed). Checkpoints are read from a local directory or the Hugging Face
cache, never downloaded.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from .common import FoldData, NotApplicable

CHARS_PER_TOKEN = 4


def fit_fields(values: list[str], budget: int) -> list[str]:
    """Cut the fields to `budget` characters in total, shortest first: a field under its share keeps all of its text
    and leaves the rest to the others, so the last field survives the token limit instead of being truncated away."""
    out, left = list(values), budget
    for rank, k in enumerate(sorted(range(len(values)), key=lambda k: len(values[k]))):
        share = left // (len(values) - rank)
        out[k] = values[k][:share]
        left -= len(out[k])
    return out


def query_texts(fd: FoldData, parts: tuple[str, ...], max_length: int) -> list[str]:
    """(N,) the text of every item."""
    parts = [p for p in parts if p in fd.ds.prior_text and any(map(str.strip, fd.ds.prior_text[p].astype(str)))]
    if not parts:
        raise NotApplicable("the fine-tuned scorers read the items' text: the dataset has none of the prior-text parts "
                            "in BaselineConfig.text_parts")
    budget = CHARS_PER_TOKEN * max_length
    cols = [fd.ds.prior_text[p].astype(str) for p in parts]
    return ["\n\n".join(f"{p}: {v}" for p, v in zip(parts, fit_fields([c[i] for c in cols], budget)))
            for i in range(fd.ds.n_items)]


def checkpoint(spec: dict) -> Path:
    """The checkpoint directory: spec["model"], or model_id at revision in the Hugging Face cache."""
    if spec.get("model"):
        return Path(spec["model"])
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
    path = hub / f"models--{spec['model_id'].replace('/', '--')}" / "snapshots" / spec["revision"]
    if not (path / "config.json").is_file():
        raise NotApplicable(f"checkpoint missing: {path} (fetch {spec['model_id']}@{spec['revision']} in a CPU job, "
                            f"or set the scorer's `model` to a local directory)")
    return path


def _require():
    try:
        import transformers  # noqa: F401
    except ImportError as e:
        raise NotApplicable("the fine-tuned scorers need the `baselines` extra (transformers)") from e


def _logit(logits):
    """One logit of Pr[target = 1]: the head's output, or with two classes the difference that softmax turns into the
    probability of class 1 (so binary cross-entropy on it is the two-class cross-entropy)."""
    logits = logits.float()
    return logits[:, 1] - logits[:, 0] if logits.shape[-1] == 2 else logits[:, 0]


def train_model(spec: dict, device: str, x_train, y_train, x_hold, y_hold):
    """Fine-tune a sequence classifier on (text, target in [0, 1]) with cross-entropy; keep the epoch with the lowest
    held-out loss. Returns the model, its tokenizer and the per-epoch losses."""
    import torch
    from torch.nn import functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(int(spec["seed"]))
    path = checkpoint(spec)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=int(spec["num_labels"]),
                                                               local_files_only=True).to(device)
    epochs, batch, max_len = int(spec["epochs"]), int(spec["batch_size"]), int(spec["max_length"])
    steps = epochs * ((len(x_train) + batch - 1) // batch)
    warmup = int(spec["warmup_steps"]) if spec.get("warmup_steps") is not None else int(float(spec["warmup_ratio"]) * steps)
    decay = [n for n, _ in model.named_parameters() if not (n.endswith("bias") or "LayerNorm" in n)]   # as Trainer
    groups = [{"params": [q for n, q in model.named_parameters() if n in decay], "weight_decay": float(spec["weight_decay"])},
              {"params": [q for n, q in model.named_parameters() if n not in decay], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=float(spec["lr"]))
    sched = get_linear_schedule_with_warmup(opt, warmup, steps)
    rng = np.random.default_rng(int(spec["seed"]))
    y_train, y_hold = np.asarray(y_train, dtype=np.float32), np.asarray(y_hold, dtype=np.float32)
    autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda")  # noqa: E731

    def loss(texts, target):
        enc = tokenizer(list(texts), truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
        with autocast():
            logits = model(**enc).logits
        return F.binary_cross_entropy_with_logits(_logit(logits), torch.as_tensor(target, device=device))

    def held_out():
        model.eval()
        with torch.no_grad():
            total = sum(float(loss(x_hold[s:s + 64], y_hold[s:s + 64])) * len(x_hold[s:s + 64])
                        for s in range(0, len(x_hold), 64))
        return total / max(len(x_hold), 1)

    best, best_state, history = np.inf, None, []
    for epoch in range(epochs):
        model.train()
        order, total = rng.permutation(len(x_train)), 0.0
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            value = loss([x_train[k] for k in idx], y_train[idx])
            opt.zero_grad(set_to_none=True)
            value.backward()
            opt.step()
            sched.step()
            total += float(value.detach()) * len(idx)
        held = held_out()
        history.append({"epoch": epoch + 1, "train_loss": total / len(order), "held_out_loss": held})
        if held < best:
            best, best_state = held, {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, tokenizer, history


def predict(model, tokenizer, texts: list[str], max_len: int) -> np.ndarray:
    """Pr[target = 1] for every text."""
    import torch
    device = next(model.parameters()).device
    out = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for s in range(0, len(texts), 128):
            enc = tokenizer(texts[s:s + 128], truncation=True, max_length=max_len, padding=True,
                            return_tensors="pt").to(device)
            out.append(torch.sigmoid(_logit(model(**enc).logits)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def answer_words(fd: FoldData) -> list[str]:
    return [str(c) for c in (fd.ds.classes or range(fd.K))]


def frugalgpt_scores(fd: FoldData, spec: dict, device: str, parts) -> tuple[np.ndarray, dict]:
    """S (N, M, K): each judge's scorer's probability that answering class v on item i is correct (NaN on items
    outside the fold's training, validation and test items)."""
    _require()
    started = time.time()
    max_len = int(spec["max_length"])
    texts, words = query_texts(fd, parts, max_len - 16), answer_words(fd)               # room for the answer
    need = np.unique(np.concatenate([fd.train, fd.val, fd.test]))
    scores = np.full((fd.ds.n_items, fd.M, fd.K), np.nan, dtype=np.float32)
    report = {}
    for j, judge in enumerate(fd.judges):
        rec = fd.recorded(fd.train)[:, j]                                                    # (n, W)
        has = rec.any(axis=1)
        items, first = fd.train[has], np.argmax(rec[has], axis=1)                           # one response per query
        votes = fd.answers[items, j, first]
        x = [f"{texts[i]} {words[v]}" for i, v in zip(items, votes)]
        y = fd.quality(items)[np.arange(len(items)), j, first]
        order = np.random.default_rng(int(spec["seed"])).permutation(len(x))
        cut = int(round((1 - float(spec["holdout"])) * len(x)))
        tr, ho = order[:cut], order[cut:]
        model, tok, history = train_model(spec, device, [x[k] for k in tr], y[tr], [x[k] for k in ho], y[ho])
        for v in range(fd.K):
            scores[need, j, v] = predict(model, tok, [f"{texts[i]} {words[v]}" for i in need], max_len)
        report[judge] = {"examples": len(x), "train": int(len(tr)), "held_out": int(len(ho)),
                         "quality_mean": float(np.mean(y)), "history": history}
        del model
    return scores, {"judges": report, "seconds": time.time() - started}


def weak_suffices(fd: FoldData, items, weak: int, strong: int) -> np.ndarray:
    """HybridLLM's probabilistic label: Pr[q(weak) >= q(strong)] over every pair of the two judges' recorded calls."""
    q = fd.quality(items)
    qw, qs = q[:, weak], q[:, strong]                                                         # (n, W), NaN: no call
    pair = (qw[:, :, None] >= qs[:, None, :]).astype(float)
    ok = np.isfinite(qw)[:, :, None] & np.isfinite(qs)[:, None, :]
    return (pair * ok).sum(axis=(1, 2)) / np.maximum(ok.sum(axis=(1, 2)), 1)


def hybridllm_router(fd: FoldData, weak: int, strong: int, spec: dict, device: str, parts) -> tuple[np.ndarray, dict]:
    """(N,) the router's Pr[q(weak) >= q(strong)] on every item (NaN outside the fold's items)."""
    _require()
    started = time.time()
    max_len = int(spec["max_length"])
    texts = query_texts(fd, parts, max_len)
    ok = fd.recorded(fd.train)[:, [weak, strong]].any(axis=2).all(axis=1)
    train = fd.train[ok]
    y_train, y_val = weak_suffices(fd, train, weak, strong), weak_suffices(fd, fd.val, weak, strong)
    model, tok, history = train_model(spec, device, [texts[i] for i in train], y_train, [texts[i] for i in fd.val], y_val)
    need = np.unique(np.concatenate([fd.val, fd.test]))
    prob = np.full(fd.ds.n_items, np.nan)
    prob[need] = predict(model, tok, [texts[i] for i in need], max_len)
    return prob, {"history": history, "train_label_mean": float(y_train.mean()), "val_label_mean": float(y_val.mean()),
                  "seconds": time.time() - started}
