"""The belief model: the BeliefNet within latent item types (docs/method.md, section 3).

For an item with features x (projected to h = text(x)) and an unobserved type t in 1..T:
    pi_t(x)          the type prior, a two-layer head on h
    q_jt(x, o)       judge j's answer distribution in type t when its call presents the item in order o (pairwise
                     tasks; one order otherwise), a two-layer head on h
    eta_t(x, s)      the label distribution in type t at the state s (the answers seen): each answer seen is a token
                     g(e_judge + e_answer [+ e_order] + e_type), the tokens are summed (the order of the answers does not
                     matter; an unasked judge adds nothing), and [h + e_type, the sum, the share of judges asked] goes
                     through a two-layer trunk to the class logits
At a state s:
    w_t(s) proportional to pi_t(x) prod_{(j, a, o) in s} q_jt(x, o)[a]       the type posterior
    P(Y | x, s) = sum_t w_t(s) eta_t(x, s)        P(X_j = a | x, s, o) = sum_t w_t(s) q_jt(x, o)[a]
so an answer counts differently, and an unasked judge's predicted answer moves with the answers seen, through the type.
With one type, the judges' predicted answers depend on the text alone.

Pairwise tasks (Dataset.features_swap, the features with the two candidates exchanged): the exchange reverses the
classes and the presentation orders. Both outputs average the log-probabilities of the two views mapped back, so the
model is exactly equivariant to the exchange.

Training, per batch of training items: the label's cross-entropy at one sampled state per item (a number of judges
uniform on 0..M, which of the judges with a recorded call uniformly, one recorded call of each), plus alpha times the
cross-entropy of every unasked judge's predicted answer against all its recorded answers on the item, at another
sampled state. AdamW with a cosine schedule; the epoch kept has the lowest label loss on fixed validation states
(early stopping). The number of types is the one, among cfg.types, with the lowest validation loss (label plus alpha
times judges, at those states). Calibration: one temperature for the label and one for the judges' answers, each
fitted on validation states (10 per item) by the likelihood of the labels and of the unasked judges' answers.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from .config import BeliefConfig


class BeliefNet(nn.Module):
    def __init__(self, d_features: int, n_judges: int, n_classes: int, n_orders: int = 1, swap: bool = False, *,
                 n_answers: int | None = None, types: int = 4, d_model: int = 64, d_hidden: int = 128):
        super().__init__()
        M, K, O, T = int(n_judges), int(n_classes), int(n_orders), int(types)
        A = K if n_answers is None else int(n_answers)              # the judges' answer values (the classes by default)
        self.M, self.K, self.A, self.O, self.T, self.swap = M, K, A, O, T, bool(swap)
        self.kw = dict(d_features=int(d_features), n_judges=M, n_classes=K, n_answers=A, n_orders=O, swap=self.swap,
                       types=T, d_model=int(d_model), d_hidden=int(d_hidden))
        self.text = nn.Linear(int(d_features), d_model)
        self.type_head = nn.Sequential(nn.Linear(d_model, d_hidden), nn.ReLU(), nn.Linear(d_hidden, T))
        self.judge_head = nn.Sequential(nn.Linear(d_model, d_hidden), nn.ReLU(), nn.Linear(d_hidden, T * M * O * A))
        self.judge_emb = nn.Embedding(M, d_model)
        self.answer_emb = nn.Embedding(A, d_model)
        self.order_emb = nn.Embedding(O, d_model) if O > 1 else None
        self.type_emb = nn.Embedding(T, d_model)
        self.token_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.trunk = nn.Sequential(nn.Linear(2 * d_model + 1, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_hidden),
                                   nn.ReLU())
        self.label_head = nn.Linear(d_hidden, K)
        self.register_buffer("temperature", torch.ones(2))         # label, judges (calibrate)
        self.register_buffer("class_perm", torch.arange(K - 1, -1, -1), persistent=False)
        self.register_buffer("answer_perm", torch.arange(A - 1, -1, -1), persistent=False)
        self.register_buffer("order_perm", torch.arange(O - 1, -1, -1), persistent=False)

    # ------------------------------------------------------------------ pieces
    def _parts(self, x):
        """h (B, d), log pi (B, T), log q (B, T, M, O, A)."""
        h = self.text(x)
        return (h, torch.log_softmax(self.type_head(h), dim=-1),
                torch.log_softmax(self.judge_head(h).view(-1, self.T, self.M, self.O, self.A), dim=-1))

    def _view(self, swapped: bool):
        """How a view reads the original classes, answers and orders: index maps (K,), (A,) and (O,)."""
        dev = self.class_perm.device
        if swapped:
            return self.class_perm, self.answer_perm, self.order_perm
        return torch.arange(self.K, device=dev), torch.arange(self.A, device=dev), torch.arange(self.O, device=dev)

    def _at(self, x, answer, order, observed, swapped: bool):
        """One view at given states (answer, order, observed: (B, M) per judge): log P(Y) (B, K) and log P(X_j = a | o)
        (B, M, O, A), indexed in the original classes, answers and orders."""
        cls, ans, omap = self._view(swapped)
        h, log_pi, log_q = self._parts(x)
        a, o = ans[answer], omap[order]
        B, ar = len(x), torch.arange(len(x), device=x.device)
        e = self.judge_emb.weight[None] + self.answer_emb(a)                                        # (B, M, d)
        if self.order_emb is not None:
            e = e + self.order_emb(o)
        mask = observed.to(h.dtype)
        tok = self.token_mlp(e[:, None] + self.type_emb.weight[None, :, None])                       # (B, T, M, d)
        pooled = (tok * mask[:, None, :, None]).sum(dim=2)
        share = (mask.sum(dim=1) / self.M)[:, None, None].expand(B, self.T, 1)
        log_eta = torch.log_softmax(self.label_head(self.trunk(
            torch.cat([h[:, None] + self.type_emb.weight[None], pooled, share], dim=-1))), dim=-1)    # (B, T, K)
        judge = torch.arange(self.M, device=x.device)
        lq = log_q[ar[:, None, None], torch.arange(self.T, device=x.device)[None, :, None], judge[None, None, :],
                   o[:, None, :]]                                                                   # (B, T, M, A)
        seen = (lq.gather(-1, a[:, None, :, None].expand(-1, self.T, -1, 1))[..., 0] * mask[:, None]).sum(dim=-1)
        lw = torch.log_softmax(log_pi + seen, dim=1)                                                # (B, T)
        log_y = torch.logsumexp(lw[..., None] + log_eta, dim=1)
        log_x = torch.logsumexp(lw[:, :, None, None, None] + log_q, dim=1)                          # (B, M, O, A)
        return log_y[:, cls], log_x[:, :, omap][..., ans]

    def beliefs(self, x, x_swap, answer, order, observed, calibrated: bool = True):
        """At given states: log P(Y | x, s) (B, K) and every judge's log P(X_j = a | x, s, o) (B, M, O, A)."""
        log_y, log_x = self._at(x, answer, order, observed, False)
        if self.swap:
            sy, sx = self._at(x_swap, answer, order, observed, True)
            log_y, log_x = 0.5 * (log_y + sy), 0.5 * (log_x + sx)
        t = self.temperature if calibrated else torch.ones_like(self.temperature)
        return torch.log_softmax(log_y / t[0], dim=-1), torch.log_softmax(log_x / t[1], dim=-1)

    # ------------------------------------------------------------------ every DP state at once
    def _view_tables(self, x, orders, digits, swapped: bool, max_cells: float):
        """One view at every state (digits (S, M), digit A = not asked; orders (B, M) the order each judge's call
        presents): log P(Y) (B, S, K) and log P(X_j) at its own order (B, S, M, A), in the original indices. The
        trunk's first layer is linear, so its input splits into a text part per item and type and a token part per
        (judge, answer), added up per state."""
        cls, ans, omap = self._view(swapped)
        h, log_pi, log_q = self._parts(x)
        B, S, M, A, T = len(x), digits.shape[0], self.M, self.A, self.T
        first, act1, second, act2 = self.trunk
        D = self.kw["d_model"]
        w_text, w_pool, w_share = first.weight[:, :D], first.weight[:, D:2 * D], first.weight[:, 2 * D]
        dev, judge = x.device, torch.arange(M, device=x.device)
        o = omap[orders]                                                                            # (B, M)
        e = self.judge_emb.weight[None, :, None] + self.answer_emb(ans)[None, None]                 # (1, M, A, d)
        if self.order_emb is not None:
            e = e + self.order_emb(o)[:, :, None]                                                   # (B, M, A, d)
        tok = self.token_mlp(e[:, None] + self.type_emb.weight[None, :, None, None]) @ w_pool.T     # (1|B, T, M, A, h)
        tok = torch.cat([tok, tok.new_zeros(*tok.shape[:3], 1, tok.shape[-1])], dim=3)
        text_part = (h[:, None] + self.type_emb.weight[None]) @ w_text.T + first.bias                # (B, T, h)
        lq = log_q[torch.arange(B, device=dev)[:, None, None], torch.arange(T, device=dev)[None, :, None],
                   judge[None, None, :], o[:, None, :]][..., ans]                                   # (B, T, M, A)
        lq_seen = torch.cat([lq, lq.new_zeros(*lq.shape[:3], 1)], dim=3)
        chunk = max(1, int(max_cells) // max(1, B * T * first.out_features))
        log_y, log_x = [], []
        for start in range(0, S, chunk):
            d = digits[start:start + chunk]                                                         # (s, M)
            share = (d != A).to(x.dtype).sum(dim=1) / M
            pre = text_part[:, :, None] + share[None, None, :, None] * w_share + sum(tok[:, :, m, d[:, m]] for m in range(M))
            log_eta = torch.log_softmax(self.label_head(act2(second(act1(pre)))), dim=-1)          # (B, T, s, K)
            lw = log_pi[:, :, None] + sum(lq_seen[:, :, m, d[:, m]] for m in range(M))               # (B, T, s)
            lw = torch.log_softmax(lw, dim=1)
            log_y.append(torch.logsumexp(lw[..., None] + log_eta, dim=1)[..., cls])                 # (B, s, K)
            log_x.append(torch.logsumexp(lw[:, :, :, None, None] + lq[:, :, None], dim=1))           # (B, s, M, A)
        return torch.cat(log_y, dim=1), torch.cat(log_x, dim=1)

    @torch.no_grad()
    def tables(self, x, x_swap, orders, space, max_cells: float = 2 ** 26):
        """At every state of `space`, for B items whose calls present in `orders` (B, M): P(Y | state) (B, S, K) and
        P(X_j | state) (B, S, M, A), calibrated."""
        log_y, log_x = self._view_tables(x, orders, space.digits, False, max_cells)
        if self.swap:
            sy, sx = self._view_tables(x_swap, orders, space.digits, True, max_cells)
            log_y, log_x = 0.5 * (log_y + sy), 0.5 * (log_x + sx)
        return torch.softmax(log_y / self.temperature[0], dim=-1), torch.softmax(log_x / self.temperature[1], dim=-1)


# ---------------------------------------------------------------------------------------------- training
class _Data:
    """A dataset's arrays on the training device: x, x_swap (N, d); answers, orders (N, M, R) with MISSING < 0; labels
    (N, K) annotation counts."""

    def __init__(self, data: dict, n_orders: int, n_answers: int, device):
        self.answers, self.orders = np.asarray(data["answers"]), np.asarray(data["orders"])
        self.valid = self.answers >= 0
        N, M, R = self.answers.shape
        counts = np.zeros((N, M, n_orders, n_answers), dtype=np.float32)       # every recorded answer, per order
        i, j, r = np.nonzero(self.valid)
        np.add.at(counts, (i, j, self.orders[i, j, r], self.answers[i, j, r]), 1.0)
        t = lambda a: None if a is None else torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
        self.x, self.x_swap, self.labels, self.counts = t(data["x"]), t(data.get("x_swap")), t(data["labels"]), t(counts)
        self.device = device

    def states(self, items, rng):
        """One sampled state per item: a number of judges uniform on 0..M (capped by the judges with a call), which
        of them uniformly, one recorded call of each. Returns tensors src, answer, order, observed."""
        items = np.asarray(items, dtype=np.int64)
        valid = self.valid[items]
        n, M, R = valid.shape
        eligible = valid.any(axis=2)
        k = np.minimum(rng.integers(0, M + 1, size=n), eligible.sum(axis=1))
        rank = np.argsort(np.argsort(-np.where(eligible, rng.random((n, M)), -1.0), axis=1), axis=1)
        observed = rank < k[:, None]
        call = np.where(valid, rng.random((n, M, R)), -1.0).argmax(axis=2)
        at = (items[:, None], np.arange(M)[None, :], call)
        t = lambda a: torch.as_tensor(a, device=self.device)  # noqa: E731
        return (t(items), t(np.where(observed, self.answers[at], 0)), t(np.where(observed, self.orders[at], 0)),
                t(observed))

    def beliefs(self, net, state, calibrated=False, chunk: int = 8192):
        src, answer, order, observed = state
        out = [net.beliefs(self.x[src[s:s + chunk]], None if self.x_swap is None else self.x_swap[src[s:s + chunk]],
                           answer[s:s + chunk], order[s:s + chunk], observed[s:s + chunk], calibrated)
               for s in range(0, len(src), chunk)]
        return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])

    def targets(self, state):
        """Label counts (B, K) and the unasked judges' recorded answers per order (B, M, O, A)."""
        src, _, _, observed = state
        return self.labels[src], self.counts[src] * (~observed)[:, :, None, None]


def _label_loss(log_y, labels):
    """Cross-entropy against annotation counts, every annotation counted once."""
    return -(labels * log_y).sum() / labels.sum().clamp_min(1e-12)


def _judge_loss(log_x, counts):
    """Cross-entropy against the unasked judges' recorded answers, each item with any weighing one."""
    per = counts.flatten(1).sum(dim=1)
    loss = -(counts * log_x).flatten(1).sum(dim=1)
    active = per > 0
    return (loss[active] / per[active]).mean() if active.any() else log_x.sum() * 0.0


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def fit_once(d: _Data, train_idx, val_state, types: int, cfg: BeliefConfig, seed: int, model_kw: dict):
    """One network with `types` types; the epoch with the lowest validation label loss."""
    device = d.device
    devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
    _sync(device)
    started = time.perf_counter()
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed + 1)
        net = BeliefNet(**{**model_kw, "types": types}).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        val_labels, val_counts = d.targets(val_state)
        best, best_state, best_epoch, stale, history = (np.inf, np.inf), None, -1, 0, []
        for epoch in range(cfg.epochs):
            net.train()
            order = rng.permutation(train_idx)
            for s in range(0, len(order), cfg.batch_size):
                batch = order[s:s + cfg.batch_size]
                at_y, at_x = d.states(batch, rng), d.states(batch, rng)
                log_y, _ = d.beliefs(net, at_y)
                _, log_x = d.beliefs(net, at_x)
                loss = _label_loss(log_y, d.targets(at_y)[0]) + cfg.alpha * _judge_loss(log_x, d.targets(at_x)[1])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
            net.eval()
            with torch.no_grad():
                log_y, log_x = d.beliefs(net, val_state)
                v = (float(_label_loss(log_y, val_labels)), float(_judge_loss(log_x, val_counts)))
            history.append(v)
            if v[0] < best[0] - 1e-5:
                best, best_epoch, stale = v, epoch, 0
                best_state = {k: p.detach().clone() for k, p in net.state_dict().items()}
            elif (stale := stale + 1) >= cfg.patience:
                break
        net.load_state_dict(best_state)
    _sync(device)
    return net.eval(), {"types": types, "best_epoch": best_epoch, "epochs_run": len(history),
                        "val_label_loss": best[0], "val_judge_loss": best[1],
                        "val_loss": best[0] + cfg.alpha * best[1], "seconds": time.perf_counter() - started}


def _temperature(nll, lo: float = 0.05, hi: float = 20.0, steps: int = 60) -> float:
    """The temperature in [lo, hi] minimising nll(t) (golden-section search on log t); 1 unless it lowers nll."""
    g = (np.sqrt(5) - 1) / 2
    a, b = np.log(lo), np.log(hi)
    c, e = b - g * (b - a), a + g * (b - a)
    fc, fe = nll(np.exp(c)), nll(np.exp(e))
    for _ in range(steps):
        if fc < fe:
            b, e, fe = e, c, fc
            c = b - g * (b - a)
            fc = nll(np.exp(c))
        else:
            a, c, fc = c, e, fe
            e = a + g * (b - a)
            fe = nll(np.exp(e))
    t = float(np.exp((a + b) / 2))
    return t if nll(t) < nll(1.0) else 1.0


@torch.no_grad()
def calibrate(net: BeliefNet, d: _Data, val_idx, seed: int, per_item: int = 10) -> dict:
    """The label and judge temperatures on `per_item` sampled states of every validation item."""
    state = d.states(np.repeat(np.asarray(val_idx), per_item), np.random.default_rng(seed + 20_000))
    log_y, log_x = d.beliefs(net, state, calibrated=False)
    labels, counts = d.targets(state)
    ny = lambda t: float(_label_loss(torch.log_softmax(log_y / t, dim=-1), labels))  # noqa: E731
    nx = lambda t: float(_judge_loss(torch.log_softmax(log_x / t, dim=-1), counts))  # noqa: E731
    t = (_temperature(ny), _temperature(nx))
    net.temperature.copy_(torch.as_tensor(t, dtype=net.temperature.dtype))
    return {"label": t[0], "judges": t[1], "label_nll": [ny(1.0), ny(t[0])], "judges_nll": [nx(1.0), nx(t[1])]}


def fit(data: dict, train_idx, val_idx, model_kw: dict, cfg: BeliefConfig, device, seed: int = 0):
    """The network, among one per number of types in cfg.types, with the lowest validation loss, calibrated. data: x
    (N, d) features, x_swap (N, d) or None, answers and orders (N, M, R), labels (N, K); one row per item. Returns the
    network and a record (every fit's validation losses and seconds, the temperatures, the seconds of each stage)."""
    d = _Data(data, model_kw["n_orders"], model_kw.get("n_answers") or data["labels"].shape[1], device)
    val_state = d.states(val_idx, np.random.default_rng(seed + 10_000))
    trials = {t: fit_once(d, train_idx, val_state, t, cfg, seed, model_kw) for t in cfg.types}
    chosen = min(trials, key=lambda t: trials[t][1]["val_loss"])
    net = trials[chosen][0]
    _sync(device)
    started = time.perf_counter()
    temps = calibrate(net, d, val_idx, seed, cfg.calibration_states)
    _sync(device)
    return net, {"types": chosen, "fits": {str(t): r for t, (_, r) in trials.items()}, "temperature": temps,
                 "seconds": {"fits": float(sum(r["seconds"] for _, r in trials.values())),
                             "calibration": time.perf_counter() - started}}
