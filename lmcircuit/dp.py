"""Exact backward induction over every state of a pool, for all cost prices at once (K classes).

A state is a digit string over the M pool judges in base A + 1: digit A = not asked, digits 0..A-1 = the value the
judge answered (A = the K classes, or the judges' own alphabet). In a state the circuit either stops and predicts one
of the K classes, or pays lambda * c_j to ask an unasked judge j, whose answer follows the belief model's predictive
distribution in that state.

Stopping risk with class weights w (lmcircuit.objectives): predicting y' risks
sum_{y != y'} w_y p_y, so the circuit predicts argmax_y w_y p_y and its loss is sum_y w_y p_y - max_y w_y p_y.
Values are negative expected losses; a query must beat the current best by `tol` to replace it (stopping wins ties,
the lower judge index wins among queries).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class _Action:
    judge: int
    local: torch.Tensor      # positions within the layer where `judge` is unasked
    rows: torch.Tensor       # those states
    children: torch.Tensor   # (K, n) successor state for each answer


class StateSpace:
    """The (A+1)^M states of M judges answering one of A values, in layers from all-asked down to none."""

    def __init__(self, n_judges: int, n_answers: int = 2, device="cpu"):
        M, K = int(n_judges), int(n_answers)
        self.M, self.A, self.device = M, K, torch.device(device)
        self.powers = (K + 1) ** np.arange(M, dtype=np.int64)
        self.n_states = int((K + 1) ** M)
        self.root = self.n_states - 1                                   # every digit K: nothing asked
        codes = np.arange(self.n_states)
        digits = (codes[:, None] // self.powers[None, :]) % (K + 1)
        self.digits_np = digits
        self.digits = torch.as_tensor(digits, dtype=torch.long, device=self.device)
        asked = (digits != K).sum(axis=1)
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long, device=self.device)  # noqa: E731
        self.layers = []
        for depth in range(M, -1, -1):
            states = np.flatnonzero(asked == depth)
            actions = []
            for j in range(M):
                local = np.flatnonzero(digits[states, j] == K)
                if len(local):
                    rows = states[local]
                    children = np.stack([rows + (v - K) * self.powers[j] for v in range(K)])
                    actions.append(_Action(j, t(local), t(rows), t(children)))
            self.layers.append((depth, t(states), actions))

    def state_of(self, answers: np.ndarray, asked: np.ndarray) -> np.ndarray:
        """State codes of (n, M) answers where `asked` (n, M) is True."""
        d = np.where(asked, answers, self.A)
        return (d * self.powers[None, :]).sum(axis=1)


def stop_loss(post: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """post (..., K) -> the smallest weighted risk of a prediction (...)."""
    wp = post * weights
    return wp.sum(dim=-1) - wp.max(dim=-1).values


def solve(space: StateSpace, post, nu, penalties, weights, tol: float = 1e-12):
    """post (B, S, K) = P(Y | state); nu (B, S, M, A) = P(judge answers | state); penalties (L, M) or (B, L, M) =
    lambda * cost (inf: never ask); weights (K,) or (G, K): G objectives solved at once, as G blocks of the L prices.
    Returns values (B, G L, S) float64 and actions (B, G L, S) int8 (-1 = stop); column g L + l: weights g, price l."""
    dev = space.device
    post = torch.as_tensor(post, dtype=torch.float64, device=dev)
    nu = torch.as_tensor(nu, dtype=torch.float64, device=dev)
    pen = torch.as_tensor(penalties, dtype=torch.float64, device=dev)
    w = torch.as_tensor(np.asarray(weights, dtype=np.float64), device=dev).reshape(-1, post.shape[-1])
    B = post.shape[0]
    if pen.dim() == 2:
        pen = pen.expand(B, -1, -1)
    G, L0 = len(w), pen.shape[1]
    pen, L = pen.repeat(1, G, 1), G * L0
    stop = (-stop_loss(post[:, None], w[None, :, None, :])).repeat_interleave(L0, dim=1)   # (B, G L, S)
    V = torch.empty((B, L, space.n_states), dtype=torch.float64, device=dev)
    A = torch.full((B, L, space.n_states), -1, dtype=torch.int8, device=dev)
    for _, states, actions in space.layers:
        best = stop[:, :, states].clone()
        choice = torch.full(best.shape, -1, dtype=torch.int8, device=dev)
        for act in actions:
            value = sum(nu[:, act.rows, act.judge, v][:, None, :] * V[:, :, act.children[v]] for v in range(space.A))
            value = value - pen[:, :, act.judge, None]
            current = best[:, :, act.local]
            better = value > current + tol
            best[:, :, act.local] = torch.where(better, value, current)
            choice[:, :, act.local] = torch.where(better, torch.tensor(act.judge, dtype=torch.int8, device=dev),
                                                  choice[:, :, act.local])
        V[:, :, states] = best
        A[:, :, states] = choice
    return V, A


def replay(space: StateSpace, actions, post, answers, weights):
    """Follow each policy against recorded answers. actions (B, L, S); post (B, S, K); answers (B, M) the value each
    judge gives in this world. Returns asked (B, L, M) bool, prediction (B, L) and the final belief (B, L, K)."""
    actions, post, answers = (np.asarray(a) for a in (actions, post, answers))
    B, L, _ = actions.shape
    state = np.full((B, L), space.root, dtype=np.int64)
    asked = np.zeros((B, L, space.M), dtype=bool)
    b, l_ = np.meshgrid(np.arange(B), np.arange(L), indexing="ij")
    for _ in range(space.M):
        a = actions[b, l_, state].astype(np.int64)
        moving = a >= 0
        if not moving.any():
            break
        bm, lm, am = b[moving], l_[moving], a[moving]
        asked[bm, lm, am] = True
        state[bm, lm] += (answers[bm, am] - space.A) * space.powers[am]
    final = post[b, state]                                              # (B, L, K)
    prediction = np.argmax(final * np.asarray(weights)[None, None, :], axis=-1)
    return asked, prediction, final


def lambda_grid(count: int, lo: float, hi: float) -> np.ndarray:
    """0 (ask freely), `count` geometric values in [lo, hi], and inf (never ask: the prior alone)."""
    return np.concatenate(([0.0], np.geomspace(lo, hi, count), [np.inf]))


def penalties(lambdas, cost) -> np.ndarray:
    """lambda * cost: cost (M,) -> (L, M); per-item cost (B, M) -> (B, L, M)."""
    lambdas, cost = np.asarray(lambdas, dtype=np.float64), np.asarray(cost, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        pen = lambdas[:, None] * cost[..., None, :]
    pen[..., np.isinf(lambdas), :] = np.inf
    return pen
