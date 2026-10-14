"""RouteLLM (Ong et al., ICLR 2025, "RouteLLM: Learning to Route LLMs with Preference Data"; lm-sys/RouteLLM), the
matrix-factorisation router.

A pairwise, single-call router: each item goes to the strong or the weak model, never both, decided up front. The
model is MFModel_Train of routellm/routers/matrix_factorization/train_matrix_factorization.py, copied: a learned
embedding per judge, a projection of the item embedding, and a Bradley-Terry loss (BCEWithLogits on (win-embedding
minus loss-embedding) x projected item). It is trained jointly over every pool judge on the training items' pairwise
comparisons of correctness (judge i beats judge j on an item when i is right and j wrong; ties dropped, as in their
`filtered_data`); with several recorded calls the target is the share of worlds in which i beat j among those where
they differed, weighted by that count. Adam (lr 3e-4, weight decay 1e-5), 30 epochs, batch 4096, dimension 16, prompt
noise alpha 0.05 (their training script). Routing is theirs: strong when sigmoid(score_strong - score_weak) >= tau,
tau swept over linspace(0, 1, 200).

The pair: RouteLLM routes between a strong and a weak model chosen in advance. Its own pair comes from the router's
training: weak = the cheapest pool judge, strong = the judge with the highest mean learned score over the training
items (the highest-scoring costlier judge when the cheapest scores highest). With pairs "all" (the default) the same
router also routes every valid (cheap, expensive) pair (FoldData.valid_pairs), each over the same thresholds, and
validation picks the pair at each budget as it picks the threshold. No validation data in the fit.

Substitutions (made in our experiments): the router is trained from scratch on this fold's training items
instead of the released routellm/mf_gpt4_augmented checkpoint (Chatbot Arena, a different domain); the item
embedding is the dataset's prior features instead of OpenAI text-embedding-3-small (the projection maps from that
space); with pairs "all", every valid (cheap, expensive) pair is one more set of operating points of the one
method.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .common import FoldData, Point, Result

SUBSTITUTIONS = [
    "mf router trained from scratch on the fold's training items' pairwise correctness comparisons instead of the "
    "released routellm/mf_gpt4_augmented checkpoint",
    "item embeddings: the dataset's prior features instead of OpenAI text-embedding-3-small",
]


def _build_model(dim: int, num_models: int, text_dim: int):
    import torch
    from torch import nn
    from torch.nn import functional as F

    # ---- copied from routellm/routers/matrix_factorization/train_matrix_factorization.py, with the frozen npy-lookup
    # Q embedding replaced by a plain input tensor and text_dim a constructor argument (was 1536 for OpenAI
    # embeddings); the bilinear architecture and BT loss are unchanged.
    class MFModel_Train(nn.Module):
        def __init__(self, dim, num_models, text_dim, num_classes=1, use_proj=True):
            super().__init__()
            self.use_proj = use_proj
            self.P = nn.Embedding(num_models, dim)
            if self.use_proj:
                self.text_proj = nn.Linear(text_dim, dim, bias=False)
            else:
                assert text_dim == dim
            self.classifier = nn.Linear(dim, num_classes, bias=False)  # bias should be False!

        def forward(self, model_win, model_loss, prompt_embed, test=False, alpha=0.05):
            model_win_embed = F.normalize(self.P(model_win), p=2, dim=1)
            model_loss_embed = F.normalize(self.P(model_loss), p=2, dim=1)
            if not test:
                prompt_embed = prompt_embed + torch.randn_like(prompt_embed) * alpha
            if self.use_proj:
                prompt_embed = self.text_proj(prompt_embed)
            return self.classifier((model_win_embed - model_loss_embed) * prompt_embed).squeeze(-1)

        @torch.no_grad()
        def model_scores(self, model_ids, prompt_embed):
            # score(model, prompt) such that sigmoid(score_a - score_b) is the predicted win rate of a over b,
            # matching MFModel.pred_win_rate in routers/matrix_factorization/model.py
            m_embed = F.normalize(self.P(model_ids), p=2, dim=1)
            p_embed = self.text_proj(prompt_embed) if self.use_proj else prompt_embed
            return self.classifier(m_embed * p_embed.unsqueeze(1)).squeeze(-1)
    # ---- end copied block ----
    return MFModel_Train(dim=dim, num_models=num_models, text_dim=text_dim, use_proj=True)


def training_rows(fd: FoldData):
    """(first, second, item, target, weight): non-tied pairwise comparisons on the training items' recorded calls."""
    rows = [[], [], [], [], []]
    u, ok = fd.utility(fd.train), fd.recorded(fd.train)
    for a, b in combinations(range(fd.M), 2):
        differ = (u[:, a] != u[:, b]) & ok[:, a] & ok[:, b]
        count = differ.sum(axis=1)
        keep = np.flatnonzero(count > 0)
        wins = ((u[:, a] == 1) & differ).sum(axis=1)
        for r, v in zip(rows, ([a] * len(keep), [b] * len(keep), fd.train[keep], wins[keep] / count[keep], count[keep])):
            r.extend(np.asarray(v).tolist())
    return tuple(np.asarray(v, dtype=t) for v, t in zip(rows, (np.int64, np.int64, np.int64, np.float32, np.float32)))


def run(fd: FoldData, p: dict, cfg) -> Result:
    import torch
    from torch.nn import functional as F
    torch.manual_seed(fd.seed)
    np.random.seed(fd.seed)
    with fd.stage("fit"):
        first, second, item, target, weight = training_rows(fd)
        if not len(first):
            raise ValueError("RouteLLM received no non-tied training comparisons")
        features = torch.from_numpy(np.asarray(fd.features, dtype=np.float32))
        tensors = tuple(torch.from_numpy(v) for v in (first, second, item, target, weight))
        model = _build_model(int(p["dimension"]), fd.M, features.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
        batch_size, losses = int(p["batch_size"]), []
        for _ in range(int(p["epochs"])):
            order, total = torch.randperm(len(first)), 0.0
            for s in range(0, len(first), batch_size):
                a, b, i, y, w = (v[order[s:s + batch_size]] for v in tensors)
                loss_each = F.binary_cross_entropy_with_logits(model(a, b, features[i]), y, reduction="none")
                loss = (loss_each * w).sum() / w.sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float((loss_each * w).sum().detach())
            losses.append(total / float(weight.sum()))
        model.eval()
        with torch.no_grad():
            train_scores = model.model_scores(torch.arange(fd.M), features[torch.as_tensor(fd.train)]).numpy()
        mean_score = train_scores.mean(axis=0)
        ranked = np.argsort(-mean_score)
        weak, strong = 0, int(ranked[0]) if ranked[0] != 0 else int(ranked[1])                # judge 0 is the cheapest
        pairs = fd.pairs((weak, strong), p.get("pairs", "main"))
    points = []
    with fd.stage("decide"):
        with torch.no_grad():
            scores = model.model_scores(torch.arange(fd.M), features).numpy()                   # (N, M)
        for w, s in pairs:
            p_strong = 1 / (1 + np.exp(-(scores[:, s] - scores[:, w])))
            for tau in np.linspace(0.0, 1.0, int(p["thresholds"])):
                out = {name: fd.single(items, np.where(p_strong[items] >= tau, s, w))
                       for name, items in (("val", fd.val), ("test", fd.test))}
                points.append(Point({"pair": [fd.judges[w], fd.judges[s]], "tau": float(tau),
                                     "strong_route_rate": float((p_strong[fd.test] >= tau).mean())},
                                    out["val"], out["test"]))
    return Result(points, {"reference": "Ong et al., RouteLLM (lm-sys/RouteLLM) matrix-factorization router MFModel_Train",
                           "substitutions": SUBSTITUTIONS, "pair": [fd.judges[weak], fd.judges[strong]],
                           "pairs": [[fd.judges[w], fd.judges[s]] for w, s in pairs],
                           "learned_mean_score": dict(zip(fd.judges, map(float, mean_score))),
                           "training_rows": int(len(first)), "comparisons": int(weight.sum()), "loss": losses})
