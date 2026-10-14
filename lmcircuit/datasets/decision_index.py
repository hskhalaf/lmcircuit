"""Decision Index entrants on six binary task families: each entrant run's recorded choice on every two-option request
of the suite, no inference (leaderboard huggingface.co/spaces/multimodalart/jev-decision-index; suite kit
github.com/apolinario/decision-index, MIT). Reproduces the dataset used in the LMCircuit paper.

    lc.datasets.fetch("decision_index", raw_dir); ds = lc.datasets.load("decision_index", raw_dir)

Raw layout (written by `fetch`):
    suite/rows.jsonl.gz                                one line per request of the families: run_id, payload hash,
                                                       state, questions, expected (gold), group id
    entrants/<repo, / as __>__<run>/results.jsonl.gz   one run's result lines on those families only

The suite is not redistributed (several of its sources forbid it), so `fetch` rebuilds its rows from the pinned public
sources with the official kit at its pinned revision: the same normalizers and freeze step, run for these families
only. A rebuilt row is the request the entrants answered when its payload_sha256 equals the one in their result line;
`load` checks this for every answer. HoVer needs the kit's 2.2 GB Wikipedia database (downloaded into `work_dir`).

Items: every request of CONFIG["tasks"] that is one two-option choice question and that every judge answered with a
valid choice on the payload the suite rebuilds; id = the suite's run_id, ordered by task, then id; group = the suite's
group_id (BPoMP's perturbations of one limerick share one); the label is the index of the gold option. Prior text:
`item` is the request as the engine saw it (its state, the question's instructions, the two options).
Judges: one entrant run each; the answer is its choice, one call per item.
Cost: the runs record no billing, so the compute proxy prices them: 0.01 x active parameters (billions) per million
tokens (a mixture of experts counts its per-token active experts), times the request's proxy_tokens (the suite's size
estimate, the same for every entrant). Nothing is generated, so the planning cost (known before the call) equals the
call's cost.
"""
from __future__ import annotations

import gzip
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..data import Dataset
from ._common import call_cost, git_checkout, make_judge, options, planning_cost, price, read_jsonl, require, retry

CONFIG = {
    "sources": {"kit": {"url": "https://github.com/apolinario/decision-index.git",
                        "revision": "87d4650b42b377c0291a89c1f1a879f9b31082bf"}},
    # Decision Index benchmarks; descriptions follow the leaderboard's own one-line explainers
    "tasks": {
        "NLI4CT": "NLI4CT (SemEval-2024 Task 2): given sections of one or two clinical-trial reports and a statement "
                  "about them, decide whether the trial evidence entails or contradicts the statement; some statements "
                  "were deliberately altered.",
        "CLadder": "CLadder: answer a yes/no question about cause and effect in a small made-up world whose causal "
                   "structure and probabilities are stated.",
        "BPoMP": "BPoMP: given two versions of a limerick, one of them perturbed so that its rhyme, meter or sense is "
                 "broken, pick the original.",
        "HoVer": "HoVer: decide whether a claim that links two to four Wikipedia articles is supported by the "
                 "introductions of those articles.",
        "iSarcasmEval-A": "iSarcasmEval (SemEval-2022 Task 6, subtask A): say whether a tweet, in English or Arabic, is "
                          "intended to be sarcastic.",
        "Humicroedit": "Humicroedit (SemEval-2020 Task 7, subtask 2): given two one-word edits of a news headline, pick "
                       "the one people rated funnier.",
    },
    # task -> (catalog id, the suite tracks kept). Every request of these is a two-option choice except iSarcasmEval
    # track B (six yes/no questions per tweet) and C, left out.
    "families": {"NLI4CT": (42, ["NLI4CT-2024"]), "CLadder": (44, ["CLadder"]),
                 "BPoMP": (20, ["BPoMP-original-limerick"]), "HoVer": (61, ["2-hop", "3-hop", "4-hop"]),
                 "iSarcasmEval-A": (40, ["iSarcasmEval-A-En", "iSarcasmEval-A-Ar"]), "Humicroedit": (21, ["Humicroedit"])},
    "classes": ["option 0", "option 1"],   # 1: the second of the request's two options, in the order it lists them
    # The roster: 18 entrant runs with every selected request answered, by served size; one run per base model and
    # vendor except the Decision 2.0 ladder (one recipe at five sizes). Each is runs/<name>/results.jsonl.gz of `repo`
    # at `revision`, under `licence` (research use permitted). params_b: served parameters (billions): the
    # leaderboard's served_params (space data/index.json) for the model or, for a run not on the board, its base
    # model's; else the Hub safetensors count. active_params_b: a mixture of experts' nominal per-token active count.
    # price: none listed, so the compute proxy at the active parameters. flag: training overlap with a family, as the entrant declares it (kept, marked).
    # Left out: dm-jepa (94 unsupported) and jevk5-lite-preview-1 (3,259 unsupported); the six zev-* runs (one
    # degenerate model, identical answers, near chance); dinah-0 (trained on CLadder with no stated suite
    # decontamination; a 150M encoder at 0.84 on CLadder); near duplicates on a shared base: onejev-0.8b,
    # jev-style-0.8b, Sieve-2B/4B/9B, RSI-Jev-v4.0/v5.0, ezjev-4b-s2, jev-9b, jevk5-9b, jev-27b, jebadiah-27b,
    # eikos-27b-fp8, blink-v0.3 (NVFP4 of the 26B-A4B base), exaone-jev v0.1/v0.2 (earlier versions of v0.3). Gated
    # repositories and those without a licence were never considered.
    "judges": {
        "lavoir": {
            "repo": "moganai/lavoir-decision-index-results", "revision": "2a9f764674436a1957f517312711bbb5878daff4",
            "licence": "apache-2.0", "model": "moganai/lavoir", "params_b": 0.395882,
            "note": "leaderboard served_params (lavoir)"},
        "decision-2.0-kai-0.6b": {
            "repo": "vllm-sr/decision-2.0-decision-index", "revision": "09ef6f55d7554c4f17a20b6914e51636cbf77a9e",
            "licence": "apache-2.0", "model": "vllm-sr/Decision-2.0-Kai-0.6B", "params_b": 0.59605,
            "note": "base Qwen/Qwen3-0.6B-Base, leaderboard served_params (bosun-v3.1-0.6b); adapter and head not "
                      "counted"},
        "decision-2.0-eos-0.8b": {
            "repo": "vllm-sr/decision-2.0-decision-index", "revision": "09ef6f55d7554c4f17a20b6914e51636cbf77a9e",
            "licence": "apache-2.0", "model": "vllm-sr/Decision-2.0-Eos-0.8B", "params_b": 0.873439,
            "note": "leaderboard served_params of Decision-1.0-Eos-0.8B (same Qwen3.5-0.8B-Base backbone)"},
        "kodiak-v0.2-1b-accuracy": {
            "repo": "cortex-agent-llc/decision-index-results", "revision": "7cd5f3b1e51e09ed649639fc58ffa9938540b340",
            "licence": "apache-2.0", "model": "cortex-agent-llc/kodiak-v0.2-1b", "params_b": 1.04412,
            "note": "Hub safetensors total"},
        "exaone-jev-1.2b-v0.3": {
            "repo": "carrtesy/decision-index-results", "revision": "a5381aecd8d8c4f68f6bacf5f3884b1615acd0de",
            "licence": "exaone (EXAONE AI model licence: research, non-commercial)",
            "model": "carrtesy/EXAONE-4.0-1.2B-JEV", "params_b": 1.279391, "note": "Hub safetensors total"},
        "jev-style-2b-decision-v3": {
            "repo": "chaoliangUNSW/decision-index-results", "revision": "1f13ebf66b90f1654f2fc1acfec532be410e9542",
            "licence": "apache-2.0", "model": "chaoliangUNSW/Jev-Style-2B-Decision-v3", "params_b": 1.881825,
            "note": "Hub safetensors total"},
        "decision-2.0-sol-2b": {
            "repo": "vllm-sr/decision-2.0-decision-index", "revision": "09ef6f55d7554c4f17a20b6914e51636cbf77a9e",
            "licence": "apache-2.0", "model": "vllm-sr/Decision-2.0-Sol-2B", "params_b": 2.27407,
            "note": "leaderboard served_params of Decision-1.0-Sol (same Qwen3.5-2B-Base backbone)"},
        "jevk5-v0.3": {
            "repo": "alibiserikbay/decision-index-results", "revision": "4164adcfc2b61f38b260db6d040e59195d6d1960",
            "licence": "apache-2.0", "model": "alibiserikbay/JevK5", "params_b": 4.659865,
            "note": "leaderboard served_params (jevk5)"},
        "decision-2.0-nox-4b": {
            "repo": "vllm-sr/decision-2.0-decision-index", "revision": "09ef6f55d7554c4f17a20b6914e51636cbf77a9e",
            "licence": "apache-2.0", "model": "vllm-sr/Decision-2.0-Nox-4B", "params_b": 4.659865,
            "note": "leaderboard served_params of Decision-1.0-Nox (same Qwen3.5-4B-Base backbone)"},
        "decision-2.0-lux-9b": {
            "repo": "vllm-sr/decision-2.0-decision-index", "revision": "09ef6f55d7554c4f17a20b6914e51636cbf77a9e",
            "licence": "apache-2.0", "model": "vllm-sr/Decision-2.0-Lux-9B", "params_b": 9.653104,
            "note": "leaderboard served_params of Decision-1.0-Lux-9B (same Qwen3.5-9B-Base backbone)"},
        "bespoke-nimble-9b-v3": {
            "repo": "bespokelabs/bespoke-nimble-9b-v3-decision-index",
            "revision": "654b124d3508a7da0d849405744bde80c4bb1a1b", "licence": "cc-by-nc-4.0",
            "model": "bespokelabs/Bespoke-Nimble-9B-v3", "params_b": 9.653104,
            "note": "leaderboard served_params of Bespoke-Nimble-9B-v2 (same Qwen3.5-9B backbone, LoRA)"},
        "xor-26b-a4b": {
            "repo": "juspay/xor-26b-a4b-decision-index", "revision": "28740419da6769b14662b648208ee970d584d6dd",
            "licence": "apache-2.0", "model": "juspay/Xor-26B-A4B", "params_b": 25.805936, "active_params_b": 4.0,
            "note": "Hub safetensors total (MoE, ~4B active)"},
        "jev-gemma4-26b-a4b": {
            "repo": "autotrust/jev-decision-index-results", "revision": "6c4044b2128772a30620adb4c75da615755f0f8d",
            "licence": "apache-2.0", "model": "autotrust/JEV-Gemma4-26B-A4B", "params_b": 25.805936, "active_params_b": 4.0,
            "note": "Hub safetensors total (MoE, ~4B active)",
            "flag": "trained on HoVer training-split rows (suite items excluded before training; 16,155 suite HoVer "
                    "requests share its fixed question template, none share a state); HoVer accuracy is "
                    "in-distribution"},
        "torchcast-decision-27b": {
            "repo": "torchcast-ai/torchcast-decision-27b-decision-index",
            "revision": "9396dcb2d5250aa4d18a321182e378f9a77dc958", "licence": "cc-by-nc-4.0",
            "model": "torchcast-ai/torchcast-decision-27b", "params_b": 27.356729, "note": "Hub safetensors total",
            "flag": "training data not documented per family (unverified)"},
        "clef": {
            "repo": "lucataco/clef-decision-index-results", "revision": "a15784d858ee2d2173a43a8bdbf6749ff5b49f83",
            "licence": "apache-2.0", "model": "Cloudflare/clef", "params_b": 27.356729,
            "note": "Hub safetensors total",
            "flag": "CLadder 0.94 and BPoMP 0.97, far above same-size entrants; its card does not list per-family "
                    "training data (unverified)"},
        "decision-2.0-vega-27b": {
            "repo": "vllm-sr/decision-2.0-decision-index", "revision": "09ef6f55d7554c4f17a20b6914e51636cbf77a9e",
            "licence": "apache-2.0", "model": "vllm-sr/Decision-2.0-Vega-27B", "params_b": 27.781428,
            "note": "base Qwen/Qwen3.8-27B, leaderboard served_params (jebadiah-27b)"},
        "mogan-decision-31B-it": {
            "repo": "moganai/mogan-decision-31B-it-decision-index-results",
            "revision": "e751b050c5fbafd07f9b1a01d59f3debaf2df75e", "licence": "cc-by-nc-4.0",
            "model": "moganai/mogan-decision-31B-it", "params_b": 32.682373,
            "note": "base google/gemma-4-31B-it, leaderboard served_params (decider-chat-gemma4-31b)",
            "flag": "trained on public training splits of CLadder, HoVer, Humicroedit, iSarcasmEval and NLI4CT, "
                    "decontaminated against the suite; 5 HoVer requests still share a long evidence span "
                    "(dedup/residual.jsonl), no identical item"},
        "xor-1.2": {
            "repo": "juspay/xor-1.2-decision-index", "revision": "1df03933f536a94f05768c89c38e9048c2fb9fcd",
            "licence": "apache-2.0", "model": "juspay/xor", "params_b": 35.951823, "active_params_b": 3.0,
            "note": "leaderboard served_params (xor; MoE, ~3B active)"},
    },
}
CATALOG = re.compile(rb'"catalog_id":\s*(\d+)')


def _run_dir(raw: Path, name: str, spec: dict) -> Path:
    return raw / "entrants" / f"{spec['repo'].replace('/', '__')}__{name}"


# --------------------------------------------------------------------------------------------------------------- fetch
def fetch(raw_dir, judges=None, work_dir=None, kit_dir=None, suite: bool = True) -> Path:
    """Stream each judge's results file at its pinned revision, keeping the selected families' lines only, and
    rebuild the suite rows with the kit (cloned at its pinned revision into kit_dir, default raw_dir/kit; its
    sources and the HoVer database go to work_dir, default raw_dir/work). CPU and network only; files present are
    kept."""
    raw = Path(raw_dir)
    for name in judges or CONFIG["judges"]:
        spec = CONFIG["judges"][name]
        dest = _run_dir(raw, name, spec) / "results.jsonl.gz"
        if not dest.exists():
            retry(lambda: _stream_filter(spec["repo"], spec["revision"], f"runs/{name}/results.jsonl.gz", dest),
                  tries=4, wait=10.0)
    if suite and not (raw / "suite" / "rows.jsonl.gz").exists():
        kit = git_checkout(CONFIG["sources"]["kit"]["url"], CONFIG["sources"]["kit"]["revision"], kit_dir or raw / "kit")
        _rebuild_suite(kit, Path(work_dir or raw / "work"), raw / "suite" / "rows.jsonl.gz")
    return raw


def _stream_filter(repo: str, revision: str, path: str, dest: Path) -> None:
    """Results files are large (every family); stream one and keep only the selected families' lines and fields."""
    from huggingface_hub import get_session, hf_hub_url

    families = {c for c, _ in CONFIG["families"].values()}
    url = hf_hub_url(repo, path, repo_type="dataset", revision=revision)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with get_session().get(url, stream=True, timeout=600) as r, gzip.open(tmp, "wb") as out:
        r.raise_for_status()
        r.raw.decode_content = True
        for line in io.BufferedReader(gzip.GzipFile(fileobj=r.raw), buffer_size=1 << 20):
            m = CATALOG.search(line)
            if not m or int(m.group(1)) not in families:
                continue
            row = json.loads(line)
            keep = {k: row.get(k) for k in ("run_id", "catalog_id", "track", "group_id", "payload_sha256",
                                           "proxy_tokens", "status", "model_request_wall_ms", "total_wall_ms")}
            resp = row.get("response") or {}
            keep.update(answers=resp.get("answers"), usage=resp.get("usage"))
            out.write((json.dumps(keep) + "\n").encode())
    tmp.replace(dest)


def _rebuild_suite(kit: Path, work: Path, out: Path) -> None:
    """The kit's acquisition, normalizers and freeze for the selected families only (HoVer is an added family with its
    own adapter); rows the suite excludes are dropped."""
    sys.path.insert(0, str(kit))
    from decision_index.suite.build import acquire as acq
    from decision_index.suite.build import adapters_added, adapters_selection, freeze, normalize_text
    from decision_index.suite.build.layout import Layout, dump_ascii

    class NoTokens:                 # HoVer counts tiktoken tokens for proxy_tokens; the payload hash does not need them
        def encode(self, text, disallowed_special=()):
            return []

    layout = Layout(work)
    acq.acquire(layout, [20, 21, 40, 42, 44])
    normalize_text.isarcasm(layout)
    adapters_selection.bpomp(layout)
    normalize_text.humor(layout)
    normalize_text.cladder(layout)
    with (layout.normalized / "NLI4CT-2024.jsonl").open("w", encoding="utf-8") as f:   # nli_documents() also needs
        for row in normalize_text.clinical(layout):                                    # ContractNLI
            f.write(dump_ascii(row) + "\n")
    for stale in layout.normalized.glob("*.jsonl"):                                    # a partial run's leftovers
        if stale.stem not in {"NLI4CT-2024", "CLadder", "Humicroedit", "BPoMP-original-limerick"} and \
                not stale.stem.startswith("iSarcasmEval-"):
            stale.unlink()
    frozen = work / "frozen"
    freeze.freeze(layout, frozen)
    adapters_added.acquire(layout, [61])
    excluded = set(json.loads((kit / "hub" / "excluded-questions.json").read_text())["rows"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as dst:
        def emit(row, e):
            if e["run_id"] not in excluded:
                dst.write(json.dumps({
                    "run_id": e["run_id"], "catalog_id": e["catalog_id"], "dataset": e["dataset"], "track": e["track"],
                    "group_id": e["group_id"], "payload_sha256": e["payload_sha256"], "id": row["id"],
                    "family": row.get("family"), "state": row["state"], "questions": row["questions"],
                    "expected": row["expected"], "metadata": row.get("metadata") or row.get("provenance") or {}},
                    ensure_ascii=False) + "\n")
        with (frozen / "selected-rows.jsonl").open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                emit(row, row["_evaluation"])
        for row in adapters_added.hover(layout):
            emit(row, adapters_added.evaluation(61, row, NoTokens()))


# ---------------------------------------------------------------------------------------------------------------- load
def _render(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(_render(v) for v in value)
    lines = []
    for key, v in value.items():
        text = _render(v)
        lines.append(f"{key}: {text}" if "\n" not in text else f"{key}:\n{text}")
    return "\n".join(lines)


def _request_text(row: dict, question: dict) -> str:
    state = _render(row["state"])
    choices = "\n".join(f"{k}: {_render(v)}" for k, v in question["criteria"].items())
    head = f"{state}\n\n" if state else ""
    return f"{head}{_render(question['instructions'])}\nOptions:\n{choices}"


def _requests(raw: Path, cfg: dict) -> dict:
    """run_id -> request, for every single-question two-option request of the tasks' tracks with a gold option."""
    by_track = {(cfg["families"][t][0], tr): t for t in cfg["tasks"] for tr in cfg["families"][t][1]}
    out = {}
    for row in read_jsonl(require(raw / "suite" / "rows.jsonl.gz", "decision_index")):
        task = by_track.get((row["catalog_id"], row["track"]))
        if task is None or len(row["questions"]) != 1:
            continue
        (qkey, question), = row["questions"].items()
        keys = list(question["criteria"])
        if len(keys) != 2 or row["expected"][qkey] not in keys:
            continue
        out[row["run_id"]] = {"id": row["run_id"], "task": task, "task_description": cfg["tasks"][task],
                              "group": row["group_id"], "item": _request_text(row, question), "question": qkey,
                              "keys": keys, "gt": keys.index(row["expected"][qkey]),
                              "payload_sha256": row["payload_sha256"]}
    return out


def _answers(path: Path, requests: dict) -> tuple[dict, dict]:
    """run_id -> (choice index, proxy_tokens) for the run's valid answers on the payload the suite rebuilds; counts of
    the lines left out and why."""
    out, bad = {}, defaultdict(int)
    for r in read_jsonl(require(path, "decision_index")):
        req = requests.get(r["run_id"])
        if req is None:
            continue
        if r["payload_sha256"] != req["payload_sha256"]:
            bad["payload_mismatch"] += 1
            continue
        ans = (r.get("answers") or {}).get(req["question"])
        if r.get("status") != "ok" or not ans or ans.get("choice") not in req["keys"]:
            bad[f"status_{r.get('status')}" if r.get("status") != "ok" else "no_valid_choice"] += 1
            continue
        out[r["run_id"]] = (req["keys"].index(ans["choice"]), r.get("proxy_tokens"))
    return out, dict(bad)



def load(raw_dir, **overrides) -> Dataset:
    """Build the dataset from the raw files; overrides replace CONFIG keys (e.g. judges={...} for another roster)."""
    cfg = options(CONFIG, overrides)
    raw, spec = Path(raw_dir), cfg["judges"]
    requests = _requests(raw, cfg)
    names = list(spec)
    got, bad = {}, {}
    for name in names:
        got[name], bad[name] = _answers(_run_dir(raw, name, spec[name]) / "results.jsonl.gz", requests)
    order = {t: n for n, t in enumerate(cfg["tasks"])}
    ids = sorted((i for i in requests if all(i in got[j] for j in names)),
                 key=lambda i: (order[requests[i]["task"]], i))
    answers = np.asarray([[got[j][i][0] for j in names] for i in ids], dtype=np.int64)
    tokens = np.asarray([[got[j][i][1] for j in names] for i in ids], dtype=np.float64)
    if not (tokens == tokens[:, :1]).all():
        raise ValueError("entrants disagree on proxy_tokens for some request")
    p = np.asarray([price(spec[j])[0] for j in names])
    items = [requests[i] for i in ids]
    return Dataset.from_records(
        items, [make_judge(j, spec[j], lines_left_out=bad[j]) for j in names], answers, call_cost(tokens, 0.0, p, p),
        n_classes=2, classes=cfg["classes"], labels=[requests[i]["gt"] for i in ids],
        prior_parts=("task_description", "item"), plan_cost=planning_cost(tokens, p, p, 0.0),
        meta={"source": "Decision Index entrant runs (public per-request results)",
              "kit": "{url}@{revision}".format(**cfg["sources"]["kit"]),
              "cost": "0.01 x active parameters (billions) per million proxy tokens"})
