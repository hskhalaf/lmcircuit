"""HealthBench meta-evaluation (openai/healthbench, MIT; Arora et al., 2025): does the final assistant response of a
health conversation meet a rubric criterion, as physicians judge it? The meta-evaluation rows of the public source,
with every physician judgment kept as a label count.

    lc.datasets.fetch("healthbench", raw_dir); ds = lc.datasets.load("healthbench", raw_dir)     # the rows
    ds = lc.collect(ds, lc.datasets.healthbench.recipe({"gpt-oss-20b": url, ...}), repeats=8)   # the judges' calls

The judges' calls are not distributed: `load` gives the items alone, and `recipe` reruns the eight judge
configurations of the paper with lc.collect. Calls recorded earlier in <raw>/recorded (format: `_recorded`) are
joined by id instead.

Raw layout:
    source/2025-05-07-06-14-12_oss_meta_eval.jsonl   the pinned meta-evaluation file (fetch; sha256 checked)
    recorded/calls.npz, calls.json                   optional: recorded calls

Items: the 29,511 rows (the paper used the 29,131 whose calls all parsed); id hb_<line>, line = the 0-based line of
the file. Labels: the physicians' judgments (binary_labels, 2-5 per row) as counts [not met, met]; no majority is
taken. task = the HealthBench theme of the row's rubric cluster (category "cluster:<theme>_..."; seven themes,
CONFIG["tasks"]); group = the row's prompt_id (every completion and criterion of one conversation together).
Judges: HEALTH_EIGHT (`_judges`): granite-4.2-3b at three reasoning levels, gpt-oss-20b at three, gpt-oss-120b at
two, under the paper's prompt (`_prompts`: simple-evals' HealthBench grader template verbatim, one user message; JSON
answer with "criteria_met": true|false). Answers 1 = criteria met.
Prior text: `rubric` ("Criterion:\\n<rubric item>") and `context` ("Conversation:\\n<conversation>", the completion
included), from which lc.embed builds [criterion, context, criterion x context] (meta prior_layout "criterion");
`item` = criterion and conversation in one text. `context` and `item` hold the judged completion, so a prior on them
acts as a free judge. `conversation` (rendered as simple-evals does: "role: content" blocks, the completion as the
last assistant turn) and `rubric_item` fill the judge prompt.
Cost: each call's USD from its prompt and generated tokens at the judge's price (lc.collect records it).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data import Dataset
from . import _recorded, _judges
from ._common import hf_file, options, require, verify

_TASK = ("HealthBench meta-evaluation: decide whether a physician would judge that the final assistant response in a "
         "health conversation meets a rubric criterion (true if met, false if not). Theme: ")
CONFIG = {
    "sources": {"benchmark": {
        "repo": "openai/healthbench", "revision": "40ee1968852fc57f625934251ac22be47077a8fb", "licence": "MIT",
        "file": ("2025-05-07-06-14-12_oss_meta_eval.jsonl",
                 "6a343ed3fcb6f28353c2c6c157ae30ddf2db874c947b27ba474ad54da9c5dbac")}},
    "classes": ["not met", "met"],
    # HealthBench's seven themes (Arora et al., 2025), the prefix of each rubric cluster's name
    "tasks": {
        "emergency_referrals": _TASK + "emergency referrals: does the response recognise an emergency, or its "
                               "absence, and steer the user to the right level of care?",
        "communication": _TASK + "expertise-tailored communication: does the response suit a health professional or "
                         "a lay user, in accuracy, completeness and wording?",
        "hedging": _TASK + "responding under uncertainty: does the response hedge, seek context or answer outright as "
                   "the uncertainty of the question warrants?",
        "context_seeking": _TASK + "context seeking: does the response ask for missing information when it needs it, "
                           "and answer helpfully and safely when it has enough?",
        "global_health": _TASK + "global health: does the response fit the user's resource setting and regional "
                         "context, accurately?",
        "health_data_tasks": _TASK + "health data tasks: does the response complete a structured health-data task "
                             "(such as clinical documentation) accurately, safely and as instructed?",
        "complex_responses": _TASK + "response depth: are the response's detail and structure appropriate and "
                             "accurate for what the user needs?",
    },
    "judges": _judges.HEALTH_EIGHT,
    "prompt": {"name": "healthbench", "fields": ["conversation", "rubric_item"], "classes": ["false", "true"]},
}


def fetch(raw_dir) -> Path:
    """The pinned meta-evaluation file (136 MB; sha256 checked; a present file is kept)."""
    raw, src = Path(raw_dir), CONFIG["sources"]["benchmark"]
    name, sha = src["file"]
    verify(hf_file(src["repo"], name, src["revision"], raw / "source" / name), sha)
    return raw


def prompt():
    p = CONFIG["prompt"]
    return _judges.prompt(p["name"], p["fields"], p["classes"])


def recipe(base_urls: dict, served: dict | None = None):
    """lc.JudgeSpecs that rerun the eight configurations: one vLLM server per model ("granite-4.2-3b",
    "gpt-oss-20b", "gpt-oss-120b"; the paper served max_model_len 24576), each level a request field; see
    `_judges.recipe`. The paper made 8 calls per item: lc.collect(ds, specs, repeats=8)."""
    return _judges.recipe(CONFIG["judges"], prompt(), base_urls, served)


def render(messages: list[dict]) -> str:
    """simple-evals' rendering: one "role: content" block per message."""
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def theme(category: str, themes) -> str:
    """The theme a rubric cluster ("cluster:<theme>_<rest>") belongs to."""
    hits = [t for t in themes if category.startswith(f"cluster:{t}_")]
    if len(hits) != 1:
        raise ValueError(f"healthbench: category {category!r} matches themes {hits}")
    return hits[0]


def load(raw_dir, **overrides) -> Dataset:
    cfg = options(CONFIG, overrides)
    raw = Path(raw_dir)
    recorded = _recorded.read(raw)
    rows = {}
    with require(raw / "source" / cfg["sources"]["benchmark"]["file"][0], "healthbench").open(encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line.strip():
                rows[f"hb_{line_no:06d}"] = json.loads(line)
    items, labels = [], []
    for i, r in _recorded.pick(recorded, rows, "healthbench"):
        conversation = render([*r["prompt"], {"role": "assistant", "content": r["completion"]}])
        task = theme(r["category"], cfg["tasks"])
        items.append({"id": i, "task": task, "task_description": cfg["tasks"][task], "group": r["prompt_id"],
                      "rubric": f"Criterion:\n{r['rubric']}", "context": f"Conversation:\n{conversation}",
                      "item": f"Criterion:\n{r['rubric']}\n\nConversation:\n{conversation}",
                      "conversation": conversation, "rubric_item": r["rubric"]})
        met = [int(bool(v)) for v in r["binary_labels"]]
        labels.append([len(met) - sum(met), sum(met)])
    src = cfg["sources"]["benchmark"]
    return _recorded.dataset(recorded, items, cfg["judges"], cfg["classes"], np.asarray(labels, dtype=np.float64),
                             ("task_description", "item", "rubric", "context", "conversation",
                              "rubric_item"),
                             {"source": f"{src['repo']}@{src['revision']}/{src['file'][0]}", "licence": src["licence"],
                              "prior_layout": {"kind": "criterion", "criterion": "rubric", "context": "context"}})
