"""PolitiCAUSE (Garcia Corral et al., 2024), as packaged by polsci-open-bench (github.com/hhilbig/polsci-open-bench,
pinned commit): does a sentence from political text state a causal relation?

    lc.datasets.fetch("politicause", raw_dir); ds = lc.datasets.load("politicause", raw_dir)    # the sentences
    ds = lc.collect(ds, lc.datasets.politicause.recipe({"Qwen3-8B": url, ...}), repeats=8)      # the judges' calls

The judges' calls are not distributed: `load` gives the items alone, and `recipe` reruns the seven judges of the
paper with lc.collect. Calls recorded earlier in <raw>/recorded (format: `_recorded`) are joined by id instead.

Licence: polsci-open-bench's code is MIT; its README says the data in data/ "is subject to each upstream source's
license" and names none for PolitiCAUSE. The upstream licence of the sentences is unverified here.

Raw layout:
    source/politicause_causal_relation.csv   the pinned CSV (fetch; sha256 checked): source_id, text,
                                             gt_causal_relation, source_split, source_index
    recorded/calls.npz, calls.json           optional: recorded calls

Items: the 17,767 sentences of the CSV but the four used as the prompt's demonstrations (CONFIG["demonstrations"]);
id = source_id; label 1 when gt_causal_relation is "1". task "all"; group = the item (the source has no grouping
unit).
Judges: SHARED_SEVEN (`_judges`) under the paper's prompt (`_prompts`: the benchmark's instruction verbatim with four
training demonstrations; JSON answer {"causal_relation": 0|1}). Answers 1 = "causal relation".
Prior text: `item` the sentence (the judges read the sentence itself, so the prior sees the whole judged object);
`text` fills the prompt.
Cost: each call's USD from its prompt and generated tokens at the judge's price (lc.collect records it).
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..data import Dataset
from . import _recorded, _judges
from ._common import download, options, require

CONFIG = {
    "sources": {"benchmark": {
        "url": "https://raw.githubusercontent.com/hhilbig/polsci-open-bench/76cb3c6493266e99ff8910659adaf99887e626e7/"
               "data/politicause_causal_relation.csv",
        "revision": "76cb3c6493266e99ff8910659adaf99887e626e7",
        "sha256": "0d5ddf5b29149715a9ee43c1590c320bbbc5811349ac17b86923c7dca78a812b",
        "licence": "polsci-open-bench code MIT; PolitiCAUSE data: upstream licence (unverified)"}},
    "demonstrations": ("politicause_train_678", "politicause_train_11179", "politicause_train_1658",
                       "politicause_train_11832"),   # in the prompt (prompts/politicause_examples.json): never items
    "classes": ["no causal relation", "causal relation"],
    "description": ("Causal relation detection: decide whether a sentence from political text states or clearly "
                    "implies that one event, condition, decision, action or process causes, enables, prevents or "
                    "otherwise leads to another. Answer 1 if it contains a causal relation, 0 otherwise."),
    "judges": _judges.SHARED_SEVEN,
    "prompt": {"name": "politicause", "fields": ["text"], "classes": ["0", "1"]},
}
SOURCE = "source/politicause_causal_relation.csv"


def fetch(raw_dir) -> Path:
    """The pinned CSV (sha256 checked; a present file is kept and checked)."""
    raw, src = Path(raw_dir), CONFIG["sources"]["benchmark"]
    download(src["url"], raw / SOURCE, src["sha256"])
    return raw


def prompt():
    p = CONFIG["prompt"]
    return _judges.prompt(p["name"], p["fields"], p["classes"])


def recipe(base_urls: dict, served: dict | None = None):
    """lc.JudgeSpecs that rerun the seven judges (one vLLM server per model, keyed by judge name); see
    `_judges.recipe`. The paper made 8 calls per item: lc.collect(ds, specs, repeats=8)."""
    return _judges.recipe(CONFIG["judges"], prompt(), base_urls, served)


def load(raw_dir, **overrides) -> Dataset:
    cfg = options(CONFIG, overrides)
    raw = Path(raw_dir)
    recorded = _recorded.read(raw)
    with require(raw / SOURCE, "politicause").open(encoding="utf-8", newline="") as f:
        rows = {r["source_id"]: r for r in csv.DictReader(f) if r["source_id"] not in cfg["demonstrations"]}
    if recorded is not None and (demos := set(cfg["demonstrations"]) & set(recorded.ids)):
        raise ValueError(f"politicause: demonstrations among the recorded items: {sorted(demos)}")
    items, labels = [], []
    for _, r in _recorded.pick(recorded, rows, "politicause"):
        items.append({"id": r["source_id"], "task": "all", "task_description": cfg["description"],
                      "group": r["source_id"], "item": r["text"], "text": r["text"]})
        labels.append({"0": 0, "1": 1}[r["gt_causal_relation"]])
    src = cfg["sources"]["benchmark"]
    return _recorded.dataset(recorded, items, cfg["judges"], cfg["classes"], labels,
                             ("task_description", "item", "text"),
                             {"source": f"{src['url']} (sha256 {src['sha256']})", "licence": src["licence"]})
