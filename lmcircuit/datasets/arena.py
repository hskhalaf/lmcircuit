"""LMArena human preferences (lmarena-ai/arena-human-preference-140k, CC BY 4.0): which of two AI assistants' responses
to a prompt did the human prefer? The single-turn English battles of the public source.

    lc.datasets.fetch("arena", raw_dir); ds = lc.datasets.load("arena", raw_dir)    # the battles
    ds = lc.collect(ds, lc.datasets.arena.recipe({"Qwen3-8B": url, ...}), repeats=8)   # the judges' calls

The judges' calls are not distributed: `load` gives the items alone, and `recipe` reruns the seven judges of the
paper with lc.collect. Calls recorded earlier in <raw>/recorded (format: `_recorded`; `orders` holds each call's
presented order) are joined by id instead.

Raw layout:
    source/data/train-0000k-of-00007.parquet   the pinned battles (fetch, 1.6 GB; sha256 checked)
    recorded/calls.npz, calls.json            optional: recorded calls

Items: battles that are single-turn, English, text-only, evaluation_order 1, with winner model_a or model_b (the paper
drew 35,000 of them with balanced winners and code strata, and kept the 34,952 whose calls all gave A>B or B>A).
id = the battle id; a message's text blocks joined by newlines; label 0 when the human preferred response A, 1 for B
(the source's order, canonical). task "code" or "general" (the source's is_code); group = sha256 of the prompt.
Judges: SHARED_SEVEN (`_judges`) under the paper's prompt (`_prompts`: forced choice [[A>B]] or [[B>A]], with
MT-Bench's length sentence). The prompt has permute=("response_a", "response_b"): lc.collect shows the responses in
an order drawn per (judge, item, call) and maps each answer back to the canonical order.
Prior text: `candidate_a` and `candidate_b` hold each response with the prompt ("User prompt:\\n<prompt>\\n\\nAssistant
response:\\n<response>"), from which lc.embed builds pair features with the exact A/B swap (PRIOR_LAYOUT); `item` =
the prompt and both responses in canonical order (one text). Both hold the judged responses, so a prior on them acts
as a free judge. `prompt`, `response_a`, `response_b` fill the judge prompt.
Cost: each call's USD from its prompt and generated tokens at the judge's price (lc.collect records it).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..data import Dataset
from . import _recorded, _judges
from ._common import hf_file, options, require, verify

_SHARDS = ("7b871a964dc9b238fe0d59f817dbe1a7a56934e23ec4d989f86e6891e5a891cf",
           "0cd29a6e3261159f41a69b28cd7f08c0bbbc1ef08ba18438d386a1b0c144af8a",
           "c633264fe8f0fa26cc9aa49961a75dc428791f70c7a476cbed766963b88bee24",
           "5bcf04c9f4488006a4a7ee944d886ee123e118b7f7d2db04b4ff6dbd8125e6b0",
           "2c50b4b3f6a5c02d660f5a70d5006332ddc00f5eb21f8e4d92bde3856647969b",
           "de59f74dce7477beaec55936b31a88537b93c5cbb4b483cfc6879a6862ce5e3b",
           "1a32a1c6934ebcaa3019643a08fb57563ee0a9be282d3222def3c572a49e9f4c")
CONFIG = {
    "sources": {"benchmark": {
        "repo": "lmarena-ai/arena-human-preference-140k", "revision": "6322995ab34d7c2693e3f47dd13fa5caa0789a74",
        "licence": "CC BY 4.0",
        "files": {f"data/train-{k:05d}-of-00007.parquet": sha for k, sha in enumerate(_SHARDS)}}},
    "classes": ["A", "B"],          # the response the human preferred, in the source's order
    "tasks": {
        "code": ("Human preference on a coding prompt: given a user's prompt about programming and two AI "
                 "assistants' responses, predict which response the human user preferred, A or B."),
        "general": ("Human preference on a general (non-coding) prompt: given a user's prompt and two AI assistants' "
                    "responses, predict which response the human user preferred, A or B."),
    },
    "judges": _judges.SHARED_SEVEN,
    "prompt": {"name": "arena", "fields": ["prompt", "response_a", "response_b"], "classes": ["A>B", "B>A"],
               "permute": ("response_a", "response_b"),
               "rename": {"presented_response_a": "response_a", "presented_response_b": "response_b"}},
}
# lc.embed: each candidate embedded with the prompt, pair features with the exact A/B swap (the paper's layout:
# lengths of the prompt and of each response, is_code as the task)
PRIOR_LAYOUT = {"kind": "pair", "a": "candidate_a", "b": "candidate_b", "length_a": "response_a",
                "length_b": "response_b", "lengths": ["prompt"]}
COLUMNS = ["id", "winner", "evaluation_order", "language", "is_code", "conversation_a", "conversation_b"]


def fetch(raw_dir) -> Path:
    """The seven pinned parquet shards and the dataset card (sha256 checked; present files are kept)."""
    raw, src = Path(raw_dir), CONFIG["sources"]["benchmark"]
    for name, sha in src["files"].items():
        verify(hf_file(src["repo"], name, src["revision"], raw / "source" / name), sha)
    hf_file(src["repo"], "README.md", src["revision"], raw / "source" / "README.md")
    return raw


def prompt():
    p = CONFIG["prompt"]
    return _judges.prompt(p["name"], p["fields"], p["classes"], p["permute"], p["rename"])


def recipe(base_urls: dict, served: dict | None = None):
    """lc.JudgeSpecs that rerun the seven judges (one vLLM server per model, keyed by judge name; the paper served
    max_model_len 32768); see `_judges.recipe`. The paper made 8 calls per item: lc.collect(ds, specs, repeats=8)."""
    return _judges.recipe(CONFIG["judges"], prompt(), base_urls, served)


def _text(message: dict) -> str | None:
    """A message's text: a string, or its text blocks joined by newlines (None when it holds anything else)."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None or not len(content):
        return None
    parts = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text" or block.get("image") is not None \
                or not isinstance(block.get("text"), str):
            return None
        parts.append(block["text"])
    return "\n".join(parts)


def _battle(row: dict) -> dict | None:
    """The single-turn reading of a battle (its texts and winner), else None."""
    a, b = list(row["conversation_a"] if row["conversation_a"] is not None else []), \
        list(row["conversation_b"] if row["conversation_b"] is not None else [])
    if row["winner"] not in ("model_a", "model_b") or len(a) != 2 or len(b) != 2:
        return None
    if [m.get("role") for m in a] != ["user", "assistant"] or [m.get("role") for m in b] != ["user", "assistant"]:
        return None
    texts = [_text(a[0]), _text(b[0]), _text(a[1]), _text(b[1])]
    if not all(isinstance(t, str) and t.strip() for t in texts) or texts[0] != texts[1]:
        return None
    return {"prompt": texts[0], "response_a": texts[2], "response_b": texts[3], "is_code": bool(row["is_code"]),
            "winner": 0 if row["winner"] == "model_a" else 1}


def _rows(raw: Path, files, wanted: set | None) -> dict:
    """id -> battle for the wanted ids, or every battle the filters keep (streamed in record batches: only the wanted
    rows are converted)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    out, keep = {}, None if wanted is None else pa.array(sorted(wanted))
    for name in files:
        for batch in pq.ParquetFile(require(raw / "source" / name, "arena")).iter_batches(4096, columns=COLUMNS):
            rows = batch if keep is None else batch.filter(pc.is_in(batch.column("id"), value_set=keep))
            for row in rows.to_pylist():
                if str(row["language"]).strip().lower() not in ("en", "english") or row["evaluation_order"] != 1:
                    continue
                if (battle := _battle(row)) is not None:
                    out.setdefault(str(row["id"]), battle)
    return out


def load(raw_dir, **overrides) -> Dataset:
    cfg = options(CONFIG, overrides)
    raw = Path(raw_dir)
    recorded = _recorded.read(raw)
    rows = _rows(raw, cfg["sources"]["benchmark"]["files"], None if recorded is None else set(recorded.ids))
    items, labels = [], []
    for i, r in _recorded.pick(recorded, rows, "arena"):
        task = "code" if r["is_code"] else "general"
        candidate = lambda resp: f"User prompt:\n{r['prompt']}\n\nAssistant response:\n{resp}"   # noqa: E731
        items.append({"id": i, "task": task, "task_description": cfg["tasks"][task],
                      "group": hashlib.sha256(r["prompt"].encode()).hexdigest(),
                      "item": (f"User prompt:\n{r['prompt']}\n\nResponse A:\n{r['response_a']}\n\n"
                               f"Response B:\n{r['response_b']}"),
                      "candidate_a": candidate(r["response_a"]),
                      "candidate_b": candidate(r["response_b"]), "prompt": r["prompt"],
                      "response_a": r["response_a"], "response_b": r["response_b"]})
        labels.append(r["winner"])
    src = cfg["sources"]["benchmark"]
    return _recorded.dataset(recorded, items, cfg["judges"], cfg["classes"], labels,
                             ("task_description", "item", "candidate_a", "candidate_b", "prompt",
                              "response_a", "response_b"),
                             {"source": f"{src['repo']}@{src['revision']}", "licence": src["licence"],
                              "prior_layout": PRIOR_LAYOUT})
