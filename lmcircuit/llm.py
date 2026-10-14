"""Calling language models for labels (lc.label) and judge answers (lc.collect).

One OpenAI-compatible client serves both a vLLM server (`Backend.vllm("http://host:8000/v1", model)`) and the
OpenAI API (`Backend.openai(model)`, key from OPENAI_API_KEY or ~/.config/openai/key). Requests send the chat
messages, `max_tokens` and a seed only: every other decoding setting stays at the model's default.
Calls run concurrently (bounded), are retried with backoff, and are cached in a local SQLite file keyed by a hash
of (base url, model, messages, max_tokens, seed), so rerunning a labelling or collection never pays twice.
With `logprobs` set, a call also reports each answer token's top alternatives (reported, not used to sample), from
which `verdict_probs` reads the probability of every class at the token where the verdict is written.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Backend:
    base_url: str
    model: str
    api_key: str = field(default="EMPTY", repr=False)
    price_in: float | None = None          # USD per million tokens (None: no USD cost recorded)
    price_out: float | None = None
    concurrency: int = 64
    timeout: float = 600.0
    extra_body: dict = field(default_factory=dict)   # e.g. {"reasoning_effort": "low"} when the user asks for it
    logprobs: int = 0                      # top alternatives per answer token to report (0: none); for verdict_probs

    @classmethod
    def vllm(cls, base_url: str, model: str, **kw) -> "Backend":
        return cls(base_url=base_url.rstrip("/"), model=model, **kw)

    @classmethod
    def openai(cls, model: str, **kw) -> "Backend":
        key = os.environ.get("OPENAI_API_KEY")
        path = Path.home() / ".config" / "openai" / "key"
        if not key and path.exists():
            key = path.read_text().strip()
        if not key:
            raise RuntimeError("no OpenAI key: set OPENAI_API_KEY or write it to ~/.config/openai/key (mode 0600)")
        return cls(base_url="https://api.openai.com/v1", model=model, api_key=key, concurrency=kw.pop("concurrency", 16), **kw)

    def cost(self, usage: dict) -> float | None:
        if self.price_in is None:
            return None
        return (usage.get("prompt_tokens", 0) * self.price_in + usage.get("completion_tokens", 0) * (self.price_out or 0)) / 1e6


class Cache:
    """(request hash) -> response json, in one SQLite file; one connection for the cache's lifetime."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, value TEXT)")
        self.db.commit()

    def get(self, key):
        row = self.db.execute("SELECT value FROM calls WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO calls VALUES (?, ?)", (key, json.dumps(value)))
        self.db.commit()

    def close(self):
        self.db.close()


def request_key(backend: Backend, messages, max_tokens, seed) -> str:
    blob = json.dumps([backend.base_url, backend.model, messages, max_tokens, seed, backend.extra_body]
                      + ([backend.logprobs] if backend.logprobs else []), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


async def _one(client, backend: Backend, sem, messages, max_tokens, seed, cache: Cache | None, retries: int = 5):
    key = request_key(backend, messages, max_tokens, seed)
    if cache is not None and (hit := cache.get(key)) is not None:
        return hit
    limit = "max_completion_tokens" if "api.openai.com" in backend.base_url else "max_tokens"   # OpenAI's newer name
    body = {"model": backend.model, "messages": messages, limit: max_tokens, "seed": seed, **backend.extra_body}
    if backend.logprobs:
        body.update(logprobs=True, top_logprobs=backend.logprobs)
    async with sem:
        for attempt in range(retries):
            try:
                r = await client.post(f"{backend.base_url}/chat/completions", json=body,
                                      headers={"Authorization": f"Bearer {backend.api_key}"})
                if r.status_code == 200:
                    j = r.json()
                    out = {"text": j["choices"][0]["message"].get("content") or "", "usage": j.get("usage", {}),
                           "finish_reason": j["choices"][0].get("finish_reason")}
                    if backend.logprobs:
                        out["logprobs"] = (j["choices"][0].get("logprobs") or {}).get("content") or []
                    if cache is not None:
                        cache.put(key, out)
                    return out
                if r.status_code not in (408, 409, 429, 500, 502, 503, 504):
                    return {"text": "", "usage": {}, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
            except Exception as e:                                    # network errors: retry
                err = repr(e)
            await asyncio.sleep(min(60, 2 ** attempt))
        return {"text": "", "usage": {}, "error": locals().get("err", "retries exhausted")}


def call_many(backend: Backend, conversations: list[list[dict]], max_tokens: int, seeds=None,
              cache_path=None) -> list[dict]:
    """Run every conversation through the backend; returns, in order, {"text", "usage", ...} (an "error" key when a
    call failed after retries)."""
    import httpx
    seeds = seeds if seeds is not None else [0] * len(conversations)
    cache = Cache(cache_path) if cache_path else None

    async def main():
        sem = asyncio.Semaphore(backend.concurrency)
        async with httpx.AsyncClient(timeout=backend.timeout) as client:
            return await asyncio.gather(*[_one(client, backend, sem, m, max_tokens, s, cache)
                                          for m, s in zip(conversations, seeds)])
    try:
        try:
            asyncio.get_running_loop()                    # inside a running loop (Jupyter): run in a worker thread
        except RuntimeError:
            return asyncio.run(main())
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            return ex.submit(asyncio.run, main()).result()
    finally:
        if cache is not None:
            cache.close()


# -------------------------------------------------------------------- prompts and parsers
@dataclass
class Prompt:
    """A chat prompt with {field} slots filled from an item (str.format: write literal braces as {{ }}); `classes` are
    the answer labels the parser looks for, in label order.

    `permute`: K fields shown in a per-call random order with the classes (a pairwise prompt: ("response_a",
    "response_b") with classes ["A", "B"]): presented position i holds field permute[order[i]], and presented class i
    means label order[i]. The order is a hash of (item id, judge, call), so it is reproducible."""
    user: str
    classes: list[str]
    system: str | None = None
    permute: tuple[str, ...] = ()

    def order(self, item_id: str, judge: str, call: int) -> list[int]:
        """The presented order of the labels for one call (identity without `permute`)."""
        K = len(self.classes)
        if not self.permute:
            return list(range(K))
        import numpy as np
        digest = hashlib.sha256(f"order:{judge}:{item_id}:{call}".encode()).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "big")).permutation(K).tolist()

    def messages(self, fields: dict, order: list[int] | None = None) -> list[dict]:
        if self.permute and order is not None:
            if len(self.permute) != len(self.classes):
                raise ValueError("permute needs one field per class")
            fields = {**fields, **{self.permute[i]: fields[self.permute[order[i]]] for i in range(len(order))}}
        out = [{"role": "system", "content": self.system}] if self.system else []
        return out + [{"role": "user", "content": self.user.format(**fields)}]


THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_choice(text: str, classes: list[str]) -> int:
    """The class named last in the response (after any reasoning block), as a whole word or a JSON value; -1 if none."""
    return locate_choice(text, classes)[0]


def locate_choice(text: str, classes: list[str]) -> tuple[int, int]:
    """(class, character offset in `text`) of the class named last after any reasoning block; (-1, -1) if none."""
    text = text or ""
    start = max((m.end() for m in THINK.finditer(text)), default=0)
    best, where = -1, -1
    for k, c in enumerate(classes):
        for m in re.finditer(rf"(?<![\w]){re.escape(c)}(?![\w])", text[start:], flags=re.I):
            if start + m.start() > where:
                best, where = k, start + m.start()
    return best, where


def verdict_probs(entries: list[dict], text: str, offset: int, classes: list[str]) -> list[float] | None:
    """Each class's probability at the token where the verdict starts (`offset` in `text`, the call's answer):
    every reported alternative at that position that spells the start of exactly one class counts toward it, and the
    result is normalized over the classes. None when the token cannot be located or no alternative names a class."""
    if not entries or offset < 0:
        return None
    cursor, index = 0, None
    for k, e in enumerate(entries):
        token = e.get("token") or ""
        if cursor <= offset < cursor + len(token):
            index = k
            break
        cursor += len(token)
    if index is None:
        return None
    before = (entries[index].get("token") or "")[:offset - cursor].lower()
    alternatives = {a["token"]: float(a["logprob"]) for a in entries[index].get("top_logprobs") or []}
    alternatives.setdefault(entries[index]["token"], float(entries[index]["logprob"]))
    probs = [0.0] * len(classes)
    for token, logprob in alternatives.items():
        token = token.lower()
        if not token.startswith(before) or not (rest := token[len(before):].strip()):
            continue
        hits = [k for k, c in enumerate(classes) if c.lower().startswith(rest) or rest.startswith(c.lower())]
        if len(hits) == 1:
            probs[hits[0]] += math.exp(logprob)
    total = sum(probs)
    return [p / total for p in probs] if total > 0 else None


def family(model: str) -> str:
    """A model's family for overlap warnings: the name without organisation, size and version suffixes."""
    name = model.split("/")[-1].lower()
    return re.split(r"[-_.]?\d", name, maxsplit=1)[0] or name
