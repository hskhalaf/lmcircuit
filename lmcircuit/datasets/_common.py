"""What the dataset loaders share: reading raw files, options over CONFIG, fetching pinned sources, clipping text,
and the one pricing rule.

Every loader's CONFIG follows one schema (dataset-specific keys added where needed):
    sources  {name: {"repo" or "url", "revision"}}    the pinned public sources `fetch` reads
    tasks    {name: description}                      the task description each item's prior text carries
    classes  [name, ...]                              class names in label order
    judges   {name: {"model", "params_b", "active_params_b", "price", "note", ...}}
             params_b: total parameters (billions); active_params_b: per token, a mixture of experts only;
             price: (USD per million input, output tokens), or None for the compute proxy at the active parameters
    cost     {...}                                    how call and planning costs are computed
    text     {...}                                    how the prior text is rendered
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from ..data import Judge

COMPUTE_USD_PER_B = 0.01   # compute proxy (as the Open LLM Leaderboard): USD per million tokens per billion active
                           # parameters (a forward pass costs about 2 x active parameters FLOPs per token)
JUDGE_KEYS = ("model", "params_b", "active_params_b", "price")


# ------------------------------------------------------------------------------------------------------ files
def read_jsonl(path):
    """Yields the records of a .jsonl or .jsonl.gz file (blank lines skipped)."""
    path = Path(path)
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def require(path, dataset: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run lc.datasets.fetch({dataset!r}, raw_dir) first")
    return path


def options(config: dict, overrides: dict, deep=()) -> dict:
    """CONFIG with the overrides: an unknown key raises; a key in `deep` (a dict) is merged one level, others replaced."""
    out = dict(config)
    for key, value in overrides.items():
        if key not in config:
            raise KeyError(f"unknown option {key!r}; known: {sorted(config)}")
        if key in deep:
            if unknown := set(value) - set(config[key]):
                raise KeyError(f"unknown {key} option(s) {sorted(unknown)}; known: {sorted(config[key])}")
            value = {**config[key], **value}
        out[key] = value
    return out


# ------------------------------------------------------------------------------------------------------ fetch
def retry(fn, tries: int = 8, wait: float = 15.0, cap: float = 300.0):
    """fn(), retried with exponential backoff on any error (network: rate limits, resets)."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:                                   # noqa: BLE001
            if attempt == tries - 1:
                raise
            print(f"[retry {attempt + 1}/{tries - 1}] {type(exc).__name__}: {exc}", flush=True)
            time.sleep(min(cap, wait * 2 ** attempt))


def hf_file(repo: str, filename: str, revision: str, dest, repo_type: str = "dataset") -> Path:
    """One file of a Hub repository at a pinned revision, saved as `dest` (nothing done when dest exists). The
    download's scratch directory is removed afterwards (retried: on network file systems the Hub client's lock files
    can linger for a moment)."""
    dest = Path(dest)
    if not dest.exists():
        from huggingface_hub import hf_hub_download
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.mkdtemp(dir=dest.parent)
        try:
            got = retry(lambda: hf_hub_download(repo, filename, repo_type=repo_type, revision=revision, local_dir=tmp))
            Path(got).replace(dest)
        finally:
            try:
                retry(lambda: shutil.rmtree(tmp), tries=4, wait=2.0)
            except OSError:
                print(f"[hf_file] could not remove {tmp}", flush=True)
    return dest


def download(url: str, dest, sha256: str | None = None) -> Path:
    """The file at url saved as `dest` (nothing done when dest exists), then checked against sha256 when given."""
    dest = Path(dest)
    if not dest.exists():
        from urllib.request import urlopen
        dest.parent.mkdir(parents=True, exist_ok=True)

        def get():
            with urlopen(url, timeout=120) as response:
                return response.read()
        part = dest.with_name(dest.name + ".part")
        part.write_bytes(retry(get))
        part.replace(dest)
    return verify(dest, sha256) if sha256 else dest


def checksum(path) -> str:
    """sha256 of a file's bytes (hex)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path, sha256: str) -> Path:
    """path, after checking that its bytes have the pinned sha256 (a mismatch is an error, the file is kept)."""
    if (got := checksum(path)) != sha256:
        raise ValueError(f"{path}: sha256 {got} differs from the pinned {sha256}")
    return Path(path)


def git_checkout(url: str, commit: str, dest, sparse=None) -> Path:
    """A git clone of url at commit in dest: cloned when absent (blobs on demand and only the `sparse` paths, when
    given), then checked out."""
    dest = Path(dest)
    git = lambda *a: subprocess.run(["git", *map(str, a)], check=True, capture_output=True)  # noqa: E731

    def clone():
        shutil.rmtree(dest, ignore_errors=True)
        git("clone", "-q", *(("--filter=blob:none", "--sparse", "--no-checkout") if sparse else ()), url, dest)

    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        retry(clone, tries=4)
    if sparse:
        git("-C", dest, "sparse-checkout", "set", *sparse)
    retry(lambda: git("-C", dest, "checkout", "-q", commit), tries=4)
    return dest


# ------------------------------------------------------------------------------------------------------- text
def clip(text, limit: int, mark: str = " ...", squash: bool = False, fit: bool = False) -> str:
    """text if it has at most `limit` characters, else its start and `mark`. squash: runs of whitespace become one
    space first. fit: the start is cut so that start and mark have `limit` characters; otherwise it has `limit`
    characters, trailing whitespace dropped."""
    text = " ".join(str(text).split()) if squash else str(text)
    if len(text) <= limit:
        return text
    return (text[:limit - len(mark)] if fit else text[:limit].rstrip()) + mark


# ------------------------------------------------------------------------------------------------------ costs
def compute_price(active_b: float) -> float:
    """USD per million tokens of an open model under the compute proxy: 0.01 x active parameters (billions)."""
    return COMPUTE_USD_PER_B * float(active_b)


def price(spec: dict) -> tuple[float, float]:
    """(input, output) USD per million tokens of a CONFIG judge: its listed price, else the compute proxy."""
    if spec.get("price") is not None:
        return float(spec["price"][0]), float(spec["price"][1])
    p = compute_price(spec["active_params_b"] if spec.get("active_params_b") is not None else spec["params_b"])
    return p, p


def make_judge(name: str, spec: dict, **meta) -> Judge:
    """A Judge from a CONFIG judge entry; its other keys (note, source files, ...) and `meta` go to Judge.meta."""
    p_in, p_out = price(spec)
    return Judge(name, model=spec.get("model"), price_in=p_in, price_out=p_out, params_b=spec.get("params_b"),
                 active_params_b=spec.get("active_params_b"),
                 meta={**{k: v for k, v in spec.items() if k not in JUDGE_KEYS}, **meta})


def call_cost(tokens_in, tokens_out, price_in, price_out) -> np.ndarray:
    """USD of calls: tokens x USD per million tokens (prices per judge broadcast along the last axis)."""
    return (np.asarray(tokens_in) * np.asarray(price_in) + np.asarray(tokens_out) * np.asarray(price_out)) / 1e6


def output_cap(tokens_out, valid=None, q: float = 0.95) -> np.ndarray:
    """(M,) each judge's output cap: the q quantile of its valid calls' whole output tokens (N, M); 0 with none."""
    tokens = np.round(np.asarray(tokens_out, dtype=np.float64))
    valid = np.ones(tokens.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    return np.asarray([np.quantile(tokens[valid[:, m], m], q) if valid[:, m].any() else 0.0
                       for m in range(tokens.shape[1])])


def planning_cost(tokens_in, price_in, price_out, out_cap) -> np.ndarray:
    """(N, M) USD the circuit plans with, known before the call: whole input tokens x input price + the judge's
    output cap x output price."""
    return call_cost(np.round(np.asarray(tokens_in, dtype=np.float64)), out_cap, price_in, price_out)
