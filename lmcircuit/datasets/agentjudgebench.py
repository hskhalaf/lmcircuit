"""AgentJudgeBench (ServiceNow-AI/AgentJudgeBench, Apache-2.0; arXiv 2608.26623): seven judges' recorded verdicts on
agentic tool-call plans, no inference. Reproduces the dataset used in the LMCircuit paper.

    lc.datasets.fetch("agentjudgebench", raw_dir); ds = lc.datasets.load("agentjudgebench", raw_dir)

The benchmark: 3,808 records (a user request, 8-19 tool schemas, a reference tool-call sequence), each asked at three
difficulties (easy, medium, hard rewrites of the request); five generator models planned tool calls for each
(record, difficulty); seven judges scored each plan on four metrics in {0, 0.5, 1}; a programmatic scorer compared the
plan with the reference (scores in [0, 1]). The judges are read without the reference (`without_gt`): the judge sees
the request, the tool schemas and the plan, as a deployed judge would.

Items: one per (record, generator, difficulty, metric) for the metrics in CONFIG["tasks"]; id
"<record>__<generator>__<difficulty>__<metric>"; task = the metric; group = the record. The label is 1 ("yes") when the
programmatic score is exactly 1 (the plan is fully right on that metric); a judge answers 1 when its score is 1 (its
0.5, "minor issue", counts as 0). Prior text: `item` is the request, the tool schemas compacted to one line each and
the planned calls, each block truncated (CONFIG["text"]); it holds the judged plan, so a prior on it acts as a free
judge.
Join: judge and generator rows meet on (record, generator, difficulty), unique once rows without an id are dropped (the
six general judges each have ~1,650 empty rows; QwQ, Gemini and gpt-oss-120b also miss some outputs). The programmatic
scores of every judge row must equal each other's and the generator row's. An output enters only if every judge
scored it and its generator row exists (the Llama-3.3-70B file repeats 1,375 rows in place of ~1,690 others).
Subsample: CONFIG["n_records"] records drawn with the seed among those with a complete output left; every remaining
generator, difficulty and metric of a drawn record is kept.
Cost (none recorded): input = the judges' own without-reference prompt (SyGra tasks/agentic_bfcl_judge_eval/
graph_config.yaml, `template_chars`) filled with the request, tool schemas and planned calls as the harness serialised
them (json.dumps, indent 2), at 4 characters per token; output = an assumed allowance per judge (verdict JSON plus
reasoning). One general-judge call scores every metric of a plan, so each (plan, metric) item is charged `call_share`
of the call. Planning cost (known before the call): the same input tokens, rounded to whole tokens, x input price plus
the (whole) output allowance x output price.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..data import Dataset
from ._common import call_cost, clip, hf_file, make_judge, options, output_cap, planning_cost, price, read_jsonl, require

CONFIG = {
    "sources": {"benchmark": {"repo": "ServiceNow-AI/AgentJudgeBench",                     # Apache-2.0
                              "revision": "13ac79a58bb39ded101e1222f1bdb80dcfbe27f6"}},
    "generators": ("llama_3_3_70b_instruct", "llama_3_1_8b_instruct", "qwen3_32b", "smollm3_3b", "gpt_5_4"),
    "exclude_generators": ("gpt_5_4",),   # GPT-5.4 is also a judge: its own plans are left out rather than self-graded
    # Two of the four metrics. On parameter_structure and sequence every general judge says 1 on 92-97% of plans and is
    # near chance against the programmatic score (balanced accuracy 0.53-0.58 on all 53k plans); tool_selection
    # (0.57-0.74) and query_coverage (0.62-0.85) separate the judges. Descriptions: what the label means (programmatic
    # scorer, SyGra task_executor.ProgrammaticJudge).
    "tasks": {
        "tool_selection": "Tool selection: decide whether an AI agent's planned tool calls use exactly the right set "
                          "of tools for the user's request, with every required tool present and no unneeded tool "
                          "added. Answer 1 if the tool set is fully correct, 0 otherwise.",
        "query_coverage": "Query coverage: decide whether an AI agent's planned tool calls cover every part of the "
                          "user's request, with each tool the request needs called at least once (extra calls are not "
                          "penalised). Answer 1 if coverage is complete, 0 otherwise.",
    },
    "classes": ["no", "yes"],   # 1: the plan is fully right on the metric (programmatic score 1)
    "n_records": 1000,
    "seed": 0,
    "text": {"tools_chars": 4000, "tool_description_chars": 200, "calls_chars": 3000},   # truncation of each block
    "cost": {"call_share": 0.5,          # 1 / the number of kept metrics: one call scores all metrics of a plan
             "template_chars": 3082,     # SyGra judge_tool_calls_answer_only prompt, placeholders removed
             "chars_per_token": 4.0},
    # The benchmark's seven judges, open-weight then API. file: data/judges/judge_<file>.jsonl. price: USD per million
    # input and output tokens (open models: the cheapest API host, or a same-size proxy; dated 2026-10-04 unless noted).
    # params_b: total parameters (billions); active_params_b: per token (mixtures of experts). verdict, reasoning:
    # assumed output tokens per call (none recorded): the four-metric JSON with one-sentence justifications, and a
    # reasoning allowance at the model's default setting.
    "judges": {
        "Prometheus-2-7B": {"file": "prometheus2_7b", "model": "prometheus-eval/prometheus-7b-v2.0", "params_b": 7.24,
                            "price": (0.15, 0.15), "verdict": 150, "reasoning": 0,
                            "note": "no API host; priced as OpenRouter mistralai/ministral-8b-2512; writes feedback "
                                    "then '[RESULT] n', one metric per call; no reasoning"},
        "gpt-oss-20b": {"file": "gpt_oss_20b", "model": "openai/gpt-oss-20b", "params_b": 20.9, "active_params_b": 3.6,
                        "price": (0.018, 0.09), "verdict": 200, "reasoning": 400,
                        "note": "OpenRouter provider Darkbloom, fp8; default (medium) effort"},
        "gpt-oss-120b": {"file": "gpt_oss_120b", "model": "openai/gpt-oss-120b", "params_b": 116.8, "active_params_b": 5.1,
                         "price": (0.03, 0.17), "verdict": 200, "reasoning": 400,
                         "note": "OpenRouter providers AkashML and CoreWeave; default (medium) effort"},
        "QwQ-32B": {"file": "qwq_32b", "model": "Qwen/QwQ-32B", "params_b": 32.5,
                    "price": (0.08, 0.28), "verdict": 200, "reasoning": 1500,
                    "note": "no longer listed on OpenRouter; priced as Qwen3-32B (DeepInfra); always thinks, long "
                            "traces"},
        "Gemini-2.5-Pro": {"file": "gemini_2_5_pro", "model": "google/gemini-2.5-pro",
                           "price": (1.25, 10.0), "verdict": 200, "reasoning": 800,
                           "note": "standard, prompts <= 200k; thinking cannot be switched off, billed as output"},
        "GPT-5.4": {"file": "gpt_5_4", "model": "openai/gpt-5.4",
                    "price": (2.5, 15.0), "verdict": 200, "reasoning": 0,
                    "note": "standard; reasoning effort 'none' assumed (the GPT-5.x API default), unverified"},
        "Claude-Sonnet-4.5": {"file": "claude_large", "model": "anthropic/claude-sonnet-4.5",
                              "price": (3.0, 15.0), "verdict": 200, "reasoning": 0,
                              "note": "extended thinking off by default"},
    },
}
METRICS = {"tool_selection": "score_tool_selection", "parameter_structure": "score_parameter_structure",
           "sequence": "score_sequence_accuracy", "query_coverage": "score_query_coverage"}   # -> generator-row field


def fetch(raw_dir) -> Path:
    """Download the records, the generators' plans and the judges' verdicts at the pinned revision (1.4 GB; files
    present are kept)."""
    raw, src = Path(raw_dir), CONFIG["sources"]["benchmark"]
    files = (["data/main/records.jsonl", "README.md"] + [f"data/{g}/records.jsonl" for g in CONFIG["generators"]]
             + [f"data/judges/judge_{j['file']}.jsonl" for j in CONFIG["judges"].values()])
    for name in files:
        hf_file(src["repo"], name, src["revision"], raw / name)
    (raw / "source.json").write_text(json.dumps(src))
    return raw


def _tools(tools, limit: int, description_chars: int) -> str:
    """One line per tool: name(required params; optional: others): description."""
    lines = []
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        params = fn.get("parameters") or {}
        props = list((params.get("properties") or {}).keys())
        required = [p for p in (params.get("required") or []) if p in props]
        optional = [p for p in props if p not in required]
        sig = ", ".join(required) + (("; optional: " + ", ".join(optional)) if optional else "")
        description = clip(str(fn.get("description") or "").strip(), description_chars)
        lines.append(f"- {fn.get('name', '?')}({sig}): {description}")
    return clip("\n".join(lines), limit)


def _calls(calls, limit: int) -> str:
    """The planned calls in order, one per line: name(arguments as compact JSON)."""
    if isinstance(calls, str):
        return clip(calls.strip() or "(no tool calls)", limit)
    lines = []
    for n, call in enumerate(calls or [], 1):
        fn = call.get("function", call) if isinstance(call, dict) else {}
        args = fn.get("arguments")
        args = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False, separators=(", ", ": "))
        lines.append(f"{n}. {fn.get('name', '?')}({args})")
    return clip("\n".join(lines), limit) if lines else "(no tool calls)"


def _request(row: dict, spec: dict) -> str:
    return (f"User request:\n{row['user_query'].strip()}\n\n"
            f"Available tools:\n{_tools(row['available_tools'], spec['tools_chars'], spec['tool_description_chars'])}")


def _input_chars(row: dict, template_chars: int) -> int:
    """Characters of the without-reference judge prompt: the template plus the fields as SyGra serialised them."""
    ser = lambda v: v if isinstance(v, str) else json.dumps(v, indent=2)   # noqa: E731
    return template_chars + sum(len(ser(row[f])) for f in ("user_query", "available_tools", "generated_tool_calls"))


def _generator_rows(raw: Path, generators) -> dict:
    """(record, generator, difficulty) -> row; a repeated row must repeat the same content."""
    gen = {}
    for g in generators:
        for r in read_jsonl(require(raw / f"data/{g}/records.jsonl", "agentjudgebench")):
            key = (r["id"], g, r["difficulty"])
            if key in gen:
                if any(json.dumps(gen[key][f], sort_keys=True) != json.dumps(r[f], sort_keys=True)
                       for f in ("user_query", "generated_tool_calls", *METRICS.values())):
                    raise ValueError(f"generator rows {key} repeat with different content")
                continue
            gen[key] = r
    return gen


def _verdicts(path: Path) -> dict:
    """(record, generator, difficulty) -> metric -> (judge score, programmatic score), without the reference; rows
    without an id are dropped, a repeated row must repeat the same scores."""
    out = {}
    for r in read_jsonl(require(path, "agentjudgebench")):
        if r["id"] is None:
            continue
        key = (r["id"], r["generator"], r["difficulty"])
        block = r.get("without_gt") or {}
        v = {m: (b.get("llm_score"), b.get("programmatic_score")) if isinstance(b := block.get(m), dict)
             else (None, None) for m in METRICS}
        if out.setdefault(key, v) != v:
            raise ValueError(f"{path.name}: {key} repeats with different verdicts")
    return out


def load(raw_dir, **overrides) -> Dataset:
    """Build the dataset from the raw files; overrides replace CONFIG keys (e.g. n_records=..., seed=...)."""
    cfg = options(CONFIG, overrides, deep=("text", "cost"))
    raw, tasks, names, spec, cs = Path(raw_dir), list(cfg["tasks"]), list(cfg["judges"]), cfg["text"], cfg["cost"]
    if abs(cs["call_share"] * len(tasks) - 1.0) > 1e-9:
        raise ValueError(f"cost.call_share {cs['call_share']} should be 1 / {len(tasks)} kept metrics")
    gen = _generator_rows(raw, cfg["generators"])
    verdicts = {j: _verdicts(raw / f"data/judges/judge_{cfg['judges'][j]['file']}.jsonl") for j in names}

    # complete outputs: every judge scored every kept metric, programmatic scores agree, the plan exists
    by_record = defaultdict(list)
    for key in sorted(set.intersection(*(set(v) for v in verdicts.values()))):
        progs = {tuple(verdicts[j][key][m][1] for m in METRICS) for j in names}
        if len(progs) != 1:
            raise ValueError(f"{key}: programmatic scores differ across judges")
        if any(verdicts[j][key][m][0] is None for j in names for m in tasks) or key not in gen:
            continue
        if [round(float(p), 4) for p in progs.pop()] != [round(float(gen[key][f]), 4) for f in METRICS.values()]:
            raise ValueError(f"{key}: programmatic scores differ from the generator row")
        if key[1] not in cfg["exclude_generators"]:
            by_record[key[0]].append(key)
    rng = np.random.default_rng(int(cfg["seed"]))
    eligible = sorted(by_record)
    records = sorted(rng.choice(eligible, size=min(int(cfg["n_records"]), len(eligible)), replace=False).tolist())

    items, answers, labels, chars = [], [], [], []
    for key in (k for rec in records for k in sorted(by_record[rec])):
        row = gen[key]
        request = _request(row, spec)
        item = f"{request}\n\nPlanned tool calls:\n{_calls(row['generated_tool_calls'], spec['calls_chars'])}"
        for m in tasks:
            items.append({"id": "__".join((*key, m)), "task": m, "task_description": cfg["tasks"][m], "group": key[0],
                          "item": item})
            answers.append([int(verdicts[j][key][m][0] == 1.0) for j in names])
            labels.append(int(float(verdicts[names[0]][key][m][1]) == 1.0))
            chars.append(_input_chars(row, cs["template_chars"]))

    share = float(cs["call_share"])
    p_in, p_out = np.asarray([price(cfg["judges"][j]) for j in names]).T
    tokens_out = np.broadcast_to([(cfg["judges"][j]["verdict"] + cfg["judges"][j]["reasoning"]) * share for j in names],
                                 (len(items), len(names)))
    tokens_in = (np.asarray(chars, dtype=np.float64) / cs["chars_per_token"])[:, None] * share       # (N, 1)
    return Dataset.from_records(
        items, [make_judge(j, cfg["judges"][j]) for j in names], np.asarray(answers),
        call_cost(tokens_in, tokens_out, p_in, p_out), n_classes=2, classes=cfg["classes"], labels=labels,
        prior_parts=("task_description", "item"),
        plan_cost=planning_cost(tokens_in, p_in, p_out, output_cap(tokens_out)),
        meta={"source": "{repo}@{revision}".format(**cfg["sources"]["benchmark"]), "licence": "Apache-2.0",
              "condition": "without_gt", "label": "programmatic_score == 1", "answer": "llm_score == 1 (0.5 and 0 -> 0)",
              "records": records, "cost": cs})
