"""Registered datasets: `fetch(name, raw_dir)` downloads the pinned public sources (files present are kept),
`load(name, raw_dir, **options)` builds the Dataset. Each loader module keeps its choices (sources, tasks, classes,
judges and prices, costs, text rendering) in a CONFIG dict at its top (helpers: `_common`); the options of `load`
override its keys.

    lc.datasets.fetch("oll_bbh", raw); ds = lc.datasets.load("oll_bbh", raw)

Three datasets come with their judges' recorded answers (Decision Index, AgentJudgeBench, OLL BBH). For the four
whose judges the paper queried itself (PolitiCAUSE, BeaverTails, LMArena, HealthBench), the calls are not
distributed: `load` gives the items alone, and each module's `recipe(base_urls)` returns the lc.JudgeSpecs that rerun
the paper's calls with lc.collect (their judges and prompts: `_judges`, `_prompts`). Calls recorded earlier in
<raw>/recorded (the format in `_recorded`) are joined by id instead.
"""
from __future__ import annotations

import importlib

REGISTRY = (
    "decision_index",       # 18 leaderboard entrants' recorded choices on six binary task families
    "agentjudgebench",      # 7 judges of agents' planned tool calls
    "oll_bbh",              # 46 open models on eight binary BBH tasks (Open LLM Leaderboard details)
    "politicause",          # causal sentences; rerun 7 open judges with `recipe`
    "beavertails",          # response safety; rerun 7 open judges with `recipe`
    "arena",                # human A/B preference; rerun 7 open judges with `recipe` (per-call presented order)
    "healthbench",          # physician judgments as counts; rerun 8 reasoning configurations with `recipe`
)


def _module(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; registered: {list(REGISTRY)}")
    return importlib.import_module(f"{__name__}.{name}")


def fetch(name: str, raw_dir, **options):
    return _module(name).fetch(raw_dir, **options)


def load(name: str, raw_dir, **options):
    return _module(name).load(raw_dir, **options)
