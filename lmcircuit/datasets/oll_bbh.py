"""Open LLM Leaderboard v2, the binary BIG-Bench Hard tasks: 46 open models' recorded per-question results (one
log-likelihood scoring with the leaderboard's lm-eval harness and few-shot prompt); no inference. Reproduces the
dataset used in the LMCircuit paper.

    lc.datasets.fetch("oll_bbh", raw_dir); ds = lc.datasets.load("oll_bbh", raw_dir)

Items: every question of the 8 binary BBH tasks that every judge scored, id "<task>_<doc_id>", task the BBH task;
label the index of the gold option among the two options in the order the harness scored them. A judge's answer is
the option the leaderboard scored best (the gold one if its `acc_norm` is 1, else the other). Prior text: `item` the
question with its options as the leaderboard shows them; `task_description` BBH's own description of
the task (CONFIG["tasks"]: the system line of the leaderboard prompt of a chat-template model).
Cost (USD): one forward pass over the model's own prompt, tokens = characters / 4, at the compute proxy 0.01 x
active parameters (billions) per million tokens; nothing is generated. Planning cost: whole input tokens x price.
The detail datasets are gated with automatic approval: the Hugging Face token must have accepted them.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..data import Dataset
from ._common import call_cost, hf_file, make_judge, options, planning_cost, price, read_jsonl, require, retry

# The roster by size: name, model, total parameters (billions, open-llm-leaderboard/contents), revision of
# open-llm-leaderboard/<org>__<model>-details (its latest run of each task is used). Every model with details for all
# 8 tasks; Phi-3-small is left out (formal_fallacies at 1.00), dbrx has no details.
_ROSTER = (
    ("SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct", 0.362, "a65cde931ded7ffad125f9326986ce0cafd82ba9"),
    ("Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", 0.5, "4721da58318d33ba2ca9ff4d6d62ffe958719061"),
    ("flan-t5-large", "google/flan-t5-large", 0.783, "0e608e573d35ce9ce52dc7b90e878aa01bbe6dd9"),
    ("Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", 1.24, "1e5bafda0aae72d351dbc1ed542f7f63ae3080cf"),
    ("Falcon3-1B-Instruct", "tiiuae/Falcon3-1B-Instruct", 1.669, "b1a34cf2abc9a3cadbc93c7d68c5967a78e08e77"),
    ("SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct", 1.711, "fe1c833d11264bee46b890165f2e68128985f86f"),
    ("internlm2_5-1_8b-chat", "internlm/internlm2_5-1_8b-chat", 1.89, "ba3beae904e4443acc4b36830cbb3eedc3ac13f8"),
    ("gemma-2-2b-jpn-it", "google/gemma-2-2b-jpn-it", 2.614, "e4f670ed41c5987f6a63142b7b1ae8cbbd2c0404"),
    ("granite-3.0-2b-instruct", "ibm-granite/granite-3.0-2b-instruct", 2.634, "398fd99c1dd4bf2add8440d45e36ef5a44db18e4"),
    ("phi-2", "microsoft/phi-2", 2.78, "db9ccb47f9e038b47ee52a2b49b85a776e54ad74"),
    ("stablelm-zephyr-3b", "stabilityai/stablelm-zephyr-3b", 2.795, "38c80df24dacea230a69d8ee3b3ac3f5684bcc84"),
    ("Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct", 3.0, "f5b005e407b8546e16de9349a82bbd40b0133abe"),
    ("Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", 3.213, "1298247067f53fb3b1de7376ce5798dbeaae20b3"),
    ("granite-3.1-3b-a800m-instruct", "ibm-granite/granite-3.1-3b-a800m-instruct", 3.299,
     "c4d9ce6208dae45981bf91cbd86c7011819572b1"),
    ("Yi-1.5-6B-Chat", "01-ai/Yi-1.5-6B-Chat", 6.061, "46f188974351bc7cdaff2247c633958c3c78c206"),
    ("deepseek-llm-7b-chat", "deepseek-ai/deepseek-llm-7b-chat", 7.0, "9e3fa0a886f031787a74836da9761b50639b652c"),
    ("Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.3", 7.248, "9c17f27f82aab79c5c749ae5cd42289b2d939b30"),
    ("Falcon3-7B-Instruct", "tiiuae/Falcon3-7B-Instruct", 7.456, "824d33dab38bf69579a14379d484a3a240b85f07"),
    ("Qwen2-7B-Instruct", "Qwen/Qwen2-7B-Instruct", 7.616, "1bab2d4dfa3c554938354d005d496fa3ce05c19f"),
    ("internlm2_5-7b-chat", "internlm/internlm2_5-7b-chat", 7.738, "65139f35c9d1398485a51ba305a22edac6da5631"),
    ("Llama-3.1-Tulu-3-8B-DPO", "allenai/Llama-3.1-Tulu-3-8B-DPO", 8.0, "9de6e56b7d04ea4fc21a978dab513e4a54fabe62"),
    ("c4ai-command-r7b-12-2024", "CohereForAI/c4ai-command-r7b-12-2024", 8.028, "92625096a1930c0bbd108db993c54e5612efae54"),
    ("Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct", 8.03, "ed2f4d103e3721b246eeb6eb5b28943423212686"),
    ("granite-3.2-8b-instruct", "ibm-granite/granite-3.2-8b-instruct", 8.171, "7a6031ac54001373641fed247e92a8d61a480e7b"),
    ("gemma-2-9b-it", "google/gemma-2-9b-it", 9.0, "266b91b5514a9b8c768f2a5beb10681666bd2e40"),
    ("Falcon3-10B-Instruct", "tiiuae/Falcon3-10B-Instruct", 10.306, "b170d77f2755620dce0c8c0a5fb9359b8749103f"),
    ("SOLAR-10.7B-Instruct-v1.0", "upstage/SOLAR-10.7B-Instruct-v1.0", 10.732, "6abc47779ff29582ad5b22de306c5fcec621014f"),
    ("Mistral-Nemo-Instruct-2407", "mistralai/Mistral-Nemo-Instruct-2407", 12.248, "f808545982e269018ed5c6f57a8b7742ed00fb7c"),
    ("phi-4", "microsoft/phi-4", 14.66, "e15be88ebe32f7d656c1583b14a9cc0b9be3dd2e"),
    ("Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-14B-Instruct", 14.77, "2701cdcd15de5ea44cf54390e0024c8fa22dbe45"),
    ("DeepSeek-R1-Distill-Qwen-14B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 14.77,
     "9bd1423474d97ad005b2639b5624e7378ee6e4bb"),
    ("internlm2_5-20b-chat", "internlm/internlm2_5-20b-chat", 19.86, "c406156908fbe55777e269b0c3f9dfee7ea57854"),
    ("Mistral-Small-Instruct-2409", "mistralai/Mistral-Small-Instruct-2409", 22.05, "de99240461743e9dfdd02d6259277cbcea66f683"),
    ("solar-pro-preview-instruct", "upstage/solar-pro-preview-instruct", 22.14, "c735a4dda18c976bb6341f519a79f50b222b8b20"),
    ("gemma-2-27b-it", "google/gemma-2-27b-it", 27.227, "aa6ae095c48d20abf5e6b16d450c5fdf4a99c654"),
    ("aya-expanse-32b", "CohereForAI/aya-expanse-32b", 32.296, "632cf980e68c8f933592f7dc477c9923062f35c5"),
    ("Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-32B-Instruct", 32.764, "ce4158e5b934917c7bb03d74b80506191a4b90a5"),
    ("Yi-1.5-34B-Chat-16K", "01-ai/Yi-1.5-34B-Chat-16K", 34.389, "3481a8c231ffcd23bcd7761480518f71532c6062"),
    ("Phi-3.5-MoE-instruct", "microsoft/Phi-3.5-MoE-instruct", 42.0, "6efabcc8b19ae5849f53c958fb907c0cd0358f90"),
    ("Mixtral-8x7B-Instruct-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1", 46.703, "173743c78962f4e10c8fa8976a27e524fcf58833"),
    ("Jamba-v0.1", "ai21labs/Jamba-v0.1", 51.57, "217c92b9fbc053d56035c7c36cc1543955f4b05e"),
    ("Llama-3.3-70B-Instruct", "meta-llama/Llama-3.3-70B-Instruct", 70.554, "055514a0f75714c56f937ef40dcdf4093af76e1e"),
    ("Llama-3.1-Tulu-3-70B", "allenai/Llama-3.1-Tulu-3-70B", 70.554, "884e1dab69b9a83cdc3e1cf449b1880370615ef8"),
    ("Qwen2.5-72B-Instruct", "Qwen/Qwen2.5-72B-Instruct", 72.706, "23bcafd84477a3c523873f8d9374618499bba167"),
    ("c4ai-command-r-plus-08-2024", "CohereForAI/c4ai-command-r-plus-08-2024", 103.811,
     "f74d4c16c5a0a701b0cf9ff4cbea51844720e79b"),
    ("Mistral-Large-Instruct-2411", "mistralai/Mistral-Large-Instruct-2411", 122.61, "4a639c6f4d53fdea7d4958f6e7087ef4632c27a7"),
)
# mixtures of experts: parameters active per token (billions), from the model cards
_ACTIVE = {"granite-3.1-3b-a800m-instruct": 0.8, "Phi-3.5-MoE-instruct": 6.6, "Mixtral-8x7B-Instruct-v0.1": 12.9,
           "Jamba-v0.1": 12.0}

CONFIG = {
    "sources": {"details": {"repo": "open-llm-leaderboard/<org>__<model>-details", "revision": "per judge"}},
    # the 8 binary BBH tasks and BBH's own descriptions (the system line of the leaderboard's chat-template prompt)
    "tasks": {
        "bbh_boolean_expressions": "Evaluate the result of a random Boolean expression.",
        "bbh_causal_judgement": "Answer questions about causal attribution.",
        "bbh_formal_fallacies": "Distinguish deductively valid arguments from formal fallacies.",
        "bbh_hyperbaton": "Order adjectives correctly in English sentences.",
        "bbh_navigate": "Given a series of navigation instructions, determine whether one would end up back at the "
                        "starting point.",
        "bbh_snarks": "Determine which of two sentences is sarcastic.\n\nAccording to Cambridge University Dictionary, "
                      "sarcasm is \"the use of remarks that clearly mean the opposite of what they say, made in order to "
                      "hurt someone's feelings or to criticize something in a humorous way.\" Sarcastic sentences often "
                      "contain satirical or ironic utterances, hyperboles, ambivalent or witty remarks.",
        "bbh_sports_understanding": "Determine whether an artificially constructed sentence relating to sports is "
                                    "plausible or not.",
        "bbh_web_of_lies": "Evaluate a random boolean function expressed as a word problem.",
    },
    "classes": ["option 0", "option 1"],   # 1: the second option in the harness's order (e.g. True; No; invalid; (B))
    # price None: the compute proxy at the active parameters (dense models: the total)
    "judges": {n: {"model": m, "params_b": p, "active_params_b": _ACTIVE.get(n), "price": None, "revision": r}
               for n, m, p, r in _ROSTER},
    "cost": {"chars_per_token": 4.0},   # tokenizers not loaded
}


def _dir(raw: Path, model: str) -> Path:
    return raw / model.replace("/", "__")


def fetch(raw_dir) -> Path:
    """raw/<org>__<model>/<task>.jsonl: the latest run of each task in the pinned details revision (files present are
    kept)."""
    from huggingface_hub import HfApi
    raw = Path(raw_dir)
    for spec in CONFIG["judges"].values():
        todo = [t for t in CONFIG["tasks"] if not (_dir(raw, spec["model"]) / f"{t}.jsonl").exists()]
        if not todo:
            continue
        repo = f"open-llm-leaderboard/{spec['model'].replace('/', '__')}-details"
        files = retry(lambda: HfApi().list_repo_files(repo, repo_type="dataset", revision=spec["revision"]))
        for task in todo:
            runs = sorted((f for f in files if re.search(rf"samples_leaderboard_{task}_\d{{4}}-.*\.jsonl?$", f)),
                          key=lambda f: re.search(r"_(\d{4}-\d\d-\d\dT[\d.-]+)\.json", f).group(1))
            dest = _dir(raw, spec["model"]) / f"{task}.jsonl"
            part = hf_file(repo, runs[-1], spec["revision"], dest.with_suffix(".part"))
            text = part.read_text()
            if runs[-1].endswith(".json") and text.lstrip().startswith("["):   # older runs: one JSON array
                text = "".join(json.dumps(r) + "\n" for r in json.loads(text))
            dest.write_text(text)
            part.unlink()
    return raw


def load(raw_dir, **overrides) -> Dataset:
    """overrides replace CONFIG keys (e.g. a smaller `judges` roster; the items are those every judge scored)."""
    cfg = options(CONFIG, overrides, deep=("cost",))
    raw, judges = Path(raw_dir), list(cfg["judges"])
    records, items = defaultdict(dict), {}                     # (task, doc_id) -> judge -> (answer, characters)
    for judge in judges:
        for task in cfg["tasks"]:
            for r in read_jsonl(require(_dir(raw, cfg["judges"][judge]["model"]) / f"{task}.jsonl", "oll_bbh")):
                args = r["arguments"]
                choices = [args[f"gen_args_{i}"]["arg_1"].strip() for i in range(len(args))]
                if len(choices) != 2 or r["target"].strip() not in choices:
                    continue
                y, key = choices.index(r["target"].strip()), (task, int(r["doc_id"]))
                records[key][judge] = (y if int(r["acc_norm"]) == 1 else 1 - y, len(args["gen_args_0"]["arg_0"]))
                prev = items.setdefault(key, {"id": f"{task}_{key[1]}", "task": task, "item": r["doc"]["input"],
                                              "task_description": cfg["tasks"][task],
                                              "gt": y, "hash": r["doc_hash"]})
                if (prev["gt"], prev["hash"]) != (y, r["doc_hash"]):
                    raise ValueError(f"{key}: judges disagree on the item or its gold option")
    keys = sorted(k for k in items if len(records[k]) == len(judges))          # every judge scored it
    answers = np.asarray([[records[k][j][0] for j in judges] for k in keys], dtype=np.int64)
    tokens = np.asarray([[records[k][j][1] for j in judges] for k in keys]) / cfg["cost"]["chars_per_token"]
    p = np.asarray([price(cfg["judges"][j])[0] for j in judges])
    return Dataset.from_records(
        [items[k] for k in keys], [make_judge(j, cfg["judges"][j]) for j in judges], answers,
        call_cost(tokens, 0.0, p, p), n_classes=2, classes=cfg["classes"], labels=np.asarray([items[k]["gt"] for k in keys]),
        prior_parts=("task_description", "item"), plan_cost=planning_cost(tokens, p, p, 0.0),
        meta={"source": "open-llm-leaderboard per-question details (latest run per task at each pinned revision)",
              "cost": cfg["cost"],
              "price": "compute proxy: 0.01 x active parameters (billions) per million tokens, input only (one forward pass)",
              "plan_cost": "whole input tokens x price (nothing generated)"})
