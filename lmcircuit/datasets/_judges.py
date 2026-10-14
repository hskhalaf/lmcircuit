"""What the four datasets whose judges the paper queried (politicause, beavertails, arena, healthbench) share: their
judges, their prompts as lc.Prompt, and the recipe that reruns those calls with lc.collect (the same models, prompts,
reasoning levels and token limit; decoding at the models' defaults).
"""
from __future__ import annotations

import re

from ..collection import JudgeSpec
from ..llm import Backend, Prompt
from ._prompts import PROMPTS

TOP_LOGPROBS = 5        # alternatives per generated token the paper's calls asked for
MAX_TOKENS = 8192       # the paper's cap on reasoning + answer per call

# ---------------------------------------------------------------------------------------------- the paper's judges
# Price snapshot 2026-09-24. Per judge: Hub model id and pinned revision, the
# vLLM reasoning parser the server ran with, USD per million input and output tokens (each model's cheapest listed
# provider, blended 3:1 input:output) and its source. `request`: the fields each call added to messages, max_tokens
# and seed (a reasoning level; nothing else). Decoding stayed at each checkpoint's generation_config.
_GEMMA = {"reasoning_parser": "gemma4", "request": {}}
_QWEN = {"reasoning_parser": "qwen3", "request": {}}
SHARED_SEVEN = {
    "Gemma-4-E2B-IT": {**_GEMMA, "model": "google/gemma-4-E2B-it", "revision": "70af34e20bd4b7a91f0de6b22675850c43922a03",
                       "price": (0.04, 0.08), "price_source": "Amazon Bedrock (its only API host)"},
    "Gemma-4-E4B-IT": {**_GEMMA, "model": "google/gemma-4-E4B-it", "revision": "fee6332c1abaafb77f6f9624236c63aa2f1d0187",
                       "price": (0.02, 0.1), "price_source": "DeepInfra (its only API host)"},
    "Qwen3-4B": {**_QWEN, "model": "Qwen/Qwen3-4B", "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
                 "price": (0.03, 0.03), "price_source": "Novita, fp8 (cheaper than Fireworks 0.20/0.20)"},
    "Qwen3-8B": {**_QWEN, "model": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218",
                 "price": (0.2, 0.2),
                 "price_source": "Fireworks (blended below Alibaba 0.117/0.455, OpenRouter's only provider)"},
    "Gemma-4-26B-A4B-IT": {**_GEMMA, "model": "google/gemma-4-26B-A4B-it",
                           "revision": "20da991ab4afab98e8f910c4a2e8f4fbefc404ad", "price": (0.042, 0.22),
                           "price_source": "OpenRouter provider Darkbloom (cheapest of 12)"},
    "Gemma-4-31B-IT": {**_GEMMA, "model": "google/gemma-4-31B-it", "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
                       "price": (0.09, 0.34), "price_source": "OpenRouter provider DeepInfra, fp4 (cheapest of 13)"},
    "Qwen3-32B": {**_QWEN, "model": "Qwen/Qwen3-32B", "revision": "9216db5781bf21249d130ec9da846c4624c16137",
                  "price": (0.08, 0.28), "price_source": "OpenRouter provider DeepInfra, fp8 (cheaper of 2)"},
}
# Granite 4.2: enable_thinking=false gives no reasoning, reasoning_effort=low a low-effort hint, any other effort its
# default full reasoning ("default" = high). gpt-oss: the harmony "Reasoning: <level>" line.
_GRANITE = {"model": "ibm-granite/granite-4.2-3b", "revision": "e459acceac81e5fe67c07d9cfc72329a332e7eb1",
            "reasoning_parser": "qwen3", "price": (0.03, 0.12), "price_source": "DeepInfra (its only API host)"}
_OSS20 = {"model": "openai/gpt-oss-20b", "revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
          "reasoning_parser": "openai_gptoss", "price": (0.018, 0.09),
          "price_source": "OpenRouter provider Darkbloom, fp8 (cheapest of 12)"}
_OSS120 = {"model": "openai/gpt-oss-120b", "revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
           "reasoning_parser": "openai_gptoss", "price": (0.03, 0.17),
           "price_source": "OpenRouter providers AkashML (bf16) and CoreWeave (fp4), cheapest of 24"}
HEALTH_EIGHT = {
    "granite-4.2-3b::none": {**_GRANITE, "request": {"chat_template_kwargs": {"enable_thinking": False}}},
    "granite-4.2-3b::low": {**_GRANITE, "request": {"reasoning_effort": "low"}},
    "granite-4.2-3b::default": {**_GRANITE, "request": {"reasoning_effort": "high"}},
    "gpt-oss-20b::low": {**_OSS20, "request": {"reasoning_effort": "low"}},
    "gpt-oss-20b::medium": {**_OSS20, "request": {"reasoning_effort": "medium"}},
    "gpt-oss-20b::high": {**_OSS20, "request": {"reasoning_effort": "high"}},
    "gpt-oss-120b::low": {**_OSS120, "request": {"reasoning_effort": "low"}},
    "gpt-oss-120b::medium": {**_OSS120, "request": {"reasoning_effort": "medium"}},
}



def prompt(name: str, fields: list[str], classes: list[str], permute: tuple[str, ...] = (), rename=None):
    """The paper's prompt `name` (`_prompts`) as an lc.Prompt: braces other than the `fields` slots doubled for
    str.format; rename maps a prompt slot to the item field that fills it (e.g. presented_response_a -> response_a)."""
    rename = rename or {}
    slots = set(fields) | set(rename)

    def escape(text: str) -> str:
        out = text.replace("{", "{{").replace("}", "}}")
        out = re.sub(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}",
                     lambda m: "{" + rename.get(m.group(1), m.group(1)) + "}" if m.group(1) in slots else m.group(0), out)
        return out

    spec = PROMPTS[name]
    return Prompt(user=escape(spec["user"]), classes=list(classes), system=escape(spec["system"]) or None,
                  permute=tuple(permute))


def recipe(judges: dict, prompt_, base_urls: dict, served: dict | None = None):
    """JudgeSpecs that rerun the judges with lc.collect: one vLLM server per model (every level of a model on one
    server), keyed by the judge's name before "::"; the request carries the paper's reasoning level and asks for
    TOP_LOGPROBS alternatives per token (reported, not sampled from). served: model name each server answers to
    (default the Hub id). Prices are the paper's, so lc.collect records USD at the same rates."""
    served = served or {}
    out = []
    for name, spec in judges.items():
        base = name.split("::")[0]
        if base not in base_urls:
            raise KeyError(f"no base url for {base!r}; give one per model: {sorted({n.split('::')[0] for n in judges})}")
        backend = Backend.vllm(base_urls[base], served.get(base, spec["model"]), price_in=spec["price"][0],
                               price_out=spec["price"][1], extra_body=dict(spec["request"]), logprobs=TOP_LOGPROBS)
        out.append(JudgeSpec(name, backend, prompt_, max_tokens=MAX_TOKENS))
    return out
