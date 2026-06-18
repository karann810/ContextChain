"""
ContextChain — Model Configuration v4.1
========================================
Uses OpenAI client directly for AIML API (bypasses litellm routing issues).
Uses litellm only for Featherless and fallback.

Agent routing:
  Agent 1 (Needs Analyzer)       → AIML API + deepseek/deepseek-r1
  Agent 2 (Vendor Intelligence)  → Featherless + Qwen2.5-72B
  Agent 3 (Risk Auditor)         → AIML API + deepseek/deepseek-r1
"""

import os
import re
import json
from dotenv import load_dotenv
load_dotenv()

AGENT_MODELS = {
    "needs_analyzer_v1": {
        "provider":    "aimlapi",
        "model":       "deepseek/deepseek-r1",
        "api_key_env": "AIMLAPI_KEY",
        "temperature": 0.1,
        "platform":    "AIML API",
        "reason":      "Reasoning model for implicit constraint extraction",
    },
    "vendor_intelligence_v1": {
        "provider":    "featherless",
        "model":       "featherless-ai/Qwen2-7B-Instruct",
        "api_key_env": "FEATHERLESS_API_KEY",
        "temperature": 0.05,
        "platform":    "Featherless",
        "reason":      "Open-source domain model for structured vendor scoring",
    },
    "risk_auditor_v1": {
        "provider":    "aimlapi",
        "model":       "deepseek/deepseek-r1",
        "api_key_env": "AIMLAPI_KEY",
        "temperature": 0.05,
        "platform":    "AIML API",
        "reason":      "Reasoning model for adversarial compliance auditing",
    },
}

PROVIDER_BASES = {
    "aimlapi":     "https://api.aimlapi.com/v1",
    "featherless": "https://api.featherless.ai/v1",
}


def get_model_config(agent_id: str) -> dict:
    cfg = AGENT_MODELS.get(agent_id, {})
    if not cfg:
        return {"provider":"openai","model":"gpt-4o-mini","platform":"OpenAI","temperature":0.1}

    key = os.getenv(cfg["api_key_env"], "").strip()
    if key:
        print(f"  [{agent_id}] {cfg['platform']} → {cfg['model']}")
        return {**cfg, "api_key": key, "api_base": PROVIDER_BASES[cfg["provider"]]}
    else:

       raise ValueError(
        f"Missing API key: set {cfg['api_key_env']} in your .env file or PowerShell")       
       


def call_llm(agent_id: str, messages: list, override_temp: float = None) -> str:
    """
    Single LLM call entry point.
    - AIML API and Featherless: uses openai client directly (most reliable)
    - OpenAI fallback: uses openai client with default base
    """
    from openai import OpenAI

    cfg = get_model_config(agent_id)
    temp = override_temp if override_temp is not None else cfg.get("temperature", 0.1)

    client_kwargs = {"api_key": cfg["api_key"]}
    if cfg.get("api_base"):
        client_kwargs["base_url"] = cfg["api_base"]

    client = OpenAI(**client_kwargs)

    # Retry on transient rate-limit errors with exponential backoff
    import time
    import openai as _openai

    max_retries = 3
    backoff = 1.0

    # If using Featherless, try a small set of candidate model names (some providers use different naming)
    model_candidates = [cfg.get("model")]
    if cfg.get("provider") == "featherless":
        model_candidates += [
            # common alternate namings to try
            "Qwen/Qwen2-7B-Instruct",
            "Qwen/Qwen2-14B-Instruct",
            "featherless-ai/Qwen2-7B-Instruct",
            "featherless-ai/Qwen2-3B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
        ]

    last_exc = None
    for model_name in model_candidates:
        if not model_name:
            continue
        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temp,
                )
                raw = response.choices[0].message.content.strip()
                # update cfg model to the successfully used one for future calls
                cfg["model"] = model_name
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                is_rate = False
                if hasattr(_openai, 'RateLimitError') and isinstance(e, _openai.RateLimitError):
                    is_rate = True
                elif 'concurrency_limit_exceeded' in str(e) or 'RateLimitError' in e.__class__.__name__:
                    is_rate = True

                # If model not found, break to try next candidate immediately
                if hasattr(_openai, 'NotFoundError') and isinstance(e, _openai.NotFoundError):
                    break
                if 'model_not_found' in str(e) or 'does not exist' in str(e):
                    break

                if is_rate and attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                # otherwise don't retry this model
                break
        if last_exc is None:
            break

    if last_exc is not None:
        # no candidate succeeded
        raise last_exc
    # Strip markdown code fences if model wraps output in them
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw