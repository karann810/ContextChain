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
        "model":       "gpt-3.5-turbo",
        "api_key_env": "AIMLAPI_KEY",
        "temperature": 0.1,
        "max_tokens": 200,
        "platform":    "AIML API",
        "reason":      "Reasoning model for implicit constraint extraction",
    },
    "vendor_intelligence_v1": {
        "provider":    "aimlapi",
        "model":       "gpt-3.5-turbo",
        "api_key_env": "AIMLAPI_KEY",
        "temperature": 0.05,
        "max_tokens": 300,
        "platform":    "AIML API (fast fallback)",
        "reason":      "Use faster hosted model for vendor scoring to reduce latency",
    },
    "risk_auditor_v1": {
        "provider":    "aimlapi",
        "model":       "deepseek/deepseek-r1",
        "api_key_env": "AIMLAPI_KEY",
        "temperature": 0.05,
        "max_tokens": 200,
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
        print(f"  [{agent_id}] {cfg['platform']} -> {cfg['model']}")
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
    max_tokens = cfg.get("max_tokens", 800)

    client_kwargs = {"api_key": cfg["api_key"]}
    if cfg.get("api_base"):
        client_kwargs["base_url"] = cfg["api_base"]

    client = OpenAI(**client_kwargs)

    # Ensure raw is always defined so error paths don't reference an unassigned variable
    raw = ""

    # Retry on transient rate-limit errors with exponential backoff
    import time
    import openai as _openai

    max_retries = 3
    backoff = 1.0

    # Build a candidate list of (cfg, model) tuples to try.
    # start with the configured model for this provider
    model_candidates = [(cfg, cfg.get("model"))]
    # If the configured provider is Featherless (heavy), prefer trying an AIML API candidate first
    # since remote hosted models (e.g., gpt-3.5-turbo) often respond faster than pulling large open models.
    if cfg.get("provider") == "featherless":
        aiml_agent_id = None
        for aid, a in AGENT_MODELS.items():
            if a.get("provider") == "aimlapi":
                aiml_agent_id = aid
                break
        if aiml_agent_id:
            try:
                aiml_cfg = get_model_config(aiml_agent_id)
                # put AIML candidate before the heavy featherless model
                model_candidates.insert(0, (aiml_cfg, aiml_cfg.get("model")))
            except Exception:
                pass
   
    last_exc = None
    for candidate in model_candidates:
        candidate_cfg, model_name = candidate
        if not model_name:
            continue
        # create a client for this candidate's provider/config
        client_kwargs_candidate = {"api_key": candidate_cfg["api_key"]}
        if candidate_cfg.get("api_base"):
            client_kwargs_candidate["base_url"] = candidate_cfg["api_base"]
        client_candidate = OpenAI(**client_kwargs_candidate)

        for attempt in range(1, max_retries + 1):
            try:
                response = client_candidate.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=candidate_cfg.get("max_tokens", max_tokens),
                )
                # safely extract text from various client response shapes
                ch = response.choices[0]
                msg = getattr(ch, "message", None)
                if msg is not None and getattr(msg, "content", None) is not None:
                    raw = msg.content.strip()
                else:
                    raw = (getattr(ch, "text", None) or getattr(ch, "content", None) or "").strip()
                # update original cfg model to the successfully used one for future calls
                cfg["model"] = model_name
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                is_rate = False
                se = str(e).lower()
                # common rate/limit indicators
                if hasattr(_openai, 'RateLimitError') and isinstance(e, _openai.RateLimitError):
                    is_rate = True
                elif 'concurrency_limit_exceeded' in se or 'ratelimiterror' in e.__class__.__name__.lower() or 'rate_limit' in se:
                    is_rate = True

                # treat provider capacity / 503 server errors as transient (e.g. Featherless Qwen capacity_exhausted)
                if 'capacity_exhausted' in se or 'temporarily at capacity' in se or ('503' in se and 'capacity' in se):
                    is_rate = True

                # If model not found, break to try next candidate immediately
                if hasattr(_openai, 'NotFoundError') and isinstance(e, _openai.NotFoundError):
                    break
                if 'model_not_found' in se or 'does not exist' in se:
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
        # If Featherless provider failed, try AIML API as a provider-level fallback
        try:
            if cfg.get("provider") == "featherless":
                # pick a configured AIML API model (first one found)
                aiml_agent_id = None
                for aid, a in AGENT_MODELS.items():
                    if a.get("provider") == "aimlapi":
                        aiml_agent_id = aid
                        break
                if aiml_agent_id:
                    try:
                        fallback_cfg = get_model_config(aiml_agent_id)
                        fb_client_kwargs = {"api_key": fallback_cfg["api_key"]}
                        if fallback_cfg.get("api_base"):
                            fb_client_kwargs["base_url"] = fallback_cfg["api_base"]
                        fb_client = OpenAI(**fb_client_kwargs)
                        fb_resp = fb_client.chat.completions.create(
                            model=fallback_cfg.get("model"),
                            messages=messages,
                            temperature=fallback_cfg.get("temperature", temp),
                            max_tokens=fallback_cfg.get("max_tokens", max_tokens),
                        )
                        ch = fb_resp.choices[0]
                        msg = getattr(ch, "message", None)
                        if msg is not None and getattr(msg, "content", None) is not None:
                            raw = msg.content.strip()
                        else:
                            raw = (getattr(ch, "text", None) or getattr(ch, "content", None) or "").strip()
                        return re.sub(r"^```(?:json)?\s*", "", re.sub(r"\s*```$", "", raw))
                    except Exception as fb_e:
                        # if fallback failed, prefer to raise the original error
                        pass
        except Exception:
            pass

        # no candidate or fallback succeeded
        raise last_exc
    # Strip markdown code fences if model wraps output in them
    raw = re.sub(r"^```(?:json)?\s*", "", (raw or ""))
    raw = re.sub(r"\s*```$", "", raw)
    return raw
