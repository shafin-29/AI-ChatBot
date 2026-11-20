from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load the .env that sits next to this file, so `streamlit run` works no matter
# which directory it is launched from. Existing environment variables win.
load_dotenv(Path(__file__).resolve().parent / ".env")

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

# Used only if the live model list cannot be fetched from OpenRouter. The free
# lineup changes often, so the app normally fetches the current list instead.
# Override with OPENROUTER_MODELS="model-a:free,model-b:free" if you like.
FALLBACK_MODELS: tuple[str, ...] = (
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "openai/gpt-oss-20b:free",
)

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, knowledgeable and friendly AI assistant. You help with \
general-knowledge questions, coding, summarization, brainstorming, writing and \
editing, analysis, and casual conversation.

Guidelines:
- Accuracy first. If you are unsure, or something may have changed recently, \
say so plainly instead of guessing. Never invent facts, quotes, citations, \
URLs, or API details. You cannot browse the internet or see real-time \
information unless it is provided in the conversation.
- If a request is ambiguous or missing key details, ask one or two short \
clarifying questions before answering. If a reasonable assumption is possible, \
state it and proceed.
- Match depth to the question: be brief for simple questions, structured and \
thorough for complex ones.
- Coding: give working code in fenced blocks with a language tag, explain the \
approach briefly, and note assumptions and edge cases.
- Summarization: stay faithful to the source, keep key facts, and add nothing \
that is not in the text.
- Brainstorming: offer several varied ideas, then suggest the most promising \
ones and why.
- Casual chat: be warm, natural and concise.
- Use Markdown when it improves readability.
- Politely decline requests that are clearly harmful, and offer a safer \
alternative where you can.
"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _read_secret(name: str) -> str | None:
    """Return a setting from the environment, falling back to ``st.secrets``."""
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()

    try:  # Streamlit is only needed for the deployment fallback.
        import streamlit as st

        secret = st.secrets.get(name)  # Raises if no secrets file exists.
    except Exception:  # noqa: BLE001 - any failure just means "not configured".
        return None
    return str(secret).strip() if secret else None


def _get_float(name: str, default: float) -> float:
    raw = _read_secret(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = _read_secret(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # Secret. `repr=False` keeps it out of logs and accidental prints.
    api_key: str | None = field(default=None, repr=False)

    # Provider
    base_url: str = DEFAULT_BASE_URL
    default_model: str = DEFAULT_MODEL
    fallback_models: tuple[str, ...] = FALLBACK_MODELS

    # Generation
    default_temperature: float = 0.7
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # Memory: the most history (in approximate tokens) sent with each request.
    # Capped further by 75% of the model's own context window when it is known.
    max_context_tokens: int = 16_000

    # Reliability
    request_timeout: float = 60.0  # seconds per request
    max_retries: int = 4  # retries on rate limits / transient errors
    retry_base_delay: float = 2.0  # seconds; doubles each attempt
    retry_max_delay: float = 30.0  # seconds

    # Identification sent to OpenRouter (optional; shows up in their rankings)
    app_title: str = "AI Chatbot"
    app_url: str = ""

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build the settings once per process."""
    models_raw = _read_secret("OPENROUTER_MODELS")
    models = (
        tuple(m.strip() for m in models_raw.split(",") if m.strip())
        if models_raw
        else FALLBACK_MODELS
    )

    return Settings(
        api_key=_read_secret("OPENROUTER_API_KEY"),
        base_url=(_read_secret("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
        default_model=_read_secret("OPENROUTER_MODEL") or DEFAULT_MODEL,
        fallback_models=models or FALLBACK_MODELS,
        default_temperature=min(max(_get_float("DEFAULT_TEMPERATURE", 0.7), 0.0), 2.0),
        max_context_tokens=max(_get_int("MAX_CONTEXT_TOKENS", 16_000), 1_000),
        request_timeout=max(_get_float("REQUEST_TIMEOUT", 60.0), 5.0),
        max_retries=max(_get_int("MAX_RETRIES", 4), 0),
        app_title=_read_secret("APP_TITLE") or "AI Chatbot",
        app_url=_read_secret("APP_URL") or "",
    )
