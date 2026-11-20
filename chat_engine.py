from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterator, Literal, Sequence

import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_openai import ChatOpenAI

from config import Settings

logger = logging.getLogger(__name__)

# History is never trimmed below this many tokens (the newest message is always kept).
MIN_HISTORY_TOKENS = 512


# =========================================================================== #
# Errors - friendly, UI-safe messages instead of stack traces
# =========================================================================== #


class ChatEngineError(Exception):
    """Base class. ``user_message`` is safe to show directly in the UI."""

    retryable: bool = False
    short_reason: str = "Temporary error."  # Shown in the "retrying…" banner.

    def __init__(self, user_message: str, *, detail: str | None = None) -> None:
        super().__init__(detail or user_message)
        self.user_message = user_message


class MissingApiKeyError(ChatEngineError):
    pass


class AuthError(ChatEngineError):
    pass


class CreditsError(ChatEngineError):
    pass


class ModelUnavailableError(ChatEngineError):
    pass


class InvalidRequestError(ChatEngineError):
    pass


class UnexpectedError(ChatEngineError):
    pass


class RateLimitedError(ChatEngineError):
    """HTTP 429. Retried with backoff unless a *daily* quota is exhausted."""

    short_reason = "The model is rate-limited."

    def __init__(
        self,
        user_message: str,
        *,
        retry_after: float | None = None,
        daily: bool = False,
        detail: str | None = None,
    ) -> None:
        super().__init__(user_message, detail=detail)
        self.retry_after = retry_after
        self.retryable = not daily  # Waiting a few seconds won't fix a daily cap.


class NetworkError(ChatEngineError):
    retryable = True
    short_reason = "Network problem or timeout."


class ServiceError(ChatEngineError):
    retryable = True
    short_reason = "The model provider had a hiccup."


class EmptyResponseError(ChatEngineError):
    retryable = True
    short_reason = "The model returned an empty response."


# =========================================================================== #
# Provider boundary #1: building the LLM
# =========================================================================== #


def build_llm(settings: Settings, model: str, temperature: float) -> BaseChatModel:
    """Create a streaming-capable LangChain chat model for OpenRouter.

    OpenRouter exposes an OpenAI-compatible API, so ``ChatOpenAI`` with a custom
    ``base_url`` is all that is needed. To use another provider, return a
    different ``BaseChatModel`` here (e.g. ``ChatAnthropic``, ``ChatOllama``).
    """
    headers = {"X-Title": settings.app_title}
    if settings.app_url:
        headers["HTTP-Referer"] = settings.app_url

    return ChatOpenAI(
        model=model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=temperature,
        timeout=settings.request_timeout,
        max_retries=0,  # We implement our own backoff (see ChatEngine.stream).
        default_headers=headers,
        stream_usage=True,  # Ask for token usage in the final stream chunk.
    )


# =========================================================================== #
# Provider boundary #2: translating SDK exceptions
# =========================================================================== #


def _provider_message(exc: Exception) -> str:
    return str(getattr(exc, "message", None) or exc)


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    try:
        value = response.headers.get("retry-after") if response is not None else None
        return float(value) if value is not None else None
    except (ValueError, AttributeError):
        return None


def classify_exception(exc: Exception) -> ChatEngineError:
    """Convert any exception raised while calling the LLM into a ChatEngineError."""
    if isinstance(exc, ChatEngineError):
        return exc

    detail = _provider_message(exc)
    lowered = detail.lower()

    # Order matters: APITimeoutError is a subclass of APIConnectionError.
    if isinstance(exc, openai.APITimeoutError | TimeoutError):
        return NetworkError(
            "The request timed out. The model may be busy - please try again.",
            detail=detail,
        )
    if isinstance(exc, openai.APIConnectionError | ConnectionError):
        return NetworkError(
            "Couldn't reach OpenRouter. Check your internet connection and try again.",
            detail=detail,
        )
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return AuthError(
            "OpenRouter rejected your API key. Check OPENROUTER_API_KEY in your "
            ".env file (or Streamlit secrets) and restart the app.",
            detail=detail,
        )
    if isinstance(exc, openai.RateLimitError):
        daily = "per-day" in lowered or "per day" in lowered or "daily" in lowered
        if daily:
            msg = (
                "You've hit the daily limit for free models. Try again tomorrow, "
                "pick a different model, or add credits to your OpenRouter account."
            )
        else:
            msg = (
                "The model is rate-limited right now (common on free tiers). "
                "Wait a moment and retry, or choose another model."
            )
        return RateLimitedError(msg, retry_after=_retry_after(exc), daily=daily, detail=detail)
    if isinstance(exc, openai.NotFoundError):
        return ModelUnavailableError(
            "This model isn't available right now. It may have been retired, or your "
            "OpenRouter privacy settings may block free endpoints. Pick another model "
            "(see the README troubleshooting section).",
            detail=detail,
        )
    if isinstance(exc, openai.BadRequestError):
        return InvalidRequestError(
            f"The model rejected the request: {detail}. If the conversation is very "
            "long, start a new chat or lower MAX_CONTEXT_TOKENS.",
            detail=detail,
        )
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        if status == 402:
            return CreditsError(
                "OpenRouter says this request needs credits. Choose a ':free' model "
                "or add credits to your account.",
                detail=detail,
            )
        if status == 408 or status >= 500:
            return ServiceError(
                "The model provider had a temporary problem. Please try again.",
                detail=detail,
            )
        return UnexpectedError(f"Unexpected API error ({status}): {detail}", detail=detail)
    if isinstance(exc, openai.APIError):
        # Errors delivered inside the stream have no HTTP status, only a `code`.
        code = str(getattr(exc, "code", "") or "")
        if code in {"429", "rate_limit_exceeded"}:
            return RateLimitedError(
                "The model is rate-limited right now. Please retry shortly.", detail=detail
            )
        if code.startswith("5"):
            return ServiceError(
                "The model provider had a temporary problem. Please try again.", detail=detail
            )
        return UnexpectedError(f"The model returned an error: {detail}", detail=detail)

    logger.exception("Unexpected error while calling the LLM", exc_info=exc)
    return UnexpectedError(
        "Something unexpected went wrong while generating a response. Please try again.",
        detail=detail,
    )


# =========================================================================== #
# Streaming events and statistics
# =========================================================================== #


@dataclass(frozen=True)
class ResponseStats:
    """Timing and token usage for one completed response."""

    model: str
    elapsed_s: float  # Request start -> last token (includes retry waits).
    first_token_s: float | None  # Time to first token.
    input_tokens: int
    output_tokens: int
    approximate: bool  # True when token counts are estimates, not provider-reported.
    trimmed: int  # Older messages left out of the context window.

    @property
    def tokens_per_second(self) -> float | None:
        if self.first_token_s is None:
            return None
        generation_time = self.elapsed_s - self.first_token_s
        return self.output_tokens / generation_time if generation_time > 0.05 else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Token:
    """A piece of generated text."""

    text: str


@dataclass(frozen=True)
class Retrying:
    """Emitted before sleeping for a retry. Consumers should discard partial text."""

    attempt: int
    max_attempts: int
    delay_s: float
    reason: str


@dataclass(frozen=True)
class Done:
    """Emitted once, after the final token."""

    stats: ResponseStats


StreamEvent = Token | Retrying | Done


def _chunk_text(chunk: BaseMessage) -> str:
    """Extract text from a streamed chunk (content may be a str or content blocks)."""
    content = chunk.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


# =========================================================================== #
# The engine
# =========================================================================== #


class ChatEngine:
    """Streams model replies for a conversation, with trimming and retries."""

    def __init__(
        self,
        settings: Settings,
        *,
        model: str,
        temperature: float,
        system_prompt: str,
        context_budget: int | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        if llm is None and not settings.api_key:
            raise MissingApiKeyError(
                "No API key configured. Set OPENROUTER_API_KEY in your .env file."
            )
        self.settings = settings
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt.strip()
        self.context_budget = context_budget or settings.max_context_tokens
        # `llm` can be injected for tests or to swap providers without build_llm().
        self._llm: BaseChatModel = llm or build_llm(settings, model, temperature)

    # -- memory ------------------------------------------------------------- #

    def prepare_messages(self, history: Sequence[BaseMessage]) -> tuple[list[BaseMessage], int]:
        """Return ``(messages_to_send, number_of_dropped_messages)``.

        The system prompt is prepended on every call (so edits apply immediately)
        and is never trimmed. The oldest turns are dropped first, and the window
        always starts on a human message so the model sees a coherent exchange.
        """
        system = SystemMessage(content=self.system_prompt) if self.system_prompt else None
        reserved = count_tokens_approximately([system]) if system else 0
        budget = max(self.context_budget - reserved, MIN_HISTORY_TOKENS)

        history = list(history)
        trimmed = trim_messages(
            history,
            max_tokens=budget,
            token_counter=count_tokens_approximately,
            strategy="last",
            start_on="human",
            allow_partial=False,
        )
        if not trimmed and history:
            # The newest message alone exceeds the budget: send it anyway and let
            # the provider decide, rather than silently sending no question at all.
            trimmed = history[-1:]

        dropped = len(history) - len(trimmed)
        return ([system] if system else []) + trimmed, dropped

    # -- streaming ---------------------------------------------------------- #

    def _backoff_delay(self, attempt: int, error: ChatEngineError) -> float:
        """Exponential backoff with jitter, honouring a Retry-After hint."""
        s = self.settings
        delay = min(s.retry_base_delay * (2**attempt), s.retry_max_delay)
        if isinstance(error, RateLimitedError) and error.retry_after:
            delay = max(delay, error.retry_after)
        delay = min(delay, s.retry_max_delay)
        return delay + random.uniform(0, delay * 0.25)

    def stream(self, history: Sequence[BaseMessage]) -> Iterator[StreamEvent]:
        """Yield ``Token`` events, then a single ``Done``.

        Retryable failures (rate limits, timeouts, 5xx, empty replies) are retried
        with exponential backoff **only if no text has been produced yet** -
        retrying mid-answer would duplicate output. Raises ``ChatEngineError``.
        """
        messages, dropped = self.prepare_messages(history)
        estimated_input = count_tokens_approximately(messages)
        started = time.perf_counter()
        max_retries = self.settings.max_retries
        attempt = 0

        while True:
            parts: list[str] = []
            first_token_at: float | None = None
            usage: dict[str, Any] | None = None

            try:
                for chunk in self._llm.stream(messages):
                    if getattr(chunk, "usage_metadata", None):
                        usage = dict(chunk.usage_metadata)  # type: ignore[arg-type]
                    text = _chunk_text(chunk)
                    if not text:
                        continue
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    parts.append(text)
                    yield Token(text)

                if not "".join(parts).strip():
                    raise EmptyResponseError(
                        "The model returned an empty response. Please try again or "
                        "pick a different model."
                    )
            except Exception as exc:  # noqa: BLE001 - classified below.
                error = classify_exception(exc)
                produced_text = bool("".join(parts).strip())
                if error.retryable and not produced_text and attempt < max_retries:
                    delay = self._backoff_delay(attempt, error)
                    attempt += 1
                    logger.warning(
                        "Retry %d/%d in %.1fs after: %s", attempt, max_retries, delay, error
                    )
                    yield Retrying(attempt, max_retries, delay, error.short_reason)
                    time.sleep(delay)
                    continue
                raise error from exc
            break  # Success.

        finished = time.perf_counter()
        answer = "".join(parts)
        if usage and usage.get("output_tokens") is not None:
            input_tokens = int(usage.get("input_tokens", estimated_input))
            output_tokens = int(usage["output_tokens"])
            approximate = False
        else:  # Some free providers don't report usage while streaming.
            input_tokens = estimated_input
            output_tokens = count_tokens_approximately([AIMessage(content=answer)])
            approximate = True

        yield Done(
            ResponseStats(
                model=self.model,
                elapsed_s=finished - started,
                first_token_s=(first_token_at - started) if first_token_at else None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                approximate=approximate,
                trimmed=dropped,
            )
        )


# =========================================================================== #
# Conversation memory: sessions
# =========================================================================== #

DEFAULT_TITLE = "New chat"


def _make_title(text: str, limit: int = 36) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1].rstrip() + "…"


@dataclass
class ChatSession:
    """One conversation: LangChain message objects plus a little UI state."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str = DEFAULT_TITLE
    messages: list[BaseMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    needs_reply: bool = False  # A user message is waiting for the model.
    last_error: str | None = None

    @property
    def awaiting_reply(self) -> bool:
        return bool(self.messages) and isinstance(self.messages[-1], HumanMessage)

    def add_user_message(self, text: str) -> None:
        self.messages.append(HumanMessage(content=text))
        if self.title == DEFAULT_TITLE:
            self.title = _make_title(text)
        self.needs_reply = True
        self.last_error = None

    def add_assistant_message(self, text: str, stats: ResponseStats | None = None) -> None:
        metadata = {"stats": stats.to_dict()} if stats else {}
        self.messages.append(AIMessage(content=text, response_metadata=metadata))
        self.needs_reply = False

    def clear(self) -> None:
        self.messages.clear()
        self.title = DEFAULT_TITLE
        self.needs_reply = False
        self.last_error = None


# =========================================================================== #
# Transcript export
# =========================================================================== #


def _role(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    return "system"


def export_transcript_txt(session: ChatSession) -> str:
    """Plain-text transcript."""
    lines = [f"{session.title}", f"Exported {datetime.now():%Y-%m-%d %H:%M}", "=" * 60, ""]
    for message in session.messages:
        speaker = "You" if _role(message) == "user" else "Assistant"
        lines += [f"{speaker}:", str(message.content), ""]
    return "\n".join(lines)


def export_transcript_json(session: ChatSession) -> str:
    """JSON transcript, including per-reply stats where available."""
    payload = {
        "title": session.title,
        "created_at": session.created_at.isoformat(timespec="seconds"),
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "messages": [
            {
                "role": _role(m),
                "content": m.content,
                **(
                    {"stats": m.response_metadata["stats"]}
                    if _role(m) == "assistant" and m.response_metadata.get("stats")
                    else {}
                ),
            }
            for m in session.messages
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# =========================================================================== #
# OpenRouter helpers (optional; replace or delete when swapping providers)
# =========================================================================== #

_USER_AGENT = "streamlit-langchain-chatbot/1.0"


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    context_length: int | None = None


@dataclass(frozen=True)
class KeyCheck:
    status: Literal["valid", "invalid", "unknown"]
    label: str | None = None  # Masked key label, e.g. "sk-or-v1-abc...xyz"


def _get_json(url: str, *, api_key: str | None, timeout: float) -> tuple[int, Any]:
    """GET a JSON document. Returns (status_code, body). Raises on network failure."""
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status, json.load(response)
    except urllib.error.HTTPError as err:
        return err.code, None


def fetch_free_models(base_url: str, timeout: float = 8.0) -> list[ModelInfo]:
    """List the currently-free text models on OpenRouter. Returns ``[]`` on failure."""
    try:
        status, body = _get_json(f"{base_url}/models", api_key=None, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - the caller falls back to a static list.
        logger.warning("Could not fetch model list: %s", exc)
        return []
    if status != 200 or not isinstance(body, dict):
        return []

    def is_free(entry: dict[str, Any]) -> bool:
        pricing = entry.get("pricing") or {}
        try:
            zero_priced = float(pricing["prompt"]) == 0 and float(pricing["completion"]) == 0
        except (KeyError, TypeError, ValueError):
            zero_priced = False
        return str(entry.get("id", "")).endswith(":free") or zero_priced

    models = []
    for entry in body.get("data", []):
        outputs = (entry.get("architecture") or {}).get("output_modalities") or ["text"]
        if not is_free(entry) or "text" not in outputs:
            continue
        models.append(
            ModelInfo(
                id=entry["id"],
                name=entry.get("name") or entry["id"],
                context_length=entry.get("context_length"),
            )
        )
    return sorted(models, key=lambda m: m.id)


def check_api_key(base_url: str, api_key: str, timeout: float = 8.0) -> KeyCheck:
    """Ask OpenRouter whether the key is accepted (GET /key).

    Only an explicit 401 counts as "invalid"; network trouble is "unknown" so an
    offline start-up doesn't block the app.
    """
    try:
        status, body = _get_json(f"{base_url}/key", api_key=api_key, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not verify API key: %s", exc)
        return KeyCheck("unknown")
    if status == 200:
        label = ((body or {}).get("data") or {}).get("label")
        return KeyCheck("valid", label)
    if status == 401:
        return KeyCheck("invalid")
    return KeyCheck("unknown")
