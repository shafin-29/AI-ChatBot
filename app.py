from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from chat_engine import (
    ChatEngine,
    ChatEngineError,
    ChatSession,
    Done,
    KeyCheck,
    ModelInfo,
    ResponseStats,
    Retrying,
    Token,
    UnexpectedError,
    check_api_key,
    export_transcript_json,
    export_transcript_txt,
    fetch_free_models,
)
from config import Settings, get_settings

logger = logging.getLogger(__name__)

settings = get_settings()
st.set_page_config(page_title=settings.app_title, page_icon="💬", layout="centered")

CUSTOM_MODEL = "✏️ Custom model ID…"
CURSOR = "▌"


# --------------------------------------------------------------------------- #
# Cached remote lookups (cheap, and avoids hitting OpenRouter on every rerun)
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=3600, show_spinner=False)
def load_free_models(base_url: str) -> list[ModelInfo]:
    return fetch_free_models(base_url)


@st.cache_data(ttl=600, show_spinner=False)
def verify_key(base_url: str, api_key: str) -> KeyCheck:
    return check_api_key(base_url, api_key)


# --------------------------------------------------------------------------- #
# Startup validation
# --------------------------------------------------------------------------- #


def show_setup_instructions(problem: str) -> None:
    """Explain how to configure the key, then halt the script."""
    st.title("🔑 One more step: add your OpenRouter API key")
    st.error(problem)
    st.markdown(
        """
**Quick setup**

1. Create a free key at **[openrouter.ai/keys](https://openrouter.ai/keys)** (sign-up is free; no card needed for `:free` models).
2. Copy `.env.example` to `.env` in the project folder and paste your key:
"""
    )
    st.code("OPENROUTER_API_KEY=sk-or-v1-your-key-here", language="bash")
    st.markdown(
        """
3. Restart the app: `streamlit run app.py`

**Deploying (e.g. Streamlit Community Cloud)?** Add the same value under
*App settings → Secrets* instead:
"""
    )
    st.code('OPENROUTER_API_KEY = "sk-or-v1-your-key-here"', language="toml")
    st.stop()


def validate_api_key(cfg: Settings) -> KeyCheck:
    """Stop with setup instructions if the key is missing or rejected."""
    if not cfg.api_key:
        show_setup_instructions("No `OPENROUTER_API_KEY` was found in your environment, `.env` file or Streamlit secrets.")
    check = verify_key(cfg.base_url, cfg.api_key)
    if check.status == "invalid":
        show_setup_instructions("OpenRouter rejected the API key that is configured. It may be mistyped, revoked or expired.")
    return check


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #


def init_state(cfg: Settings) -> None:
    if "sessions" not in st.session_state:
        first = ChatSession()
        st.session_state.sessions = {first.id: first}
        st.session_state.active_id = first.id
    st.session_state.setdefault("system_prompt", cfg.default_system_prompt)
    st.session_state.setdefault("temperature", cfg.default_temperature)
    st.session_state.setdefault("show_stats", True)


def active_session() -> ChatSession:
    return st.session_state.sessions[st.session_state.active_id]


# Callbacks run *before* the script reruns, so the new state is visible immediately.


def cb_new_chat() -> None:
    # Re-use an empty chat instead of piling up blank ones.
    current = active_session()
    if current.messages:
        fresh = ChatSession()
        st.session_state.sessions[fresh.id] = fresh
        st.session_state.active_id = fresh.id


def cb_select_chat(session_id: str) -> None:
    st.session_state.active_id = session_id


def cb_clear_chat() -> None:
    active_session().clear()


def cb_delete_chat() -> None:
    sessions: dict[str, ChatSession] = st.session_state.sessions
    sessions.pop(st.session_state.active_id, None)
    if not sessions:
        fresh = ChatSession()
        sessions[fresh.id] = fresh
    st.session_state.active_id = list(sessions)[-1]


def cb_reset_prompt() -> None:
    st.session_state.system_prompt = settings.default_system_prompt


def cb_retry() -> None:
    session = active_session()
    session.last_error = None
    session.needs_reply = True


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunConfig:
    """Everything the sidebar decides for the next model call."""

    model: str
    temperature: float
    system_prompt: str
    context_budget: int
    show_stats: bool


def render_model_picker(cfg: Settings) -> tuple[str, int | None]:
    """Model selectbox (live free-model list, static fallback) + custom ID option."""
    catalog = {m.id: m for m in load_free_models(cfg.base_url)}
    if not catalog:
        catalog = {m: ModelInfo(id=m, name=m) for m in cfg.fallback_models}
        st.caption("⚠️ Couldn't load the live model list; showing a built-in list.")

    ids = list(catalog)
    if cfg.default_model in ids:  # Show the configured default first.
        ids.remove(cfg.default_model)
    ids.insert(0, cfg.default_model)
    options = [*ids, CUSTOM_MODEL]

    if st.session_state.get("model_choice") not in options:
        st.session_state.model_choice = cfg.default_model

    choice = st.selectbox("Model", options, key="model_choice", help="Free models are listed. Availability changes often.")
    if choice == CUSTOM_MODEL:
        model = st.text_input("Custom model ID", key="custom_model", placeholder="provider/model-name:free").strip()
        model = model or cfg.default_model
    else:
        model = choice

    info = catalog.get(model)
    return model, (info.context_length if info else None)


def render_sidebar(cfg: Settings, key_check: KeyCheck) -> RunConfig:
    with st.sidebar:
        st.header(f"💬 {cfg.app_title}")
        if key_check.status == "valid":
            st.caption(f"✅ API key verified{f'' if key_check.label else ''}")
        else:
            st.caption("⚠️ Couldn't verify the API key (offline?). Continuing anyway.")

        # -- chats ---------------------------------------------------------- #
        st.button("➕ New chat", on_click=cb_new_chat, width="stretch")
        sessions: dict[str, ChatSession] = st.session_state.sessions
        for sid, chat in reversed(list(sessions.items())):
            st.button(
                chat.title,
                key=f"chat_{sid}",
                on_click=cb_select_chat,
                args=(sid,),
                type="primary" if sid == st.session_state.active_id else "secondary",
                width="stretch",
            )

        st.divider()

        # -- model settings -------------------------------------------------- #
        model, context_length = render_model_picker(cfg)
        st.slider("Temperature", 0.0, 2.0, step=0.05, key="temperature", help="Lower = focused and deterministic. Higher = more creative.")

        with st.expander("System prompt"):
            st.text_area("Instructions for the assistant", key="system_prompt", height=260, label_visibility="collapsed")
            st.button("Reset to default", on_click=cb_reset_prompt)

        st.checkbox("Show response stats", key="show_stats")

        st.divider()

        # -- conversation actions ------------------------------------------- #
        chat = active_session()
        col_clear, col_delete = st.columns(2)
        col_clear.button("🧹 Clear", on_click=cb_clear_chat, disabled=not chat.messages, width="stretch", help="Remove all messages in this chat")
        col_delete.button("🗑️ Delete", on_click=cb_delete_chat, width="stretch", help="Delete this chat")

        st.download_button(
            "⬇️ Transcript (.txt)",
            data=export_transcript_txt(chat),
            file_name=f"chat-{chat.id}.txt",
            mime="text/plain",
            disabled=not chat.messages,
            width="stretch",
        )
        st.download_button(
            "⬇️ Transcript (.json)",
            data=export_transcript_json(chat),
            file_name=f"chat-{chat.id}.json",
            mime="application/json",
            disabled=not chat.messages,
            width="stretch",
        )

    # Keep ~25% of the window free for the answer and for token-count inaccuracy.
    budget = cfg.max_context_tokens
    if context_length:
        budget = min(budget, int(context_length * 0.75))

    return RunConfig(
        model=model,
        temperature=float(st.session_state.temperature),
        system_prompt=st.session_state.system_prompt,
        context_budget=budget,
        show_stats=bool(st.session_state.show_stats),
    )


# --------------------------------------------------------------------------- #
# Chat rendering
# --------------------------------------------------------------------------- #


def format_stats(stats: dict[str, Any]) -> str:
    """One-line summary under an assistant reply."""
    approx = "~" if stats.get("approximate") else ""
    bits = [f"⏱ {stats['elapsed_s']:.1f}s"]
    if stats.get("first_token_s") is not None:
        bits.append(f"first token {stats['first_token_s']:.1f}s")
    bits.append(f"{approx}{stats['input_tokens']:,} in / {approx}{stats['output_tokens']:,} out tokens")
    if stats["output_tokens"] and stats.get("first_token_s") is not None:
        generation = stats["elapsed_s"] - stats["first_token_s"]
        if generation > 0.05:
            bits.append(f"{approx}{stats['output_tokens'] / generation:.0f} tok/s")
    bits.append(f"`{stats['model']}`")
    return " · ".join(bits)


def render_stats(stats: dict[str, Any] | None, show: bool) -> None:
    if not stats:
        return
    if stats.get("trimmed"):
        st.caption(f"✂️ {stats['trimmed']} older message(s) were left out of the model's context to fit its window.")
    if show:
        st.caption(format_stats(stats))


def render_error(session: ChatSession) -> None:
    """Show the last error, with a Retry button when a question is unanswered."""
    if not session.last_error:
        return
    with st.chat_message("assistant", avatar="⚠️"):
        st.error(session.last_error)
        if session.awaiting_reply:
            st.button("🔄 Retry", key=f"retry_{session.id}", on_click=cb_retry)


def render_history(session: ChatSession, show_stats: bool) -> None:
    if not session.messages and not session.last_error:
        st.info("👋 Ask me anything: general knowledge, code, summaries, brainstorming, or just chat.")
    for message in session.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
                render_stats(message.response_metadata.get("stats"), show_stats)
    render_error(session)


# --------------------------------------------------------------------------- #
# Generating a reply
# --------------------------------------------------------------------------- #


def generate_reply(session: ChatSession, cfg: Settings, run: RunConfig) -> None:
    """Stream the model's reply into the page and store it in the session."""
    if not session.awaiting_reply:
        session.needs_reply = False
        return

    text = ""
    final_stats: ResponseStats | None = None
    error: ChatEngineError | None = None

    with st.chat_message("assistant"):
        notice = st.empty()  # Retry notices
        output = st.empty()  # The streaming answer
        output.markdown(f"_Thinking…_ {CURSOR}")  # Visible until the first token arrives.

        try:
            engine = ChatEngine(
                cfg,
                model=run.model,
                temperature=run.temperature,
                system_prompt=run.system_prompt,
                context_budget=run.context_budget,
            )
            for event in engine.stream(session.messages):
                if isinstance(event, Token):
                    notice.empty()
                    text += event.text
                    output.markdown(text + CURSOR)
                elif isinstance(event, Retrying):
                    text = ""  # A retry starts the answer over.
                    output.markdown(f"_Waiting to retry…_ {CURSOR}")
                    notice.warning(f"⏳ {event.reason}\n\nRetrying in {event.delay_s:.0f}s (attempt {event.attempt} of {event.max_attempts})…")
                elif isinstance(event, Done):
                    final_stats = event.stats
        except ChatEngineError as exc:
            error = exc
        except Exception as exc:  # noqa: BLE001 - never show a stack trace to the user.
            logger.exception("Unhandled error while generating a reply")
            error = UnexpectedError("Something unexpected went wrong. Please try again.", detail=str(exc))

        notice.empty()
        if error is None:
            output.markdown(text)
            session.add_assistant_message(text, final_stats)
            render_stats(final_stats.to_dict() if final_stats else None, run.show_stats)
            return

        # Failure path: keep any partial answer so work isn't lost.
        if text.strip():
            output.markdown(text)
            session.add_assistant_message(text)  # No stats for a partial reply.
            session.last_error = f"The response was cut short. {error.user_message}"
        else:
            output.empty()
            session.last_error = error.user_message
        session.needs_reply = False

    render_error(session)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    key_check = validate_api_key(settings)  # Stops with instructions if not OK.
    init_state(settings)

    # Read the input *before* drawing the sidebar so a new chat's title appears
    # in the chat list on this very run. (The input box stays pinned to the bottom.)
    prompt = st.chat_input("Message the assistant…")
    if prompt and prompt.strip():
        active_session().add_user_message(prompt.strip())

    run = render_sidebar(settings, key_check)
    session = active_session()

    st.title(f"💬 {session.title}")
    st.caption(f"Model: `{run.model}` · Temperature: {run.temperature:.2f}")

    render_history(session, run.show_stats)
    if session.needs_reply:
        generate_reply(session, settings, run)


main()
