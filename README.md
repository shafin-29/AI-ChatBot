# AI Chatbot: Streamlit + LangChain + OpenRouter (free models)

A streaming chatbot with conversation memory, multiple chat sessions, retry/backoff for
rate limits, and friendly error messages. It talks to [OpenRouter](https://openrouter.ai)
through LangChain's `ChatOpenAI`, defaulting to the free
`meta-llama/llama-3.3-70b-instruct:free` model.

## Features

- **Streaming** responses with a "thinking" indicator and a live cursor
- **Memory** via `st.session_state` and LangChain `HumanMessage` / `AIMessage` / `SystemMessage` objects; the oldest turns are trimmed automatically when the history nears the model's context window
- **Multiple chats** in the sidebar (new, switch, clear, delete)
- **Sidebar controls:** model picker (live list of currently-free models, plus a custom-ID option), temperature slider, system prompt editor with reset
- **Resilient:** exponential backoff with jitter on rate limits, timeouts, 5xx errors and empty replies; friendly messages instead of stack traces; a **Retry** button after a failure
- **Startup validation:** missing or rejected API key shows setup instructions instead of a broken UI
- **Response stats:** time to first token, total time, token counts, tokens/second
- **Transcript download** as `.txt` or `.json`

## Project structure

```
chatbot/
├── app.py            # Streamlit UI (presentation only)
├── chat_engine.py    # LangChain logic: LLM setup, memory/trimming, streaming, retries, errors
├── config.py         # Settings + environment loading (.env, then st.secrets)
├── requirements.txt  # Pinned dependencies
├── .env.example      # Template for your local secrets
├── .gitignore        # Keeps .env out of git
└── README.md
```

## 1. Get a free OpenRouter API key

1. Go to **<https://openrouter.ai>** and sign up (Google / GitHub / email).
2. Open **<https://openrouter.ai/keys>** and click **Create Key**. Give it any name.
3. Copy the key (it starts with `sk-or-v1-`). You won't be able to view it again.

Models whose ID ends in `:free` cost nothing. No payment card is required, but they are
rate-limited (see [Troubleshooting](#troubleshooting)).

> **Free-model privacy setting:** some free endpoints are only available if you allow them
> in your OpenRouter privacy settings (<https://openrouter.ai/settings/privacy>). If you get
> a "model isn't available" error, check that page.

## 2. Set up

Requires **Python 3.10+**. Check with `python --version` (use `python3 --version` on macOS/Linux).

Pick the instructions for your shell. Run them from inside the `chatbot` folder.

### Windows (PowerShell)

```powershell
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt

# Create your .env file, then open it and paste your key
Copy-Item .env.example .env
notepad .env
```

> **"Running scripts is disabled on this system"?** Allow scripts for the current window only, then activate again:
>
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .venv\Scripts\Activate.ps1
> ```
>
> Note: PowerShell versions before 7 don't support `&&` between commands. Run each command on its own line.

### Windows (Command Prompt)

```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
notepad .env
```

### macOS / Linux / Git Bash (bash or zsh)

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt

# Create your .env file, then edit it and paste your key
cp .env.example .env
nano .env
```

> **Git Bash on Windows:** activate with `source .venv/Scripts/activate` (note `Scripts`, not `bin`).

### Conda (any OS)

If you prefer conda, use it instead of `venv`:

```bash
conda create -n chatbot python=3.12
conda activate chatbot
pip install -r requirements.txt
```

Then create `.env` using the command for your shell above.

### What goes in `.env`

Set your key on one line, with **no quotes and no spaces** around the `=`:

```
OPENROUTER_API_KEY=your-key-here
```

Save the file as exactly `.env`. On Windows.

## 3. Run

With the virtual environment activated (your prompt shows `(.venv)`), the command is the same in every shell:

```
streamlit run app.py
```

Streamlit opens <http://localhost:8501> in your browser. Press `Ctrl+C` in the terminal to stop it.

**Without activating the environment** (handy if activation is blocked):

```powershell
# Windows
.venv\Scripts\python -m streamlit run app.py
```

```bash
# macOS / Linux
.venv/bin/python -m streamlit run app.py
```

**Next time:** you only need to activate the environment and run `streamlit run app.py`. Don't repeat the install or `.env` steps.

## Configuration

All settings are read from environment variables / `.env` first, then from Streamlit
secrets. Only `OPENROUTER_API_KEY` is required.

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | *(required)* | Your OpenRouter key |
| `OPENROUTER_MODEL` | `meta-llama/llama-3.3-70b-instruct:free` | Model selected at startup |
| `OPENROUTER_MODELS` | built-in list | Comma-separated fallback list if the live free-model list can't be fetched |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | API endpoint |
| `DEFAULT_TEMPERATURE` | `0.7` | Initial slider value (0-2) |
| `MAX_CONTEXT_TOKENS` | `16000` | Max history tokens sent per request (also capped at 75% of the model's context window when known) |
| `REQUEST_TIMEOUT` | `60` | Seconds to wait per request |
| `MAX_RETRIES` | `4` | Retries for rate limits / transient errors (delays: ~2s, 4s, 8s, 16s + jitter) |
| `APP_TITLE`, `APP_URL` | `AI Chatbot`, *(empty)* | Optional identification sent to OpenRouter |

### Deploying (e.g. Streamlit Community Cloud)

Don't upload `.env`. In your app's **Settings → Secrets**, add:

```toml
OPENROUTER_API_KEY = "sk-or-v1-your-key-here"
```

For local testing of this path you can instead create `.streamlit/secrets.toml` (already in `.gitignore`).

## How it works

- **`config.py`** loads `.env` via `python-dotenv`, falls back to `st.secrets`, and exposes an immutable `Settings` object. The key is excluded from `repr()` so it can't leak into logs.
- **`chat_engine.py`** has no Streamlit dependency.
  - `build_llm()` creates `ChatOpenAI(base_url="https://openrouter.ai/api/v1", ...)`.
  - `ChatEngine.prepare_messages()` prepends the system prompt, then uses LangChain's `trim_messages` to drop the *oldest* turns when over budget. The window always starts on a user message, and the newest message is never dropped.
  - `ChatEngine.stream()` yields typed events (`Token`, `Retrying`, `Done`). It retries with exponential backoff only if no text has been produced yet, since retrying mid-answer would duplicate output.
  - `classify_exception()` turns SDK errors (401, 402, 404, 429, timeouts, 5xx...) into `ChatEngineError` subclasses carrying a UI-safe message.
- **`app.py`** renders the chat with `st.chat_message` / `st.chat_input`, runs the stream loop, and stores replies as `AIMessage`s (stats ride along in `response_metadata`, which is never sent to the model).

### Swapping the LLM provider

Only two functions are provider-specific:

1. **`build_llm()`**: return any LangChain `BaseChatModel` (e.g. `ChatAnthropic`, `ChatOllama`, `ChatGoogleGenerativeAI`).
2. **`classify_exception()`**: map that SDK's exceptions to the engine's error classes.

The OpenRouter helpers at the bottom of `chat_engine.py` (`fetch_free_models()` and `check_api_key()`) are optional conveniences. For another provider, make them return `[]` and `KeyCheck("unknown")` respectively; the app then uses the `OPENROUTER_MODELS` list and skips key verification.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Setup screen: "No `OPENROUTER_API_KEY` was found" | `.env` is missing, misnamed (must be exactly `.env`), or in the wrong folder. It must sit next to `app.py`. Restart after editing. |
| "OpenRouter rejected your API key" (401) | Key mistyped, revoked, or has stray quotes/spaces. Create a new key at openrouter.ai/keys. |
| "The model is rate-limited…" (429) | Free models have per-minute and per-day request caps. The app retries automatically with backoff; if it still fails, wait a minute, pick a different `:free` model, or add credits to your OpenRouter account. |
| "daily limit for free models" | The daily free quota is used up. It resets daily, or add credits to raise it. |
| "This model isn't available right now" (404) | The free model was retired or temporarily has no provider, or your privacy settings block free endpoints (see the note above). Choose another model from the sidebar. |
| "needs credits" (402) | You selected a paid model. Use a `:free` model or top up. |
| "request timed out" / "Couldn't reach OpenRouter" | Network issue, a firewall/proxy, or a busy free model. Try again; raise `REQUEST_TIMEOUT` for slow models. |
| Empty response | Free models occasionally return nothing. The app retries automatically; if it persists, switch models. |
| "Couldn't load the live model list" | Informational: the app is offline or OpenRouter is unreachable and is using the built-in list. |
| Chats disappear on browser refresh | Chats live in `st.session_state`, which is per browser session. Use the transcript download to keep a copy. |
| Very long chat starts failing (400) | Lower `MAX_CONTEXT_TOKENS`, or start a new chat. |

> Free model availability and limits change frequently. See OpenRouter's
> [model list](https://openrouter.ai/models?q=free) and rate-limit documentation for current details.
