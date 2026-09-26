import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# Default model configuration
# ---------------------------------------------------------------------------

GROQ_DEFAULT_MODEL = (
    os.getenv("GROQ_MODEL")
    or os.getenv("LLM_MODEL")
    or "openai/gpt-oss-120b"
)

GROQ_SMALL_MODEL = os.getenv(
    "GROQ_SMALL_MODEL",
    "openai/gpt-oss-20b",
)

GEMINI_DEFAULT_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-flash-latest",
)

# ---------------------------------------------------------------------------
# Provider factories
# ---------------------------------------------------------------------------


def create_groq(
    model: str | None = None,
    temperature: float = 0,
):
    """
    Create a Groq LLM client.

    Used primarily for:
    - Conversational RAG
    - Reasoning-heavy tasks
    """

    return ChatGroq(
        model=model or GROQ_DEFAULT_MODEL,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
        timeout=60,
    )


def create_gemini(
    model: str | None = None,
    temperature: float = 0,
):
    """
    Create a Gemini LLM client.

    Used primarily for:
    - Meeting title generation
    - Long transcript summarization
    - Structured insight extraction
    """

    return ChatGoogleGenerativeAI(
        model=model or GEMINI_DEFAULT_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,
        timeout=60,
    )


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def create_llm(
    model: str | None = None,
    temperature: float = 0,
):
    """
    Existing codebase compatibility.

    Currently returns the Groq client so existing modules continue
    to work without modification.
    """

    return create_groq(
        model=model,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Task-scoped provider selection
# ---------------------------------------------------------------------------
#
# Summarization and insight extraction each pick their LLM provider via an
# environment variable ("SUMMARY_PROVIDER" / "INSIGHTS_PROVIDER"), instead of
# hardcoding create_gemini()/create_groq() at each call site. Groq is the
# default for both — Gemini's free-tier quota is far lower and was causing
# RESOURCE_EXHAUSTED (429) errors on longer meetings, which need several
# chunked LLM calls per pipeline run. Gemini remains fully supported by
# setting the relevant variable to "gemini".


def _create_llm_for_provider(provider: str, temperature: float):
    if provider == "gemini":
        return create_gemini(temperature=temperature)
    return create_groq(temperature=temperature)


def create_summary_llm(temperature: float = 0):
    """Create the LLM used by core/transcript_summary.py.

    Provider is chosen via the SUMMARY_PROVIDER env var ("groq" or
    "gemini"), defaulting to "groq".
    """
    provider = os.getenv("SUMMARY_PROVIDER", "groq").strip().lower()
    return _create_llm_for_provider(provider, temperature)


def create_insights_llm(temperature: float = 0):
    """Create the LLM used by core/transcript_insights.py.

    Provider is chosen via the INSIGHTS_PROVIDER env var ("groq" or
    "gemini"), defaulting to "groq".
    """
    provider = os.getenv("INSIGHTS_PROVIDER", "groq").strip().lower()
    return _create_llm_for_provider(provider, temperature)


# ---------------------------------------------------------------------------
# Shared LLM error handling
# ---------------------------------------------------------------------------


class LLMServiceError(Exception):
    """User-facing error raised when an LLM provider call fails in a way
    that should not surface a raw SDK stack trace to the UI (e.g. a
    rate-limit / quota error from Groq or Gemini).

    Shared by core/transcript_summary.py and core/transcript_insights.py so
    both modules raise the same exception type regardless of which provider
    (Groq or Gemini) is currently configured — callers (e.g.
    streamlit_app.py's render_processing) only need to display str(exc).
    """


def get_groq_error_message(exc: Exception) -> str:
    """Return a safe, user-facing error message for Groq API errors
    without exposing API keys or secrets.
    """
    status_code = getattr(exc, "status_code", None)
    exc_str = str(exc).lower()

    # 404: Model not found or retired
    if (
        status_code == 404
        or "model_not_found" in exc_str
        or "does not exist" in exc_str
    ):
        return (
            "The configured Groq model is not available or has been retired. "
            "Please check GROQ_MODEL in your environment settings."
        )

    # 401 / 403: Authentication or permission errors
    if (
        status_code in (401, 403)
        or "invalid_api_key" in exc_str
        or "unauthorized" in exc_str
        or "permission" in exc_str
    ):
        return (
            "Groq authentication failed. Please verify your GROQ_API_KEY "
            "in your environment settings."
        )

    # 429: Rate limit
    if status_code == 429 or "rate_limit" in exc_str or "rate limit" in exc_str:
        return (
            "The AI service is temporarily busy. Please wait a minute and "
            "try again."
        )

    # 5xx / Network / Connection / Timeout errors
    if (
        (status_code is not None and status_code >= 500)
        or "connection" in exc_str
        or "timeout" in exc_str
        or "network" in exc_str
    ):
        return (
            "The AI service encountered a temporary network or server error. "
            "Please wait a moment and try again."
        )

    # Fallback for any other Groq API errors
    return (
        "The AI service is temporarily busy. Please wait a minute and "
        "try again."
    )

