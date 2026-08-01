import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# Default model configuration
# ---------------------------------------------------------------------------

GROQ_DEFAULT_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
)

GEMINI_DEFAULT_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
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
