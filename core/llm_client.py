from langchain_groq import ChatGroq
import os

DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


def create_llm(temperature=0):
    return ChatGroq(
        model=model or DEFAULT_MODEL,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
    )
