import os
from langchain_groq import ChatGroq

MODEL_NAME = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

def create_llm(temperature=0.3):
    return ChatGroq(
        model=MODEL_NAME,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
    )
