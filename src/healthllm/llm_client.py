import os
from langchain_ollama import ChatOllama


def get_ollama_llm(
    model=None,
    base_url=None,
    temperature=0,
):
    model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
    base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
    )

