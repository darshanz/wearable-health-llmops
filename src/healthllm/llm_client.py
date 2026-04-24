from langchain_ollama import ChatOllama


def get_ollama_llm(
    model="qwen2.5:1.5b",
    base_url="http://localhost:11434",
    temperature=0,
):
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
    )
