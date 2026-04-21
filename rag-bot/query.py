import os
import argparse
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from index import get_embeddings, CHROMA_DB_DIR

load_dotenv()

# Suppress ChromaDB telemetry noise
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# Model preference order - try newest models first (separate quota buckets)
GEMINI_MODEL_PREFERENCE = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro",
]

_genai_configured = False
_working_model = None
_available_models = []  # populated on first call


def configure_genai():
    """Configure the official Google Generative AI SDK."""
    global _genai_configured
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in your .env file!")
    genai.configure(api_key=api_key)
    _genai_configured = True


def get_working_model() -> str:
    """Find the first Gemini model available for this API key."""
    global _working_model, _available_models
    if _working_model:
        return _working_model

    try:
        _available_models = [m.name.replace("models/", "") for m in genai.list_models()
                     if "generateContent" in m.supported_generation_methods]
        print(f"[RAG Bot] Available Gemini models: {_available_models}")

        for preferred in GEMINI_MODEL_PREFERENCE:
            if preferred in _available_models:
                _working_model = preferred
                print(f"[RAG Bot] Selected model: {_working_model}")
                return _working_model

        # Use the first available one
        if _available_models:
            _working_model = _available_models[0]
            return _working_model
    except Exception as e:
        print(f"[RAG Bot] Could not list models: {e}")

    # Hard fallback
    _working_model = "gemini-2.0-flash"
    _available_models = ["gemini-2.0-flash"]
    return _working_model


def call_gemini(prompt_text: str) -> str:
    """Call Gemini using the official SDK. Cycles through all models on 429 before Ollama fallback."""
    global _working_model
    if not _genai_configured:
        configure_genai()

    # Build ordered list: start from preferred working model, try all others after
    all_models = _available_models if _available_models else [get_working_model()]
    # Put the last working model first to avoid re-scanning on success
    if _working_model and _working_model in all_models:
        ordered = [_working_model] + [m for m in all_models if m != _working_model]
    else:
        ordered = all_models

    last_error = None
    for model_name in ordered:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt_text,
                generation_config=genai.GenerationConfig(temperature=0.0)
            )
            _working_model = model_name  # Cache the model that worked
            return response.text
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower():
                print(f"[RAG Bot] 429 on '{model_name}', trying next model...")
                last_error = err_str
                _working_model = None
                continue  # Try next model
            else:
                raise Exception(f"Gemini API error on '{model_name}': {err_str}")

    # All Gemini models exhausted — try Ollama
    print("[RAG Bot] All Gemini models rate-limited, trying local Ollama fallback...")
    return call_ollama(prompt_text)


def call_ollama(prompt_text: str) -> str:
    """Call a local Ollama instance as fallback when Gemini quota is exceeded."""
    import requests as req
    ollama_models = ["gemma3:1b", "gemma2:2b", "llama3.2:1b", "llama3:8b", "mistral"]
    for model in ollama_models:
        try:
            response = req.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt_text, "stream": False},
                timeout=60
            )
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                if result:
                    print(f"[RAG Bot] Answered via local Ollama model: {model}")
                    return f"[Answered by local model: {model}]\n\n{result}"
        except Exception:
            continue
    raise Exception(
        "❌ Gemini API quota exceeded AND no local Ollama model found.\n\n"
        "To fix: (1) Create a NEW Google AI Studio project at https://aistudio.google.com "
        "and paste its API key in your .env file, OR (2) Install Ollama from https://ollama.ai "
        "and run: ollama pull gemma3:1b"
    )


_cached_vectorstore = None


def ask_question(query: str):
    global _cached_vectorstore
    if not os.path.exists(CHROMA_DB_DIR):
        return "Error: Database is missing. Please click 'Build Vector Index' first.", []

    if _cached_vectorstore is None:
        _cached_vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=get_embeddings()
        )

    retriever = _cached_vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    if not docs:
        return "I cannot answer this based on the provided documents (No relevant data found).", []

    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])

    formatted_prompt = f"""You are an AI assistant for a local Document Q&A Bot.
Answer the question below based ONLY on the provided Context.
If the context does not contain the answer, simply write: 'I cannot answer this based on the provided documents.'
Do not rely on your general knowledge.

Context:
{context_text}

Question:
{query}
"""

    try:
        answer = call_gemini(formatted_prompt)
    except Exception as e:
        answer = f"Error: {str(e)}"

    sources = [{"source": doc.metadata.get("source"), "chunk": doc.metadata.get("chunk")} for doc in docs]
    return answer, sources


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the RAG Bot via CLI")
    parser.add_argument("--query", type=str, required=True, help="Your question")
    args = parser.parse_args()

    print(f"\nLooking up Context for the Question: '{args.query}'...")
    answer, sources = ask_question(args.query)

    print(f"\n### ANSWER ###\n{answer}\n")
    print("### CITATIONS ###")
    for source in set([s["source"] for s in sources]):
        print(f"- {source}")
