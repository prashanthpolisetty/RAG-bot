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
                     if "generateContent" in m.supported_generation_methods
                     and "-tts" not in m.name.lower()
                     and "gemma" not in m.name.lower()
                     and "lyria" not in m.name.lower()]
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
    """Call Gemini using the official SDK. Cycles through all models on errors before Ollama fallback."""
    global _working_model, _available_models
    if not _genai_configured:
        configure_genai()

    # Ensure available models are loaded
    if not _available_models:
        get_working_model()

    # Sort available models: preferred first (in preference order), then others
    preferred_available = [m for m in GEMINI_MODEL_PREFERENCE if m in _available_models]
    others_available = [m for m in _available_models if m not in GEMINI_MODEL_PREFERENCE]
    ordered = preferred_available + others_available

    # Put the last working model at the very beginning of the list to speed up subsequent requests
    if _working_model and _working_model in ordered:
        ordered = [_working_model] + [m for m in ordered if m != _working_model]

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
            # If it's a hard API key validation issue, raise it immediately
            if "key" in err_str.lower() and ("valid" in err_str.lower() or "invalid" in err_str.lower() or "not found" in err_str.lower()):
                raise Exception(f"Gemini API key error: {err_str}")
                
            print(f"[RAG Bot] Error on model '{model_name}': {err_str}. Trying next model...")
            last_error = err_str
            _working_model = None
            continue  # Try next model

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


def generate_queries(original_query: str) -> list[str]:
    """Generate 2 alternative search keywords/phrases using Gemini to expand the search scope."""
    prompt = f"""You are a search query expansion assistant.
Given the user's search query, output exactly two alternative search terms, conceptual keywords, or synonyms that are likely to appear in related documents.
Output ONLY the alternative search queries, one per line. Do not number them, and do not add any extra text or conversational filler.

User query: "{original_query}"
"""
    try:
        response_text = call_gemini(prompt)
        queries = [original_query]
        for line in response_text.strip().split("\n"):
            cleaned = line.strip().strip("-*•").strip().strip('"').strip()
            if cleaned and cleaned.lower() != original_query.lower():
                queries.append(cleaned)
        return list(dict.fromkeys(queries))[:3]
    except Exception as e:
        print(f"[RAG Bot] Query expansion failed: {e}")
        return [original_query]


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

    # Multi-query expansion to capture conceptually related chunks
    expanded_queries = generate_queries(query)
    print(f"[RAG Bot] Expanded queries for search: {expanded_queries}")

    retriever = _cached_vectorstore.as_retriever(search_kwargs={"k": 8})
    all_docs = []
    seen_contents = set()

    for eq in expanded_queries:
        docs = retriever.invoke(eq)
        for doc in docs:
            content_hash = doc.page_content.strip()
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                all_docs.append(doc)

    if not all_docs:
        return "I cannot answer this based on the provided documents (No relevant data found).", []

    context_text = "\n\n---\n\n".join([doc.page_content for doc in all_docs])

    formatted_prompt = f"""You are an AI assistant for a local Document Q&A Bot.
Answer the question below based ONLY on the provided Context.
If the context does not contain the answer, simply write: 'I cannot answer this based on the provided documents.'
Do not rely on your general knowledge.

Formatting Instructions (for successful answers):
- Provide a clear, easy-to-understand explanation using simple analogies where helpful.
- Structure your answer with bullet points and bold key terms to make it easily explainable.
- Keep the language engaging, educational, and structured (e.g. Definition, How it works, Details).

Context:
{context_text}

Question:
{query}
"""

    try:
        answer = call_gemini(formatted_prompt)
    except Exception as e:
        answer = f"Error: {str(e)}"

    sources = [{"source": doc.metadata.get("source"), "chunk": doc.metadata.get("chunk"), "content": doc.page_content} for doc in all_docs]
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
