import os
import argparse
from dotenv import load_dotenv
import requests
import json
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from index import get_embeddings, CHROMA_DB_DIR
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

load_dotenv()

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=5), 
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def invoke_gemini_with_retry(prompt_text):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set!")
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"temperature": 0.0}
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        # If 404 occurs like earlier, immediately fallback to the older legacy endpoint
        if response.status_code == 404:
            url_fallback = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.0-pro:generateContent?key={api_key}"
            response = requests.post(url_fallback, headers=headers, json=payload, timeout=15)
            
        response.raise_for_status()
        
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        raise Exception(f"API Request Failed: {str(e)}")

_cached_vectorstore = None

def ask_question(query):
    global _cached_vectorstore
    if not os.path.exists(CHROMA_DB_DIR):
        return "Error: Database is missing. Please run indexing first ('python index.py').", []
        
    if _cached_vectorstore is None:
        _cached_vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR, 
            embedding_function=get_embeddings()
        )
    retriever = _cached_vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)
    
    if not docs:
        return "I cannot answer this based on the provided documents (No relevant data found).", []
        
    context_text = "\\n\\n---\\n\\n".join([doc.page_content for doc in docs])
        
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
        answer = invoke_gemini_with_retry(formatted_prompt)
    except Exception as e:
        answer = f"Error: Failed to request Gemini API.\\nDetails: {str(e)}"
        
    sources = [{"source": doc.metadata.get("source"), "chunk": doc.metadata.get("chunk")} for doc in docs]
    return answer, sources

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the RAG Bot via CLI")
    parser.add_argument("--query", type=str, required=True, help="Your question")
    args = parser.parse_args()
    
    print(f"\\nLooking up Context for the Question: '{args.query}'...")
    answer, sources = ask_question(args.query)
    
    print(f"\\n### ANSWER ###\\n{answer}\\n")
    print("### CITATIONS ###")
    for source in set([s['source'] for s in sources]):
        print(f"- {source}")
