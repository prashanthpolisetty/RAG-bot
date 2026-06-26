import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from utils.ingestion import process_directory

# Suppress ChromaDB telemetry noise
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

CHROMA_DB_DIR = "./vectordb"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class LocalEmbeddings(Embeddings):
    """
    Custom embeddings class using SentenceTransformer directly,
    bypassing LangChain's HuggingFaceEmbeddings wrapper which
    is incompatible with newer PyTorch versions (meta tensor issue).
    """
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, show_progress_bar=False, convert_to_numpy=True).tolist()


_cached_embeddings = None

def get_embeddings():
    global _cached_embeddings
    if _cached_embeddings is None:
        _cached_embeddings = LocalEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _cached_embeddings

def build_vector_store(data_directory="./data"):
    # Release any cached vectorstore locks and explicitly close the client connections
    import sys
    if 'query' in sys.modules:
        q_mod = sys.modules['query']
        if hasattr(q_mod, '_cached_vectorstore') and q_mod._cached_vectorstore is not None:
            print("[RAG Bot] Closing cached vectorstore client...")
            try:
                if hasattr(q_mod._cached_vectorstore, '_client') and hasattr(q_mod._cached_vectorstore._client, 'close'):
                    q_mod._cached_vectorstore._client.close()
            except Exception as e:
                print(f"[RAG Bot] Error closing client: {e}")
            q_mod._cached_vectorstore = None
            
    import gc
    gc.collect()

    # Reset/clear existing vector database using the Chroma API first to avoid file-locking / DBMOVED errors.
    # If the database directory exists, we delete the collection to empty it, rather than deleting the SQLite file from disk.
    if os.path.exists(CHROMA_DB_DIR):
        print(f"Clearing existing database at {CHROMA_DB_DIR}...")
        try:
            # We initialize a temporary vector store using the existing database and delete the collection.
            # This resets the data without moving/deleting SQLite files that may be open in other threads.
            from langchain_community.vectorstores import Chroma
            temp_db = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=get_embeddings()
            )
            temp_db.delete_collection()
            if hasattr(temp_db, '_client') and hasattr(temp_db._client, 'close'):
                temp_db._client.close()
            print("Successfully cleared collection using Chroma API.")
        except Exception as e:
            print(f"Warning: Could not clear database using Chroma API: {e}. Falling back to folder deletion.")
            # Fallback to rmtree if the Chroma API deletion fails
            import shutil
            try:
                shutil.rmtree(CHROMA_DB_DIR)
            except Exception as re:
                print(f"Warning: Fallback rmtree also failed: {re}")

    print("Reading documents from /data...")
    raw_docs = process_directory(data_directory)
    if not raw_docs:
        print("No documents found to process. Please put files into /data.")
        return None
    
    # Chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    import json
    docs = []
    scanned_files = []
    for filename, text in raw_docs:
        if not text.strip():
            scanned_files.append(filename)
            continue
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            doc = Document(page_content=chunk, metadata={"source": filename, "chunk": i})
            docs.append(doc)
            
    # Embed and Store in Vector DB
    embeddings = get_embeddings()
    print(f"Embedding {len(docs)} chunks locally using {EMBEDDING_MODEL_NAME}...")
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist()
    
    # Close the vectorstore client to release SQLite locks immediately
    try:
        if hasattr(vectorstore, '_client') and hasattr(vectorstore._client, 'close'):
            vectorstore._client.close()
    except Exception as e:
        print(f"Warning closing database client after indexing: {e}")
    
    # Save scanned/empty file warnings
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    warnings_path = os.path.join(CHROMA_DB_DIR, "scanned_warnings.json")
    with open(warnings_path, "w", encoding="utf-8") as f:
        json.dump(scanned_files, f, indent=4)
        
    print(f"Vector DB successfully updated and saved in {CHROMA_DB_DIR}.")
    return vectorstore

if __name__ == "__main__":
    build_vector_store()
