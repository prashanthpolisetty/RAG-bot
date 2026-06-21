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
    # Release any cached vectorstore locks in the query module to allow deletion on Windows
    import sys
    if 'query' in sys.modules:
        sys.modules['query']._cached_vectorstore = None
    import gc
    gc.collect()

    # Clear existing vector database to prevent duplicate chunks from flooding search results
    import shutil
    if os.path.exists(CHROMA_DB_DIR):
        print(f"Clearing existing vector database at {CHROMA_DB_DIR}...")
        try:
            shutil.rmtree(CHROMA_DB_DIR)
        except Exception as e:
            print(f"Warning: Could not clear database directory: {e}")

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
    
    # Save scanned/empty file warnings
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    warnings_path = os.path.join(CHROMA_DB_DIR, "scanned_warnings.json")
    with open(warnings_path, "w", encoding="utf-8") as f:
        json.dump(scanned_files, f, indent=4)
        
    print(f"Vector DB successfully updated and saved in {CHROMA_DB_DIR}.")
    return vectorstore

if __name__ == "__main__":
    build_vector_store()
