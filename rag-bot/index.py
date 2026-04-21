import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    print("Reading documents from /data...")
    raw_docs = process_directory(data_directory)
    if not raw_docs:
        print("No documents found to process. Please put files into /data.")
        return None
    
    # Chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    docs = []
    for filename, text in raw_docs:
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
    print(f"Vector DB successfully updated and saved in {CHROMA_DB_DIR}.")
    return vectorstore

if __name__ == "__main__":
    build_vector_store()
