import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from utils.ingestion import process_directory

CHROMA_DB_DIR = "./vectordb"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_cached_embeddings = None

def get_embeddings():
    global _cached_embeddings
    if _cached_embeddings is None:
        _cached_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
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
        separators=["\\n\\n", "\\n", ".", " ", ""]
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
