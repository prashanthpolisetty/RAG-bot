# Document Q&A Bot (RAG Pipeline)

A highly resilient Retrieval-Augmented Generation (RAG) chatbot designed to accurately answer questions strictly based on uploaded documents. By strategically offloading embedding generation to a local lightweight model and utilizing the Gemini API exclusively for fast question answering, the bot avoids common rate-limit crashes and seamlessly handles diverse document formats (TXT, PDF, DOCX) via an interactive Streamlit UI. 

---

### 1. Tech Stack
* **Python**: 3.11+
* **Streamlit**: 1.32.2 (Web Frontend UI)
* **LangChain**: 0.1.13 (Pipeline Orchestration)
* **Sentence-Transformers**: 2.5.1 (Local Embeddings generation)
* **ChromaDB**: 0.4.24 (Vector Database)
* **langchain-google-genai**: 0.0.11 (Gemini LLM Integration)
* **PyPDF2**: 3.0.1 (PDF parsing)
* **python-docx**: 1.1.0 (Word Document parsing)
* **Tenacity**: 8.2.3 (Exponential Backoff / Failsafe mechanism)

### 2. Architecture Overview
The system follows a standard RAG workflow optimized for high efficiency:
1. **Ingestion**: Uploaded documents are read and text is reliably extracted via format-specific parsers.
2. **Chunking**: LangChain splits the unified text into smaller, overlapping chunks.
3. **Embedding**: `all-MiniLM-L6-v2` executes entirely locally on the CPU to convert chunked text into vector space without making internet requests.
4. **Vector Storage**: Vectors are committed into an offline ChromaDB instance.
5. **Retrieval**: User questions are embedded, and ChromaDB performs a similarity search to return the top 5 most relevant chunks.
6. **LLM Generation**: The retrieved context is bundled alongside the user query into a strict prompt. The `gemini-1.5-flash` LLM processes it and generates an answer, strictly citing its sources and rejecting external knowledge.

### 3. Chunking Strategy
**Strategy**: `RecursiveCharacterTextSplitter` using chunks of `700` characters and `100` character overlaps.
**Justification**: A 700-character chunk provides enough context to encompass an average full paragraph or complex idea without overwhelming the context window. The 100-character overlap prevents structural "hard cuts" through the middle of sentences, making sure that concepts split across two boundaries retain enough surrounding context to remain semantically intact for the embedding model.

### 4. Embedding Model & Vector Database Choice
* **Embedding Model (`all-MiniLM-L6-v2`)**: Rather than relying on cloud providers for embeddings, utilizing this specific lightweight HuggingFace model executes instantly offline. This massive rate-limit optimization completely bypasses the strict `429 Quota` API errors typically encountered when submitting thousands of text chunks to Gemini on a free tier. 
* **Vector DB (`ChromaDB`)**: Chosen for its robust, lightweight offline persistence. It runs securely within the local directory (`./vectordb`) requiring no docker containers or complicated cloud cluster setup, maintaining the strict offline-first integrity of the indexing phase.

### 5. Setup Instructions
1. Clone this repository locally.
   ```bash
   git clone <repo_url>
   cd rag-bot
   ```
2. Create and activate a virtual environment (Recommended).
   ```bash
   python -m venv venv
   venv\\Scripts\\activate  # On Windows
   ```
3. Install all required dependencies.
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your `.env` file (See section below).
5. Start the frontend Streamlit Application.
   ```bash
   python -m streamlit run app.py
   ```

### 6. Environment Variables
This project requires one API key to utilize the Gemini LLM for answering queries.
Rename the provided `.env.example` file to `.env` and configure your key inside.
**CRITICAL**: *Never commit your `.env` file to Git. Ensure `.gitignore` is active.*
```env
GEMINI_API_KEY=AIzaSy...
```

### 7. Example Queries
Drop the included dummy text files into the Streamlit Web Application and test it with these queries:
1. **"What is the total budget for Project Apollo?"** 
   *(Expected Theme: The bot should state $450,000 and accurately cite exactly which file it sourced the timeline from).*
2. **"When are employees permitted to work remotely?"** 
   *(Expected Theme: Will extract HR policies specifically targeting Tuesdays and Thursdays).*
3. **"Who should I call for a server emergency on a weekend?"**
   *(Expected Theme: It should identify the Lead System Administrator and output their specific emergency contact number).*
4. **"What was the revenue growth percentage in Q3?"**
   *(Expected Theme: Will calculate or retrieve the 14% revenue growth factor from the respective finance document).*
5. **"What is the policy for bringing dogs to the office?"** *(Unanswerable Test)*
   *(Expected Theme: Because it honors the strict prompt instructions to NOT hallucinate, it will correctly output: "I cannot answer this based on the provided documents.")*

### 8. Known Limitations
1. **First-run Latency**: Because the `all-MiniLM-L6-v2` embedding model runs locally, the very first time `Build Vector Index` is pressed, the user will experience a slight delay while the ~90MB model downloads transparently in the background.
2. **Tabular Formatting**: The rudimentary text extraction libraries (`PyPDF2`, `python-docx`) do not preserve the structural boundaries of highly complex tables, which occasionally degrades retrieval accuracy on heavily formatted financial spreadsheets.
