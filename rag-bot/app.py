import streamlit as st
from query import ask_question
from index import build_vector_store
import os
import json

# Set up page configurations
st.set_page_config(page_title="RAG Pipeline Bot", page_icon="🤖", layout="wide")

# Inject premium CSS for visual excellence
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Glowing title styling */
    .title-text {
        background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    /* Subtitle styling */
    .subtitle-text {
        color: #8A99AD;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Sidebar container styling */
    section[data-testid="stSidebar"] {
        background-color: #0E1117;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Glassmorphic card design for files/warnings */
    .doc-card {
        padding: 12px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 8px;
        font-size: 0.95rem;
    }
    
    /* Suggested questions button styling */
    div.stButton > button {
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background-color: rgba(255, 255, 255, 0.05);
        color: #E2E8F0;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        border-color: #0072FF;
        background-color: rgba(0, 114, 255, 0.1);
        color: #00C6FF;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Caching & Core Functions -----------------

@st.cache_data(show_spinner=False)
def get_cached_answer(query_text: str):
    """Cached query execution to prevent duplicate LLM calls."""
    return ask_question(query_text)

# ----------------- Sidebar -----------------

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Show active knowledge base files
    st.markdown("### 📚 Loaded Documents")
    data_dir = "./data"
    indexed_files = []
    if os.path.exists(data_dir):
        indexed_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        
    if indexed_files:
        for file in indexed_files:
            st.markdown(f"<div class='doc-card'>📄 {file}</div>", unsafe_allow_html=True)
    else:
        st.info("No documents pre-loaded. Please upload files below.")
        
    st.markdown("---")
    st.markdown("### 📤 Upload New Documents")
    
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, DOCX, or Image files",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg'],
        label_visibility="collapsed"
    )
    
    if st.button("Build Vector Index", use_container_width=True):
        if not os.path.exists("./data"):
            os.makedirs("./data")
            
        if uploaded_files:
            # Save uploaded files into /data directory
            for file in uploaded_files:
                file_path = os.path.join(".", "data", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                    
        with st.spinner("Processing documents and updating ChromaDB..."):
            try:
                db = build_vector_store()
                if db is not None:
                    # Clear query cache and reset conversation on database rebuild
                    st.cache_data.clear()
                    st.session_state.messages = []
                    st.success("Database successfully updated and re-indexed!")
                    st.rerun()
                else:
                    st.error("No documents were found or successfully processed.")
            except Exception as e:
                st.error(f"Error indexing: {e}")

    # Display warnings for scanned PDF files
    warnings_path = "./vectordb/scanned_warnings.json"
    if os.path.exists(warnings_path):
        try:
            with open(warnings_path, "r", encoding="utf-8") as f:
                scanned_files = json.load(f)
            if scanned_files:
                st.warning(
                    "⚠️ **Scanned Files (No Text)**\n\n"
                    "The following files need OCR to be readable:\n\n" +
                    "\n".join([f"- *{sf}*" for sf in scanned_files])
                )
        except Exception:
            pass

    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------- Main Chat Window -----------------

# Header Section
st.markdown("<h1 class='title-text'>🤖 Document RAG Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Ask questions and get answers directly sourced from your textbook or notes. Chat history is saved until you close the tab.</p>", unsafe_allow_html=True)

# Check if Gemini API key is configured
api_key_configured = os.environ.get("GEMINI_API_KEY") or (
    os.path.exists(".env") and "GEMINI_API_KEY" in open(".env").read()
)

if not api_key_configured:
    st.warning("⚠️ **GEMINI_API_KEY is missing.** Please set it in your Streamlit Cloud Secrets or local `.env` file to enable querying.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display welcome message if no conversation has started
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Hello! I am your RAG Q&A chatbot. 📚\n\n"
            "I will answer your questions using the documents loaded in the sidebar. "
            "How can I help you study today?"
        )

# Display chat messages from history on rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Sources Cited"):
                for i, src in enumerate(message["sources"]):
                    st.markdown(f"**Chunk {i+1} from `{src['source']}` (Chunk {src['chunk']})**")
                    st.info(src['content'])

# Helper function to trigger queries programmatically (for suggested questions)
def handle_user_query(prompt):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Show user message in chat
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Get answer from backend
    with st.chat_message("assistant"):
        with st.spinner("Searching document context and generating answer..."):
            answer, sources = get_cached_answer(prompt)
            st.markdown(answer)
            if sources:
                with st.expander("🔍 View Sources Cited"):
                    for i, src in enumerate(sources):
                        st.markdown(f"**Chunk {i+1} from `{src['source']}` (Chunk {src['chunk']})**")
                        st.info(src['content'])
            
            # Save assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

# Suggested Questions Panel (only shown when conversation is empty)
if len(st.session_state.messages) <= 0:
    st.markdown("### 💡 Try Asking:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📌 Summarize Unit 1", use_container_width=True):
            handle_user_query("Give me a comprehensive summary of Unit 1.")
            st.rerun()
    with col2:
        if st.button("📌 What is IoT?", use_container_width=True):
            handle_user_query("What is the definition of Internet of Things (IoT)?")
            st.rerun()
    with col3:
        if st.button("📌 Explain TCP/IP Protocol Suite", use_container_width=True):
            handle_user_query("Explain the layers and protocols of the TCP/IP suite.")
            st.rerun()

# React to user input
if user_input := st.chat_input("Ask a question about your documents..."):
    handle_user_query(user_input)
