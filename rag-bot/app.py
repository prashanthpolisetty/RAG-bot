import streamlit as st
from query import ask_question
from index import build_vector_store
import os

st.set_page_config(page_title="RAG Pipeline Bot", page_icon="🤖")

st.title("🤖 Document RAG Pipeline Bot")
st.markdown("This uses **Lightweight Local Embeddings (MiniLM)** combined with the **Gemini LLM API**.")

with st.sidebar:
    st.header("1. Upload Documents")
    st.info("Upload your documents here to add them to your knowledge base.")
    
    uploaded_files = st.file_uploader("Drop PDF, TXT, or DOCX files here", accept_multiple_files=True, type=['txt', 'pdf', 'docx'])
    
    if st.button("Build Vector Index"):
        if not os.path.exists("./data"):
            os.makedirs("./data")
            
        if uploaded_files:
            # Save all the uploaded files directly into the /data folder
            for file in uploaded_files:
                file_path = os.path.join(".", "data", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                    
        with st.spinner("Analyzing UI files and embedding into ChromaDB..."):
            try:
                build_vector_store()
                st.success("Documents successfully processed and indexed!")
            except Exception as e:
                st.error(f"Error indexing: {e}")

st.header("2. Ask Questions")
query_text = st.text_input("What would you like to know from your uploaded documents?")

if st.button("Ask") and query_text:
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("Please add `GEMINI_API_KEY` to your environment or `.env` file.")
    else:
        with st.spinner("Analyzing and querying Gemini..."):
            answer, sources = ask_question(query_text)
            
            st.markdown("### Answer")
            st.write(answer)
            
            st.markdown("### Sources cited")
            if sources:
                unique_sources = {s['source'] for s in sources}
                for src in unique_sources:
                    st.write(f"- *{src}*")
            else:
                st.write("No sources returned.")
