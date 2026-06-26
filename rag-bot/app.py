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
    
    uploaded_files = st.file_uploader("Drop PDF, TXT, DOCX, or Image files here", accept_multiple_files=True, type=['txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg'])
    
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
                db = build_vector_store()
                if db is not None:
                    st.success("Documents successfully processed and indexed!")
                else:
                    st.error("No documents were found or successfully processed. Please make sure files are uploaded.")
            except Exception as e:
                st.error(f"Error indexing: {e}")

    # Show warnings for scanned PDFs (extracted 0 text characters)
    import json
    warnings_path = "./vectordb/scanned_warnings.json"
    if os.path.exists(warnings_path):
        try:
            with open(warnings_path, "r", encoding="utf-8") as f:
                scanned_files = json.load(f)
            if scanned_files:
                st.warning(
                    "⚠️ **Scanned Documents Detected**\n\n"
                    "The following files appear to be scanned images with no digital text. "
                    "They cannot be queried unless you process them with OCR first:\n\n" +
                    "\n".join([f"- *{sf}*" for sf in scanned_files])
                )
        except Exception:
            pass

st.header("Ask Query on Documents")
query_text = st.text_input("What would you like to know from your uploaded documents?")

if st.button("Ask") and query_text:
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("Please add `GEMINI_API_KEY` to your environment or `.env` file.")
    else:
        with st.spinner("Analyzing and querying Gemini..."):
            answer, sources = ask_question(query_text)
            
            st.markdown("### Answer")
            st.write(answer)
            
            # Show a helpful explanation if the answer could not be found
            if "cannot answer" in answer.lower():
                import json
                warnings_path = "./vectordb/scanned_warnings.json"
                if os.path.exists(warnings_path):
                    try:
                        with open(warnings_path, "r", encoding="utf-8") as f:
                            scanned_files = json.load(f)
                        if scanned_files:
                            st.info(
                                "💡 **Why did this happen?**\n\n"
                                "Your query is about a topic that isn't covered in the digital text documents. "
                                "It is likely located in one of the **Scanned Documents** listed in the sidebar. "
                                "Because scanned notes are stored as images, the bot has no text to search.\n\n"
                                "**How to fix**: Run OCR on your scanned note files to convert them to searchable text, then re-upload them!"
                            )
                    except Exception:
                        pass
            
            st.markdown("### Sources cited")
            if sources:
                unique_sources = {s['source'] for s in sources}
                for src in unique_sources:
                    st.write(f"- *{src}*")
            else:
                st.write("No sources returned.")
