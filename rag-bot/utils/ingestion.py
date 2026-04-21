import os
from PyPDF2 import PdfReader
from docx import Document as DocxDocument

def ingest_file(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    text = ""
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    elif ext == '.pdf':
        reader = PdfReader(file_path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\\n"
    elif ext == '.docx':
        doc = DocxDocument(file_path)
        for para in doc.paragraphs:
            if para.text:
                text += para.text + "\\n"
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return text

def process_directory(directory_path="data"):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        return []
    docs = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            try:
                content = ingest_file(file_path)
                docs.append((filename, content))
                print(f"Successfully ingested: {filename}")
            except Exception as e:
                print(f"Skipping {filename}: {str(e)}")
    return docs
