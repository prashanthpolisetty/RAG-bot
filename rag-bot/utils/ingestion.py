import os
from docx import Document as DocxDocument

def clean_extracted_text(text: str) -> str:
    if not text:
        return ""
    # Normalize common PDF extraction ligature errors (e.g., ' Ɵ' or 'Ɵ' -> 'ti')
    text = text.replace(" Ɵ", "ti").replace("Ɵ", "ti")
    return text

def ingest_file(file_path):
    filename = os.path.basename(file_path)
    cache_dir = "./vectordb/cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{filename}.txt")
    
    # Read from cache if it exists to avoid running slow OCR again
    if os.path.exists(cache_path):
        print(f"[RAG Bot Cache] Loading cached text for {filename}...")
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    text = ""
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    elif ext == '.pdf':
        import fitz  # PyMuPDF
        from PIL import Image
        import io
        import google.generativeai as genai
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY")
        
        if api_key:
            genai.configure(api_key=api_key)
            
        try:
            doc = fitz.open(file_path)
            for i, page in enumerate(doc):
                page_text = page.get_text()
                # If text is extremely short, it is likely a scanned page of images
                if len(page_text.strip()) < 40:
                    print(f"[RAG Bot OCR] Scanned page detected on page {i+1} of {os.path.basename(file_path)}. Running Gemini OCR...")
                    if not api_key:
                        print("[RAG Bot OCR] WARNING: GEMINI_API_KEY is not set. Skipping OCR.")
                        text += page_text + "\n"
                        continue
                    
                    try:
                        # Render page to image
                        pix = page.get_pixmap(dpi=150)
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        
                        # Retrieve available vision models
                        try:
                            available_vision = [m.name.replace("models/", "") for m in genai.list_models()
                                                if "generateContent" in m.supported_generation_methods
                                                and "-tts" not in m.name.lower()
                                                and "gemma" not in m.name.lower()
                                                and "lyria" not in m.name.lower()]
                        except Exception:
                            available_vision = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
                        
                        preferred_order = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"]
                        ordered = [m for m in preferred_order if m in available_vision] + [m for m in available_vision if m not in preferred_order]
                        
                        ocr_success = False
                        ocr_text = ""
                        for model_name in ordered:
                            try:
                                model = genai.GenerativeModel(model_name)
                                response = model.generate_content([
                                    "Extract all visible text from this image. Output only the extracted text exactly as it appears, without any commentary, markdown framing, or conversational filler.",
                                    img
                                ])
                                ocr_text = response.text
                                if ocr_text:
                                    ocr_success = True
                                    break
                                    
                            except Exception as e:
                                print(f"[RAG Bot OCR] Vision error on model '{model_name}': {e}. Trying next...")
                                continue
                                
                        if ocr_success:
                            text += ocr_text + "\n"
                        else:
                            text += page_text + "\n"
                    except Exception as e:
                        print(f"[RAG Bot OCR] Failed to OCR page {i+1}: {e}")
                        text += page_text + "\n"
                else:
                    text += page_text + "\n"
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            raise e
    elif ext in ['.png', '.jpg', '.jpeg']:
        from PIL import Image
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError(f"GEMINI_API_KEY is not set. Cannot run OCR on image {file_path}!")
            
        genai.configure(api_key=api_key)
        try:
            img = Image.open(file_path)
            # Cycle through preferred vision models dynamically
            try:
                available_vision = [m.name.replace("models/", "") for m in genai.list_models()
                                    if "generateContent" in m.supported_generation_methods
                                    and "-tts" not in m.name.lower()
                                    and "gemma" not in m.name.lower()
                                    and "lyria" not in m.name.lower()]
            except Exception:
                available_vision = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
            
            preferred_order = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"]
            ordered = [m for m in preferred_order if m in available_vision] + [m for m in available_vision if m not in preferred_order]

            success = False
            for model_name in ordered:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content([
                        "Extract all visible text from this image. Output only the extracted text exactly as it appears, without any commentary, markdown framing, or conversational filler.",
                        img
                    ])
                    text = response.text
                    if text:
                        success = True
                        break
                except Exception as e:
                    print(f"[RAG Bot OCR] Vision error on model '{model_name}': {e}. Trying next...")
                    continue
            if not success:
                raise ValueError("All Gemini vision models failed or rate-limited for OCR.")
        except Exception as e:
            print(f"Error running OCR on image {file_path}: {e}")
            raise e
    elif ext == '.docx':
        doc = DocxDocument(file_path)
        for para in doc.paragraphs:
            if para.text:
                text += para.text + "\n"
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    cleaned_text = clean_extracted_text(text)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
    except Exception as e:
        print(f"Warning: Could not write cache file: {e}")
    return cleaned_text

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
