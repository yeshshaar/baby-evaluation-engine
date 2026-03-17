import os
import PyPDF2
import docx

def extract_text_from_pdf(filepath):
    """Extracts text from a standard PDF file."""
    text = ""
    try:
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"🚨 Error reading PDF {filepath}: {e}")

    return text.strip()

def extract_text_from_docx(filepath):
    """Extracts text from a Microsoft Word document."""
    try:
        doc = docx.Document(filepath)
        # Using list comprehension to grab text, ignoring completely empty lines
        full_text = [para.text for para in doc.paragraphs if para.text.strip()]
        return '\n'.join(full_text)
    except Exception as e:
        print(f"🚨 Error reading DOCX {filepath}: {e}")
        return ""

def extract_text_from_file(filepath):
    """Master router: Checks the extension and routes to the correct extractor."""
    # Safety check
    if not filepath or not os.path.exists(filepath):
        print(f"🚨 Error: File not found at {filepath}")
        return ""

    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.pdf':
        return extract_text_from_pdf(filepath)
    elif ext == '.docx':
        return extract_text_from_docx(filepath)
    else:
        print(f"⚠️ Unsupported file format bypassed: {ext}")
        return ""
