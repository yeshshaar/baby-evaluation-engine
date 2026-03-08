import fitz  # This is the PyMuPDF library we just installed
import os

def extract_text_from_pdf(pdf_path):
    """Reads a PDF and extracts all raw text from it."""
    print(f"Attempting to read: {pdf_path}")
    text = ""
    
    try:
        # Open the PDF document
        doc = fitz.open(pdf_path)
        
        # Loop through every page and pull the text
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
            
        return text
        
    except Exception as e:
        print(f"Error reading {pdf_path}. The error was: {e}")
        return None

# --- Testing our function ---
if __name__ == "__main__":
    # We are pointing the script to look inside your data/raw folder
    test_resume_path = "data/raw/test_resume.pdf"
    
    # Check if the file actually exists before trying to read it
    if os.path.exists(test_resume_path):
        print("File found! Extracting text...\n")
        raw_text = extract_text_from_pdf(test_resume_path)
        
        print("--- EXTRACTED TEXT PREVIEW (First 500 characters) ---")
        # We print just the first 500 characters so it doesn't flood your terminal
        print(raw_text[:500]) 
    else:
        print(f"Waiting for data! Please put a PDF named 'test_resume.pdf' inside the data/raw/ folder.")