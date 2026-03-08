import pandas as pd
from fpdf import FPDF
import os

def csv_to_pdfs(csv_path, output_folder, text_column, name_column, limit=10):
    """
    Converts CSV rows into individual PDF resumes.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.read_csv(csv_path)
    
    # Process only a small batch to avoid Groq rate limits later
    df = df.head(limit)

    for index, row in df.iterrows():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Clean text to avoid encoding errors
        clean_name = str(row[name_column]).encode('latin-1', 'ignore').decode('latin-1')
        clean_text = str(row[text_column]).encode('latin-1', 'ignore').decode('latin-1')

        pdf.cell(200, 10, txt=f"Resume: {clean_name}", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, txt=clean_text)

        file_name = f"{clean_name.replace(' ', '_')}_{index}.pdf"
        pdf.output(os.path.join(output_folder, file_name))
        print(f"✅ Generated: {file_name}")

if __name__ == "__main__":
    # Update these with your Kaggle CSV details
    csv_to_pdfs(
        csv_path="/Users/yesh/Downloads/Resume.csv", 
        output_folder="data/raw", 
        text_column="Resume_str", 
        name_column="Category", 
        limit=20
    )