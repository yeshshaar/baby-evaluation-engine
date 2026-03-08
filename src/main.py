import os
import pandas as pd
import json

# Importing the modules you built!
# Importing the modules you built! (Updated for absolute routing)
from src.extractor import extract_text_from_pdf
from src.ai_parser import parse_resume_with_llama
from src.scorer import calculate_skill_match

def process_resumes_to_csv(resume_folder, output_csv_path, jd_skills):
    """Orchestrates the extraction, AI parsing, scoring, and CSV export."""
    print("Starting B.A.B.Y. Evaluation Pipeline...\n")
    
    results = []
    
    # Loop through every PDF in the raw data folder
    for filename in os.listdir(resume_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(resume_folder, filename)
            print(f"Processing: {filename}")
            
            # 1. Extract raw text
            raw_text = extract_text_from_pdf(pdf_path)
            if not raw_text:
                continue
                
            # 2. Use AI to structure the data
            parsed_data = parse_resume_with_llama(raw_text)
            if not parsed_data:
                continue
                
            # 3. Calculate the match score
            candidate_skills = parsed_data.get("core_skills", []) + parsed_data.get("tools", [])
           # 3. Calculate the match score and get the AI explanation
            candidate_skills = parsed_data.get("core_skills", []) + parsed_data.get("tools", [])
            
            # Catch all 4 outputs from the updated scorer!
            match_score, matched, missing, improvement = calculate_skill_match(candidate_skills, jd_skills)
            
            # 4. Compile the final record
            record = {
                "Candidate Name": parsed_data.get("name", "Unknown"),
                "Match Score (%)": match_score,
                "Matched Skills": ", ".join(matched),
                "Missing Skills": ", ".join(missing),
                "How to Improve": improvement,
                "Years of Experience": parsed_data.get("years_of_experience", 0),
                "Core Skills": ", ".join(parsed_data.get("core_skills", [])),
                "Tools": ", ".join(parsed_data.get("tools", [])),
                "Projects": ", ".join(parsed_data.get("projects", []))
            }
            results.append(record)
            print(f"Finished evaluating {filename}. Score: {match_score}%\n")
            
    # 5. Save everything to a CSV file using Pandas
    if results:
        df = pd.DataFrame(results)
        # Sort by highest score first!
        df = df.sort_values(by="Match Score (%)", ascending=False) 
        df.to_csv(output_csv_path, index=False)
        print(f"SUCCESS! Evaluation report saved to: {output_csv_path}")
    else:
        print("No resumes were successfully processed.")

# --- Running the Full Engine ---
if __name__ == "__main__":
    # Define our folders
    input_folder = "data/raw/"
    output_file = "data/processed/evaluation_report.csv"
    
    # The target Job Description skills
    target_jd = ["Python", "Machine Learning", "NLP", "AWS", "Docker", "FastAPI"]
    
    process_resumes_to_csv(input_folder, output_file, target_jd)