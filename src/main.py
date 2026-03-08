import os
import pandas as pd
import json
from groq import Groq 

from src.extractor import extract_text_from_pdf
from src.ai_parser import parse_resume_with_llama, parse_jd_with_llama
from src.scorer import calculate_skill_match
from src.sanitizer import clean_pii

def check_api():
    """Diagnostic check to see if Groq is actually working in the cloud."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("🚨 LOG ERROR: GROQ_API_KEY is missing from Streamlit Secrets!")
        return False
    print(f"✅ LOG: API Key found (starts with {api_key[:6]}...)")
    return True

def process_resumes_to_csv(resume_folder, output_csv_path, jd_text_raw):
    print(f"--- DIAGNOSTIC PIPELINE START ---")
    
    if not check_api():
        return

    # Check the folder
    abs_resume_path = os.path.abspath(resume_folder)
    print(f"✅ LOG: Checking absolute path: {abs_resume_path}")
    
    if not os.path.exists(abs_resume_path):
        print(f"🚨 LOG ERROR: {abs_resume_path} does not exist!")
        return
        
    files = [f for f in os.listdir(abs_resume_path) if f.lower().endswith(".pdf")]
    print(f"✅ LOG: Found {len(files)} files: {files}")

    if not files:
        print("🚨 LOG ERROR: No PDF files found in the directory. Check dashboard saving logic.")
        return

    try:
        jd_skills = parse_jd_with_llama(jd_text_raw)
        print(f"✅ LOG: JD Extracted: {jd_skills}")
    except Exception as e:
        print(f"🚨 LOG ERROR: JD AI Failed! Error: {e}")
        return 

    results = []
    for filename in files:
        pdf_path = os.path.join(abs_resume_path, filename)
        print(f"Processing: {filename}")
        
        try:
            raw_text = extract_text_from_pdf(pdf_path)
            sanitized_text = clean_pii(raw_text)
            parsed_data = parse_resume_with_llama(sanitized_text)
            
            candidate_skills = parsed_data.get("core_skills", []) + parsed_data.get("tools", [])
            match_score, matched, missing, improvement = calculate_skill_match(candidate_skills, jd_skills)
            
            results.append({
                "Candidate Name": parsed_data.get("name", "Unknown"),
                "Match Score (%)": match_score,
                "Matched Skills": ", ".join(matched),
                "Missing Skills": ", ".join(missing),
                "How to Improve": improvement,
                "Years of Experience": parsed_data.get("years_of_experience", 0),
                "Core Skills": ", ".join(parsed_data.get("core_skills", [])),
                "Tools": ", ".join(parsed_data.get("tools", [])),
                "Projects": ", ".join(parsed_data.get("projects", []))
            })
            print(f"✅ Successfully processed {filename}")
        except Exception as e:
            print(f"🚨 LOG ERROR: Failed on {filename}: {e}")

    if results:
        pd.DataFrame(results).to_csv(output_csv_path, index=False)
        print(f"🏆 SUCCESS: CSV created at {output_csv_path}")
    else:
        print("🚨 LOG ERROR: Pipeline finished with 0 results.")