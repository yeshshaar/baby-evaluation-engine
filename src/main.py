import os
import pandas as pd
import requests
import time
import streamlit as st  # 👉 Added this to show errors on the UI!
from src.extractor import extract_text_from_file

def process_resumes_to_csv(raw_dir, output_csv, jd_text, progress_callback=None):
    results = []
    files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.pdf', '.docx'))]
    total_files = len(files)

    for i, filename in enumerate(files):
        filepath = os.path.join(raw_dir, filename)
        candidate_name = filename.replace(".pdf", "").replace(".docx", "").replace("_", " ")
        
        resume_text = extract_text_from_file(filepath)
        
        parsed_data = None
        try:
            # 👉 Bulletproof Network Fallbacks
            primary_url = os.environ.get("API_URL", "http://api:8000/evaluate")
            payload = {
                "candidate_name": candidate_name, 
                "resume_text": resume_text, 
                "jd_text": jd_text
            }
            
            try:
                # Attempt 1: Internal Docker Network
                response = requests.post(primary_url, json=payload, timeout=60)
            except requests.exceptions.ConnectionError:
                # Attempt 2: Mac Host Routing (Bypasses Docker DNS issues)
                fallback_url = "http://host.docker.internal:8000/evaluate"
                response = requests.post(fallback_url, json=payload, timeout=60)

            # 👉 THE NOISY ERROR REPORTER
            if response.status_code == 200:
                parsed_data = response.json()
            else:
                st.error(f"🚨 API SERVER ERROR {response.status_code} on {filename}: {response.text}")
                
        except Exception as e:
            st.error(f"🚨 FATAL NETWORK ERROR: Could not reach the API at all. Details: {e}")

        # Fallback to 0 if we hit the errors above
        if not parsed_data:
            parsed_data = {
                "overall_score": 0,
                "breakdown": {"Skill Match": 0, "Semantic Match": 0, "Experience Relevance": 0},
                "matched_skills": [],
                "missing_skills": []
            }

        results.append({
            "Candidate Name": candidate_name,
            "Score": parsed_data["overall_score"],
            "Skill Match": parsed_data["breakdown"]["Skill Match"],
            "Semantic Match": parsed_data["breakdown"]["Semantic Match"],
            "Experience Relevance": parsed_data["breakdown"]["Experience Relevance"],
            "Matched Skills": ", ".join(parsed_data.get("matched_skills", [])),
            "Missing Skills": ", ".join(parsed_data.get("missing_skills", []))
        })

        if progress_callback:
            progress_callback(i, total_files, filename)
        time.sleep(1)

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="Score", ascending=False)
    df.to_csv(output_csv, index=False)
    
    return df