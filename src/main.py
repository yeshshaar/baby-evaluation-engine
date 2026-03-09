import os
import pandas as pd
import json
import time 
from groq import Groq 
from src.database import save_evaluation
from src.extractor import extract_text_from_pdf
from src.ai_parser import parse_resume_with_llama, parse_jd_with_llama
from src.scorer import calculate_skill_match
from src.sanitizer import clean_pii

def evaluate_with_llama(resume_text, jd_text):
    """Sends the prompt to Groq and forces a JSON response."""
    system_prompt = """
    You are an expert Technical Recruiter evaluating a resume against a Job Description.
    You must analyze the candidate and return strictly a JSON object. Do not include any markdown formatting or extra text outside the JSON.
    
    Format:
    {
      "skill_match_score": 70, 
      "semantic_match_score": 85,
      "experience_relevance_score": 78,
      "matched_skills": ["Skill1", "Skill2"],
      "missing_skills": ["Skill3", "Skill4"]
    }
    """
    user_prompt = f"Job Description:\n{jd_text}\n\nCandidate Resume:\n{resume_text}"

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}, 
            temperature=0.2 
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return None

def process_evaluation(llm_json_response):
    """Parses the JSON and calculates the Yield-AI weighted score."""
    try:
        data = json.loads(llm_json_response)
    except Exception:
        return None 
        
    s_skill = data.get("skill_match_score", 0)
    s_semantic = data.get("semantic_match_score", 0)
    s_exp = data.get("experience_relevance_score", 0)
    
    # Apply the 40/35/25 weights
    overall_score = round((s_skill * 0.40) + (s_semantic * 0.35) + (s_exp * 0.25), 1)
    
    return {
        "overall_score": overall_score,
        "breakdown": {
            "Skill Match": s_skill,
            "Semantic Match": s_semantic,
            "Experience Relevance": s_exp
        },
        "matched_skills": data.get("matched_skills", []),
        "missing_skills": data.get("missing_skills", [])
    }

def check_api():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("🚨 LOG ERROR: GROQ_API_KEY is missing from Streamlit Secrets!")
        return False
    return True

def process_resumes_to_csv(resume_folder, output_csv_path, jd_text_raw, progress_callback=None):

    print(f"--- 🚀 YIELD.AI PIPELINE START ---")
    
    if not check_api():
        return

    abs_resume_path = os.path.abspath(resume_folder)
    
    if not os.path.exists(abs_resume_path):
        print(f"🚨 LOG ERROR: {abs_resume_path} not found!")
        return
        
    files = [f for f in os.listdir(abs_resume_path) if f.lower().endswith(".pdf")]
    total_files = len(files)
    print(f"✅ LOG: Found {total_files} files: {files}")

    if not files:
        print("🚨 LOG ERROR: No PDF files to process.")
        return

    # 1. Parse JD (Initial Progress Update)
    if progress_callback:
        progress_callback(0, total_files, "Extracting Job Description Skills...")
        
    try:
        jd_skills = parse_jd_with_llama(jd_text_raw)
        print(f"✅ LOG: JD Extracted: {jd_skills}")
        time.sleep(1) 
    except Exception as e:
        print(f"🚨 LOG ERROR: JD AI Failed! {e}")
        return 

    results = []
    
    # 2. Process Files
    for i, filename in enumerate(files):
        # --- UI CALLBACK: Update bar and text ---
        if progress_callback:
            progress_callback(i, total_files, filename)
            
        pdf_path = os.path.join(abs_resume_path, filename)
        print(f"🔍 Analyzing: {filename}...")
        
        try:
            raw_text = extract_text_from_pdf(pdf_path)
            if not raw_text:
                print(f"⚠️ Warning: Could not extract text from {filename}")
                continue
                
            sanitized_text = clean_pii(raw_text)
            parsed_data = parse_resume_with_llama(sanitized_text)
            
            # Extract lists safely from the AI response
            core_skills = parsed_data.get("core_skills", [])
            tools = parsed_data.get("tools", [])
            projects = parsed_data.get("projects", [])

            # Scoring
            candidate_skills = core_skills + tools
            match_score, matched, missing, improvement = calculate_skill_match(candidate_skills, jd_skills)

            # --- DATA HARDENING: Prevents letter-splitting ---
            def list_to_str(val):
                if isinstance(val, list):
                    return ", ".join(val) if val else "None"
                return str(val) if val else "None"
            
            results.append({
                "Candidate Name": parsed_data.get("name", filename.replace(".pdf", "")),
                "Match Score (%)": int(match_score),
                "Matched Skills": list_to_str(matched),
                "Missing Skills": list_to_str(missing),
                "How to Improve": improvement,
                "Years of Experience": parsed_data.get("years_of_experience", 0),
                "Core Skills": list_to_str(core_skills),
                "Tools": list_to_str(tools),
                "Projects": list_to_str(projects)
            })
            print(f"✅ Successfully processed {filename}")
            
            # Pause to respect Groq Rate Limits
            time.sleep(1.5) 
            
        except Exception as e:
            print(f"🚨 LOG ERROR: Failed on {filename}: {e}")

    # 3. Final Database and CSV Save
    if results:
        save_evaluation(results)
        print(f"🏆 SUCCESS: Saved {len(results)} results to the Database.")
        
        df = pd.DataFrame(results)
        df.to_csv(output_csv_path, index=False)
    else:
        print("🚨 LOG ERROR: No results were generated.")