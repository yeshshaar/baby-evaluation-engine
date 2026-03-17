import os
import pandas as pd
import time
from src.database import save_evaluation
from src.extractor import extract_text_from_file
from src.sanitizer import clean_pii
from src.chains import run_evaluation_chain, DEFAULT_MODEL


def process_resumes_to_csv(
    raw_dir: str,
    output_csv: str,
    jd_text: str,
    progress_callback=None,
    model_name: str = DEFAULT_MODEL   # ✅ model is now selectable per-run
):
    """
    Reads PDFs/DOCXs, sanitizes PII, evaluates using the selected model,
    saves results to CSV and persists to SQLite.
    """
    results = []
    files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.pdf', '.docx'))]
    total_files = len(files)

    for i, filename in enumerate(files):
        filepath = os.path.join(raw_dir, filename)
        candidate_name = filename.replace(".pdf", "").replace(".docx", "").replace("_", " ")

        # 1. Extract text
        try:
            resume_text = extract_text_from_file(filepath)
        except Exception as e:
            print(f"Extraction Error on {filename}: {e}")
            resume_text = ""

        # 2. Sanitize PII before any LLM call
        resume_text = clean_pii(resume_text)

        # 3. Evaluate via LangChain chain (single source of truth)
        parsed_data = run_evaluation_chain(
            resume_text=resume_text,
            jd_text=jd_text,
            model_name=model_name
        )

        # 4. Map to UI columns
        row = {
            "Candidate Name":       candidate_name,
            "Score":                parsed_data["overall_score"],
            "Skill Match":          parsed_data["breakdown"]["Skill Match"],
            "Semantic Match":       parsed_data["breakdown"]["Semantic Match"],
            "Experience Relevance": parsed_data["breakdown"]["Experience Relevance"],
            "Matched Skills":       ", ".join(parsed_data["matched_skills"]),
            "Missing Skills":       ", ".join(parsed_data["missing_skills"]),
            "Model Used":           parsed_data.get("model_used", model_name),
        }
        results.append(row)

        # 5. Rate limit buffer
        time.sleep(2)

        if progress_callback:
            progress_callback(i, total_files, filename)

    # 6. Save to CSV
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="Score", ascending=False)
    df.to_csv(output_csv, index=False)

    # 7. Persist to SQLite
    if results:
        save_evaluation(results)

    return df
