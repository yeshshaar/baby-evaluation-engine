import sys
import os
import streamlit as st
import pandas as pd

# --- THE FIX: Tell Python to look in the main project folder ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now Python can successfully find the src folder!
from src.main import process_resumes_to_csv

# 1. Configure the page settings
st.set_page_config(page_title="Yield.ai Dashboard", page_icon="🤖", layout="wide")

st.title("🤖 Yield.ai: AI Resume Evaluation Engine")
st.markdown("**Designed for Recruiters:** Upload candidate resumes and a Job Description to instantly rank applicants using Llama 3.1 and Vector Embeddings.")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📥 Input Panel")
    
    # Capture the uploaded files and JD text into variables
    uploaded_files = st.file_uploader("Upload Candidate Resumes (PDF)", accept_multiple_files=True, type=['pdf'])
    jd_text = st.text_area("Paste Job Description Skills Here (comma separated):", height=150, placeholder="e.g., Python, Machine Learning, AWS, FastAPI...")
    
    # When the user clicks the button, this block runs
    if st.button("🚀 Run AI Evaluation", type="primary"):
        if not uploaded_files or not jd_text:
            st.error("⚠️ Please upload at least one resume and enter JD skills!")
        else:
            with st.spinner("Yield.ai Engine is reading resumes and calculating vectors..."):
                
                # --- Ensure cloud directories exist ---
                raw_dir = "data/raw/"
                processed_dir = "data/processed/"
                os.makedirs(raw_dir, exist_ok=True)
                os.makedirs(processed_dir, exist_ok=True)
                
                # Step A: Clean out the old resumes from data/raw/
                for file in os.listdir(raw_dir):
                    if file.endswith(".pdf"):
                        os.remove(os.path.join(raw_dir, file))
                        
                # Step B: Clean out the old CSV so we don't read old data
                output_csv = os.path.join(processed_dir, "evaluation_report.csv")
                if os.path.exists(output_csv):
                    os.remove(output_csv)
                
                # Step C: Save the newly uploaded files into data/raw/
                for uploaded_file in uploaded_files:
                    with open(os.path.join(raw_dir, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Step D: Turn the comma-separated JD text into a Python list
                jd_skills_list = [skill.strip() for skill in jd_text.split(",")]
                
                # Step E: Run your ML Pipeline!
                process_resumes_to_csv(raw_dir, output_csv, jd_skills_list)
                
                # Step F: Check if it ACTUALLY worked
                if os.path.exists(output_csv):
                    st.success("✅ Evaluation Complete!")
                    st.rerun() # This forces the page to refresh and show the table!
                else:
                    st.error("🚨 Pipeline Failed: No data was extracted. Check the Streamlit logs for API errors!")

with col2:
    st.markdown("### 📊 Candidate Leaderboard")
    
    csv_path = "data/processed/evaluation_report.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        with open(csv_path, "rb") as file:
            st.download_button(
                label="📥 Download Full Report (CSV)",
                data=file,
                file_name="yield_ai_evaluation_report.csv",
                mime="text/csv",
            )
    else:
        st.info("No candidates evaluated yet. Upload resumes and run the engine to see the leaderboard.")