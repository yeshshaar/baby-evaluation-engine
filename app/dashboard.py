import sys
import os

# --- THE FIX: Tell Python to look in the main project folder ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd

# Now Python can successfully find the src folder!
from src.main import process_resumes_to_csv

# 1. Configure the page settings
st.set_page_config(page_title="B.A.B.Y. Dashboard", page_icon="👶", layout="wide")

st.title("Yield.ai : AI Resume Evaluation Engine")
st.subheader("Biometric & Ability Based Yield-engine")
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
            # Show a loading spinner while the AI thinks
            with st.spinner("B.A.B.Y. Engine is reading resumes and calculating vectors..."):
                
                # Step A: Clean out the old resumes from data/raw/
                raw_dir = "data/raw/"
                processed_dir = "data/processed/"
                os.makedirs(raw_dir, exist_ok=True)
                os.makedirs(processed_dir, exist_ok=True)
                for file in os.listdir(raw_dir):
                    if file.endswith(".pdf"):
                        os.remove(os.path.join(raw_dir, file))
                
                # Step B: Save the newly uploaded files into data/raw/
                for uploaded_file in uploaded_files:
                    with open(os.path.join(raw_dir, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Step C: Turn the comma-separated JD text into a Python list
                jd_skills_list = [skill.strip() for skill in jd_text.split(",")]
                
                # Step D: Run your ML Pipeline!
                output_csv = "data/processed/evaluation_report.csv"
                process_resumes_to_csv(raw_dir, output_csv, jd_skills_list)
                
                st.success("✅ Evaluation Complete!")

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
                file_name="baby_evaluation_report.csv",
                mime="text/csv",
            )
    else:
        st.info("No candidates evaluated yet. Upload resumes and run the engine to see the leaderboard.")