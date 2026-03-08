import sys
import os
import streamlit as st
import pandas as pd

# --- ABSOLUTE ROUTING ---
# This finds the root directory of your project regardless of where it's hosted
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.main import process_resumes_to_csv

# Configure the page settings
st.set_page_config(page_title="Yield.ai Dashboard", page_icon="🤖", layout="wide")

st.title("🤖 Yield.ai: AI Resume Evaluation Engine")
st.markdown("**Designed for Recruiters:** Upload candidate resumes and a Job Description to instantly rank applicants using Llama 3.1 and Vector Embeddings.")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📥 Input Panel")
    
    uploaded_files = st.file_uploader("Upload Candidate Resumes (PDF)", accept_multiple_files=True, type=['pdf'])
    jd_text = st.text_area("Paste Full Job Description Here:", height=200, placeholder="Paste the entire job posting here...")
    
    if st.button("🚀 Run AI Evaluation", type="primary"):
        if not uploaded_files or not jd_text:
            st.error("⚠️ Please upload at least one resume and enter JD skills!")
        else:
            with st.spinner("Yield.ai Engine is processing..."):
                
                # Force Absolute Paths for the Cloud
                raw_dir = os.path.join(BASE_DIR, "data", "raw")
                processed_dir = os.path.join(BASE_DIR, "data", "processed")
                os.makedirs(raw_dir, exist_ok=True)
                os.makedirs(processed_dir, exist_ok=True)
                
                # Step A: Clean out the old resumes safely
                if os.path.exists(raw_dir):
                    for file in os.listdir(raw_dir):
                        file_path = os.path.join(raw_dir, file)
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        except Exception as e:
                            print(f"Skipping deletion of {file}: {e}")
                        
                # Step B: Clean out the old CSV
                output_csv = os.path.join(processed_dir, "evaluation_report.csv")
                if os.path.exists(output_csv):
                    os.remove(output_csv)
                
                # Step C: Save newly uploaded files
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(raw_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    print(f"DEBUG: Saved file to {file_path}") # This will show in logs
                
                # Step D: Run ML Pipeline
                process_resumes_to_csv(raw_dir, output_csv, jd_text)
                
                if os.path.exists(output_csv):
                    st.success("✅ Evaluation Complete!")
                    st.rerun()
                else:
                    st.error("🚨 Pipeline Failed: Check the logs to see if the files were saved correctly.")

with col2:
    st.markdown("### 📊 Candidate Leaderboard")
    
    csv_path = os.path.join(BASE_DIR, "data", "processed", "evaluation_report.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        display_df = df[["Candidate Name", "Match Score (%)", "Years of Experience"]]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.markdown("### 🧠 AI Score Explanations")
        for index, row in df.iterrows():
            with st.expander(f"🔍 {row['Candidate Name']} - {row['Match Score (%)']}% Match"):
                st.write(f"**✅ Matched Skills:** {row['Matched Skills']}")
                st.write(f"**❌ Missing Skills:** {row['Missing Skills']}")
                st.info(f"**💡 Recommendation:** {row['How to Improve']}")
        
        st.divider()
        with open(csv_path, "rb") as file:
            st.download_button(label="📥 Download Full Report (CSV)", data=file, file_name="yield_ai_report.csv", mime="text/csv")
    else:
        st.info("No candidates evaluated yet.")