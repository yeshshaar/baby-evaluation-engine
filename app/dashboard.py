import sys
import os
import pandas as pd
import streamlit as st

# 1. PATH INJECTION (Crucial for Cloud)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 2. GLOBAL PATH DEFINITIONS
# This fixes the NameError: output_csv is not defined
raw_dir = "data/raw"
processed_dir = "data/processed"
output_csv = os.path.join(processed_dir, "evaluation_report.csv")

# Ensure folders exist on the server
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

# 3. NOW PROCEED WITH OTHER IMPORTS
from src.main import process_resumes_to_csv
from src.database import init_db, get_all_evaluations
from src.visualizer import create_radar_chart

init_db()
# --- 1. SETTINGS & PATHS ---
st.set_page_config(page_title="Yield.ai | MLE Evaluation Engine", layout="wide")

# Define global paths so they are available in all tabs
raw_dir = "data/raw"
processed_dir = "data/processed"
output_csv = os.path.join(processed_dir, "evaluation_report.csv")

# Ensure directories exist
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

# --- 2. SIDEBAR DEBUGGER ---
st.sidebar.title("🛠️ System Status")
if "GROQ_API_KEY" in st.secrets:
    st.sidebar.success("✅ Cloud Secret: Detected")
elif os.environ.get("GROQ_API_KEY"):
    st.sidebar.success("✅ Local Env: Detected")
else:
    st.sidebar.error("🚨 API Key: MISSING")

st.sidebar.info("Upload resumes and a Job Description to begin the AI analysis.")

# --- 3. MAIN UI ---
st.title("🤖 Yield.ai: Biometric & Ability Based Yield-engine")
st.markdown("---")

# Input Section
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Upload Resumes")
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")

with col2:
    st.subheader("📝 Job Description")
    jd_text = st.text_area("Paste the target JD here...", height=200)

# Run Pipeline Button
if st.button("🚀 Run AI Evaluation", type="primary", use_container_width=True):
    if not uploaded_files or not jd_text:
        st.warning("⚠️ Please upload at least one resume and provide a Job Description.")
    else:
        with st.spinner("Initializing Pipeline..."):
            # Step A: Clean old files safely
            if os.path.exists(raw_dir):
                for file in os.listdir(raw_dir):
                    file_path = os.path.join(raw_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception:
                        pass
            
            # Step B: Save new files
            for uploaded_file in uploaded_files:
                with open(os.path.join(raw_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # Step C: Execute Analysis
            process_resumes_to_csv(raw_dir, output_csv, jd_text)
            st.success("✅ Evaluation Complete! Check the tabs below.")
            time.sleep(1)
            st.rerun()

st.markdown("---")

# --- 4. TABS FOR RESULTS & OPTIMIZATION ---
tab1, tab2 = st.tabs(["🏆 Leaderboard", "✨ AI Resume Optimizer"])

with tab1:
    history_df = get_all_evaluations()
    if not history_df.empty:
        st.subheader("📜 Evaluation History")
        
        # Select a candidate to see their specific analytics
        selected_candidate = st.selectbox("Select Candidate for Detailed Analysis", history_df["Candidate Name"].unique())
        
        # Filter data for the selected person
        candidate_row = history_df[history_df["Candidate Name"] == selected_candidate].iloc[0]
        
        # Create and display the Radar Chart
        chart = create_radar_chart(
            candidate_row["Candidate Name"], 
            candidate_row["Matched Skills"], 
            candidate_row["Missing Skills"]
        )
        st.plotly_chart(chart, use_container_width=True)
        
        # Display the full history table below
        st.markdown("---")
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("No evaluations in history yet.")

    history_df = get_all_evaluations()
    if not history_df.empty:
        st.subheader("📜 Historical Evaluations")
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("No evaluations in history yet.")
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        
        # Display Summary Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Candidates", len(df))
        m2.metric("Avg Match Score", f"{int(df['Match Score (%)'].mean())}%")
        m3.metric("Top Candidate", df.iloc[0]["Candidate Name"])
        
        # Interactive Table
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Download Button
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full CSV Report", data=csv_data, file_name="yield_ai_report.csv", mime="text/csv")
    else:
        st.info("No evaluations found. Upload a resume and click 'Run' to populate this leaderboard.")

with tab2:
    st.header("Bridge the Skill Gap")
    # Use the globally defined output_csv
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        
        # Select Candidate
        selected_name = st.selectbox("Select Candidate to Optimize", df["Candidate Name"])
        candidate_data = df[df["Candidate Name"] == selected_name].iloc[0]
        
        # Show current gaps
        st.write(f"**Missing Skills for {selected_name}:**")
        st.code(candidate_data["Missing Skills"])
        
        if st.button("✨ Generate Optimized Bullet Points"):
            with st.spinner("Llama 3.1 is rewriting your experience..."):
                # We split the string back into a list
                missing_list = str(candidate_data["Missing Skills"]).split(", ")
                
                # Fetch suggestions from our optimizer module
                suggestions = generate_optimized_bullets(missing_list, "Candidate Context")
                
                st.markdown("### 💡 Recommended Bullet Points")
                for i, tip in enumerate(suggestions):
                    st.success(f"**Suggestion {i+1}:** {tip}")
                
                st.caption("Copy these into your resume to increase your Match Score!")
    else:
        st.info("Run an evaluation first to see personalized AI optimization tips.")