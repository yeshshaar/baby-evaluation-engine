import sys
import os
import pandas as pd
import streamlit as st
import time
import uuid

# --- 1. PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(page_title="Yield.ai | MLE Evaluation Engine", layout="wide")

# --- 2. PATH INJECTION ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- 3. SESSION-BASED PRIVACY LOGIC ---
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

# Create a private sandbox folder for this specific visitor
session_base = os.path.join("data", "sessions", st.session_state['session_id'])
raw_dir = os.path.join(session_base, "raw")
processed_dir = os.path.join(session_base, "processed")

os.makedirs(raw_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

output_csv = os.path.join(processed_dir, "evaluation_report.csv")

# --- 4. BACKEND IMPORTS ---
from src.main import process_resumes_to_csv
from src.database import init_db, get_all_evaluations
from src.visualizer import create_radar_chart
from src.optimizer import generate_optimized_bullets

# Initialize the global DB structure if not exists
init_db()

# --- 5. UI COMPONENTS ---
def render_scorecard(candidate_name, row_data):
    """Draws the professional SaaS-style dashboard using the CSV/DB row data."""
    
    # Convert the Pandas Series to a standard dictionary to prevent extraction errors
    row_dict = dict(row_data)

    st.markdown("""
        <style>
        .matched-tag { background-color: #e6fffa; color: #2c7a7b; padding: 4px 10px; border-radius: 15px; margin: 3px; display: inline-block; font-size: 14px; border: 1px solid #81e6d9;}
        .missing-tag { background-color: #fff5f5; color: #c53030; padding: 4px 10px; border-radius: 15px; margin: 3px; display: inline-block; font-size: 14px; border: 1px solid #feb2b2;}
        </style>
    """, unsafe_allow_html=True)

    st.subheader(f"📄 Evaluation: {candidate_name}")
    
    # 👇 THIS IS THE MISSING PART THAT CAUSED THE CRASH 👇
    overall = row_dict.get("Score", 0)
    skill_m = row_dict.get("Skill Match", 0)
    sem_m = row_dict.get("Semantic Match", 0)
    exp_m = row_dict.get("Experience Relevance", 0)
    # 👆 ---------------------------------------------- 👆

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Score", f"{overall}%")
    col2.metric("Skill Match", f"{skill_m}%")
    col3.metric("Semantic Match", f"{sem_m}%")
    col4.metric("Experience", f"{exp_m}%")
    
    st.divider() 
    
    # The Explainer UI
    with st.expander("🔍 How is this score calculated?"):
        st.write("""
        **The Yield-AI Engine uses a weighted algorithm to ensure a balanced evaluation:**
        * **Skill Match (40%):** Direct overlap of technical keywords identified in the JD.
        * **Semantic Match (35%):** Contextual relevance using Llama 3.1 to understand if past projects align with the target role.
        * **Experience Relevance (25%):** Analysis of career progression and time spent with core technologies.
        """)

    # Safely parse skills strings into lists
    matched_skills = str(row_dict.get("Matched Skills", "")).split(", ")
    missing_skills = str(row_dict.get("Missing Skills", "")).split(", ")

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("**✅ Matched Skills**")
        if matched_skills and matched_skills[0] not in ["nan", "", "None"]:
            matched_html = "".join([f'<span class="matched-tag">{skill}</span>' for skill in matched_skills])
            st.markdown(matched_html, unsafe_allow_html=True)
        else:
            st.write("None found.")
            
    with right_col:
        st.markdown("**❌ Missing Skills**")
        if missing_skills and missing_skills[0] not in ["nan", "", "None"]:
            missing_html = "".join([f'<span class="missing-tag">{skill}</span>' for skill in missing_skills])
            st.markdown(missing_html, unsafe_allow_html=True)
        else:
            st.write("No missing skills identified.")

# --- 6. SIDEBAR ---
st.sidebar.title("🛠️ System Status")
if st.sidebar.button("🗑️ Clear My Session", type="secondary"):
    for f in os.listdir(raw_dir):
        os.remove(os.path.join(raw_dir, f))
    if os.path.exists(output_csv):
        os.remove(output_csv)
    st.sidebar.success("Session Cleared!")
    st.rerun()

# --- 7. MAIN UI ---
st.title("🤖 Yield.ai")
st.markdown("---")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📄 Upload Resumes")
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")
with col2:
    st.subheader("📝 Job Description")
    jd_text = st.text_area("Paste the target JD here...", height=200)

if st.button("🚀 Run AI Evaluation", type="primary", use_container_width=True):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(os.path.join(raw_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
    
    files_to_process = [f for f in os.listdir(raw_dir) if f.endswith(".pdf")]
    
    if not files_to_process:
        st.warning("⚠️ No resumes found.")
    elif not jd_text:
        st.warning("⚠️ Please paste a Job Description.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_ui_callback(current_index, total, filename):
            percent = int(((current_index + 1) / total) * 100)
            progress_bar.progress(percent)
            status_text.text(f"🧪 Analyzing ({current_index + 1}/{total}): {filename}...")

        with st.spinner("Pipeline active..."):
            process_resumes_to_csv(raw_dir, output_csv, jd_text, progress_callback=update_ui_callback)
            
        status_text.success(f"✅ Success! {len(files_to_process)} resumes analyzed.")
        time.sleep(1)
        st.rerun()

# --- 8. TABS ---
tab1, tab2 = st.tabs(["🏆 Leaderboard", "✨ AI Resume Optimizer"])

with tab1:
    st.header("Session Analysis")
    
    with st.expander("🔓 Admin Access (View Global History)"):
        pw = st.text_input("Enter Admin Password", type="password")
    
    # Initialize display_df
    display_df = pd.DataFrame()
    
    # Handle Secrets safely
    admin_pw = ""
    try:
        admin_pw = st.secrets["ADMIN_PASSWORD"]
    except Exception:
        pass # Streamlit secrets not found locally, bypass for dev

    if pw and pw == admin_pw:
        st.success("Admin Mode: Showing all historical data.")
        display_df = get_all_evaluations()
    else:
        if os.path.exists(output_csv):
            display_df = pd.read_csv(output_csv)
            st.info("Showing current session results only.")

    if not display_df.empty:
        selected_candidate = st.selectbox("Select Candidate", display_df["Candidate Name"].unique())
        candidate_row = display_df[display_df["Candidate Name"] == selected_candidate].iloc[0]
        
        # 1. Show the New Professional Scorecard
        render_scorecard(candidate_row["Candidate Name"], candidate_row)
        
        st.markdown("---")
        
        # 2. Show the Radar Chart visually
        st.subheader("📊 Skill Gap Visualization")
        chart = create_radar_chart(
            candidate_row["Candidate Name"], 
            candidate_row["Matched Skills"], 
            candidate_row["Missing Skills"]
        )
        st.plotly_chart(chart, use_container_width=True)
        
        # 3. Show the raw dataframe
        with st.expander("View Raw Data"):
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No evaluations to display yet.")

with tab2:
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        selected_name = st.selectbox("Select Candidate to Optimize", df["Candidate Name"], key="opt_select")
        candidate_data = df[df["Candidate Name"] == selected_name].iloc[0]
        
        st.write(f"**Missing Skills:** {candidate_data['Missing Skills']}")
        
        if st.button("✨ Generate Optimized Bullet Points"):
            with st.spinner("Analyzing..."):
                missing_list = str(candidate_data["Missing Skills"]).split(", ")
                suggestions = generate_optimized_bullets(missing_list, "Candidate Context")
                for i, tip in enumerate(suggestions):
                    st.success(f"**Suggestion {i+1}:** {tip}")
    else:
        st.info("Run an evaluation first to see optimization tips.")