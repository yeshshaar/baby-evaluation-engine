import sys
import os
import pandas as pd
import streamlit as st
import time
import uuid
import requests 


# --- 1. PAGE CONFIG 
st.set_page_config(page_title="Yield.ai | MLE Evaluation Engine", layout="wide")

# --- 2. GLOBAL "NEON SAAS" CSS INJECTION ---
st.markdown("""
    <style>
    /* Global App Background */
    .stApp {
        background: radial-gradient(circle at top left, #12142b 0%, #070814 100%);
    }

    /* Gradient Glowing Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00ffcc 0%, #8a2be2 100%);
        color: white !important;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(138, 43, 226, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(0, 255, 204, 0.6);
        transform: translateY(-2px);
    }

    /* Glassmorphism Expanders & Containers */
    div[data-testid="stExpander"] {
        background: rgba(20, 22, 43, 0.6);
        border: 1px solid rgba(0, 255, 204, 0.2);
        border-radius: 12px;
        backdrop-filter: blur(4px);
    }

    /* Clean typography */
    h1, h2, h3 {
        font-weight: 300 !important;
        letter-spacing: 1px;
    }

    /* Scorecard: Glowing Header */
    .overall-score-glow {
        font-size: 24px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 0 0 10px rgba(138, 43, 226, 0.6);
        font-family: sans-serif;
        font-weight: 300;
    }
    .overall-score-glow span { color: #8a2be2; font-weight: bold; font-size: 32px; }

    /* Scorecard: 3 Feature Boxes */
    .feature-box {
        background-color: rgba(10, 12, 30, 0.6);
        border: 1px solid rgba(0, 255, 204, 0.2);
        border-radius: 12px;
        padding: 24px 16px;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        backdrop-filter: blur(5px);
    }
    .feature-box:hover {
        border-color: #00ffcc;
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.3);
        transform: translateY(-5px);
    }
    .feature-icon { font-size: 32px; margin-bottom: 12px; display: block; }
    .feature-title { color: #00ffcc; font-size: 15px; font-weight: 500; margin-bottom: 8px; font-family: sans-serif; letter-spacing: 0.5px; }
    .feature-value { color: #ffffff; font-size: 36px; font-weight: 700; margin: 0; font-family: sans-serif; }
    .feature-desc { color: #8a8d9e; font-size: 12px; margin-top: 12px; line-height: 1.4; font-family: sans-serif; }
    
    /* Scorecard: Neon Glass Tags */
    .matched-tag { 
        background-color: rgba(0, 255, 204, 0.05); 
        color: #00ffcc; 
        padding: 6px 14px; 
        border-radius: 20px; 
        margin: 4px; 
        display: inline-block; 
        font-size: 13px; 
        border: 1px solid rgba(0, 255, 204, 0.5);
        box-shadow: 0 0 10px rgba(0, 255, 204, 0.15);
        transition: all 0.2s ease;
    }
    .matched-tag:hover {
        background-color: rgba(0, 255, 204, 0.15); 
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.4);
    }

    .missing-tag { 
        background-color: rgba(255, 51, 102, 0.05); 
        color: #ff3366; 
        padding: 6px 14px; 
        border-radius: 20px; 
        margin: 4px; 
        display: inline-block; 
        font-size: 13px; 
        border: 1px solid rgba(255, 51, 102, 0.5);
        box-shadow: 0 0 10px rgba(255, 51, 102, 0.15);
        transition: all 0.2s ease;
    }
    .missing-tag:hover {
        background-color: rgba(255, 51, 102, 0.15); 
        box-shadow: 0 0 15px rgba(255, 51, 102, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. PATH INJECTION & SESSION LOGIC ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

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

init_db()

# --- 5. UI COMPONENTS ---
def render_scorecard(candidate_name, row_data):
    """Draws the professional SaaS-style dashboard using the CSV/DB row data."""
    row_dict = dict(row_data)

    st.subheader(f"📄 Evaluation: {candidate_name}")
    
    overall = row_dict.get("Score", 0)
    skill_m = row_dict.get("Skill Match", 0)
    sem_m = row_dict.get("Semantic Match", 0)
    exp_m = row_dict.get("Experience Relevance", 0)

    # Glowing Overall Score
    st.markdown(f'<div class="overall-score-glow">Yield-AI Confidence Score: <span>{overall}%</span></div>', unsafe_allow_html=True)

    # 3 Feature Boxes
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="feature-box">
            <span class="feature-icon">🎯</span>
            <div class="feature-title">Skill Match</div>
            <div class="feature-value">{skill_m}%</div>
            <div class="feature-desc">Direct overlap of technical JD keywords.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="feature-box" style="border-color: rgba(138, 43, 226, 0.3);">
            <span class="feature-icon">🧠</span>
            <div class="feature-title" style="color: #8a2be2;">Semantic Match</div>
            <div class="feature-value">{sem_m}%</div>
            <div class="feature-desc">Llama 3.1 contextual project alignment.</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="feature-box" style="border-color: rgba(0, 153, 255, 0.3);">
            <span class="feature-icon">📈</span>
            <div class="feature-title" style="color: #0099ff;">Experience Relevance</div>
            <div class="feature-value">{exp_m}%</div>
            <div class="feature-desc">Career progression and tool seniority.</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("") 
    
    with st.expander("🔍 How is this score calculated?"):
        st.write("""
        **The Yield-AI Engine uses a weighted algorithm to ensure a balanced evaluation:**
        * **Skill Match (40%):** Direct overlap of technical keywords identified in the JD.
        * **Semantic Match (35%):** Contextual relevance using Llama 3.1 to understand if past projects align with the target role.
        * **Experience Relevance (25%):** Analysis of career progression and time spent with core technologies.
        """)

    # Glassmorphism Skill Tags
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
    time.sleep(1)
    st.rerun()

# --- 7. MAIN UI ---
st.title("🤖 Yield.ai")
st.markdown("---")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📄 Upload Resumes")
    uploaded_files = st.file_uploader("Choose PDF or Word files", accept_multiple_files=True, type=["pdf", "docx"])
    st.subheader("📝 Job Description")
    jd_text = st.text_area("Paste the target JD here...", height=200)

if st.button("🚀 Run AI Evaluation", type="primary", use_container_width=True):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(os.path.join(raw_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
    
    files_to_process = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.pdf', '.docx'))]
    
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
# 👉 We added a third tab here!
tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard", "✨ AI Resume Optimizer", "🧠 Semantic Search"])

with tab1:
    st.header("Session Analysis")
    
    with st.expander("🔓 Admin Access (View Global History)"):
        pw = st.text_input("Enter Admin Password", type="password")
    
    display_df = pd.DataFrame()
    admin_pw = ""
    try:
        admin_pw = st.secrets["ADMIN_PASSWORD"]
    except Exception:
        pass 

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
        
        render_scorecard(candidate_row["Candidate Name"], candidate_row)
        
        st.markdown("---")
        st.subheader("📊 Skill Gap Visualization")
        chart = create_radar_chart(
            candidate_row["Candidate Name"], 
            candidate_row["Matched Skills"], 
            candidate_row["Missing Skills"]
        )
        st.plotly_chart(chart, width="stretch")
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

# 👉 THE NEW VECTOR SEARCH UI
with tab3:
    st.header("Talent Intelligence Search")
    st.markdown("Query your Vector Database using natural language to find the perfect candidate match.")
    
    search_query = st.text_input("🔍 What are you looking for?", placeholder="e.g., 'A senior backend engineer with deep Python and Docker experience'")
    
    if st.button("Search Database", type="primary"):
        if search_query:
            with st.spinner("Searching Vector Space..."):
                response = None
                try:
                    # 1. Try the internal Docker network first
                    primary_url = os.environ.get("API_URL", "http://api:8000/evaluate").replace("/evaluate", "/search")
                    response = requests.post(primary_url, json={"query": search_query, "top_k": 3}, timeout=10)
                except requests.exceptions.ConnectionError:
                    try:
                        # 2. Fallback: Try the local Mac network if Docker DNS fails
                        fallback_url = "http://127.0.0.1:8000/search"
                        response = requests.post(fallback_url, json={"query": search_query, "top_k": 3}, timeout=10)
                    except Exception as e:
                        st.error(f"Could not connect to API on any network: {e}")
                
                if response and response.status_code == 200:
                    results = response.json().get("results", [])
                    if results:
                        for idx, res in enumerate(results):
                            st.markdown(f"### {idx + 1}. {res['name']}")
                            st.write(f"**Yield-AI Score:** {res['score']}%")
                            st.caption(f"Vector Match Distance: {res['match_distance']}") 
                            st.markdown("---")
                    else:
                        st.warning("No matching candidates found in the database. (Try running an evaluation first!)")
                elif response:
                    st.error(f"API Error: {response.status_code} - {response.text}")
        else:
            st.warning("Please enter a search query.")