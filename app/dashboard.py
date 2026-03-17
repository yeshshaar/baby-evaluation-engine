import sys
import os
import pandas as pd
import streamlit as st
import time
import uuid

# --- 1. PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(page_title="Yield.ai | MLE Evaluation Engine", layout="wide")

# --- 2. GLOBAL "NEON SAAS" CSS INJECTION ---
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #12142b 0%, #070814 100%);
    }
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
    div[data-testid="stExpander"] {
        background: rgba(20, 22, 43, 0.6);
        border: 1px solid rgba(0, 255, 204, 0.2);
        border-radius: 12px;
        backdrop-filter: blur(4px);
    }
    h1, h2, h3 {
        font-weight: 300 !important;
        letter-spacing: 1px;
    }
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

    /* Leaderboard table styling */
    .leaderboard-table {
        width: 100%;
        border-collapse: collapse;
        font-family: sans-serif;
        margin-bottom: 24px;
    }
    .leaderboard-table th {
        color: #00ffcc;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 10px 16px;
        border-bottom: 1px solid rgba(0, 255, 204, 0.3);
        text-align: left;
    }
    .leaderboard-table td {
        padding: 12px 16px;
        color: #e0e0e0;
        font-size: 14px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .leaderboard-table tr:hover td {
        background: rgba(0, 255, 204, 0.04);
    }
    .rank-badge {
        font-weight: bold;
        font-size: 16px;
    }
    .score-bar-wrap {
        background: rgba(255,255,255,0.07);
        border-radius: 20px;
        height: 8px;
        width: 100%;
        min-width: 120px;
    }
    .score-bar-fill {
        height: 8px;
        border-radius: 20px;
        background: linear-gradient(90deg, #00ffcc, #8a2be2);
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

# --- 5. HELPER: SCORE → COLOR ---
def score_color(score):
    """Returns a CSS color based on the score value."""
    if score >= 75:
        return "#00ffcc"
    elif score >= 50:
        return "#f5a623"
    else:
        return "#ff3366"

def rank_emoji(rank):
    medals = ["🥇", "🥈", "🥉"]
    return medals[rank - 1] if rank - 1 < len(medals) else f"#{rank}"

# --- 6. UI COMPONENT: SCORECARD ---
def render_scorecard(candidate_name, row_data):
    row_dict = dict(row_data)
    st.subheader(f"📄 Evaluation: {candidate_name}")

    overall = row_dict.get("Score", 0)
    skill_m = row_dict.get("Skill Match", 0)
    sem_m = row_dict.get("Semantic Match", 0)
    exp_m = row_dict.get("Experience Relevance", 0)

    st.markdown(f'<div class="overall-score-glow">Yield-AI Confidence Score: <span>{overall}%</span></div>', unsafe_allow_html=True)

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


# --- 7. UI COMPONENT: LEADERBOARD TABLE ---
def render_leaderboard(df):
    """Renders a ranked leaderboard table with score bars and color coding."""
    st.markdown("### 🏆 Candidate Rankings")

    ranked = df.sort_values(by="Score", ascending=False).reset_index(drop=True)

    # ✅ FIX: Build each row as a single compact line — multi-line indented HTML
    # inside f-strings triggers Streamlit's markdown code-block detection (4-space rule).
    rows = []
    for i, row in ranked.iterrows():
        rank = i + 1
        score = float(row.get("Score", 0))
        color = score_color(score)
        bar_width = min(int(score), 100)
        skill = row.get("Skill Match", 0)
        semantic = row.get("Semantic Match", 0)
        experience = row.get("Experience Relevance", 0)
        name = row["Candidate Name"]

        bar = f'<div style="display:flex;align-items:center;gap:10px;"><div class="score-bar-wrap"><div class="score-bar-fill" style="width:{bar_width}%;"></div></div><span style="color:{color};font-weight:700;min-width:40px;">{score}%</span></div>'
        rows.append(
            f'<tr>'
            f'<td><span class="rank-badge" style="color:{color};">{rank_emoji(rank)}</span></td>'
            f'<td><strong style="color:#ffffff;">{name}</strong></td>'
            f'<td>{bar}</td>'
            f'<td style="color:#8a8d9e;">{skill}%</td>'
            f'<td style="color:#8a2be2;">{semantic}%</td>'
            f'<td style="color:#0099ff;">{experience}%</td>'
            f'</tr>'
        )

    header = '<tr><th>Rank</th><th>Candidate</th><th>Overall Score</th><th>🎯 Skill</th><th>🧠 Semantic</th><th>📈 Experience</th></tr>'
    table_html = f'<table class="leaderboard-table"><thead>{header}</thead><tbody>{"".join(rows)}</tbody></table>'
    st.markdown(table_html, unsafe_allow_html=True)


# --- 8. SIDEBAR ---
st.sidebar.title("🛠️ System Status")

# ✅ NEW: Multi-model selector
from src.chains import AVAILABLE_MODELS, DEFAULT_MODEL
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Model Selection")
selected_model_label = st.sidebar.selectbox(
    "LLM Backend",
    options=list(AVAILABLE_MODELS.keys()),
    index=0,
    help="Switch models to trade off speed vs accuracy. Scores may vary between models."
)
selected_model = AVAILABLE_MODELS[selected_model_label]
st.sidebar.caption(f"`{selected_model}`")

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear My Session", type="secondary"):
    import shutil
    if os.path.exists(session_base):
        shutil.rmtree(session_base)
    st.sidebar.success("Session Cleared!")
    time.sleep(1)
    st.rerun()

# --- 9. MAIN UI ---
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

        with st.spinner(f"Pipeline active · model: `{selected_model}`..."):
            process_resumes_to_csv(
                raw_dir, output_csv, jd_text,
                progress_callback=update_ui_callback,
                model_name=selected_model          # ✅ pass selected model
            )

        status_text.success(f"✅ Success! {len(files_to_process)} resumes analyzed.")
        time.sleep(1)
        st.rerun()

# --- 10. TABS ---
tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard", "✨ AI Resume Optimizer", "⚡ Live Analysis"])

with tab1:
    st.header("Session Analysis")

    with st.expander("🔓 Admin Access (View Global History)"):
        pw = st.text_input("Enter Admin Password", type="password")

    display_df = pd.DataFrame()

    # ✅ BUG FIX: Guard against empty admin_pw granting access when secrets aren't set
    admin_pw = ""
    try:
        admin_pw = st.secrets.get("ADMIN_PASSWORD", "")
    except Exception:
        pass

    admin_unlocked = pw and admin_pw and pw == admin_pw

    if admin_unlocked:
        st.success("Admin Mode: Showing all historical data.")
        display_df = get_all_evaluations()
        # Normalize column names from DB (may differ slightly)
        col_map = {"Match Score (%)": "Score"}
        display_df = display_df.rename(columns=col_map)
    else:
        if os.path.exists(output_csv):
            display_df = pd.read_csv(output_csv)
            st.info("Showing current session results only.")

    if not display_df.empty:
        # ✅ NEW: Ranked leaderboard table
        render_leaderboard(display_df)

        st.markdown("---")

        selected_candidate = st.selectbox("Select Candidate for Deep Dive", display_df["Candidate Name"].unique())
        candidate_row = display_df[display_df["Candidate Name"] == selected_candidate].iloc[0]

        render_scorecard(candidate_row["Candidate Name"], candidate_row)

        st.markdown("---")

        st.subheader("📊 Skill Gap Visualization")
        # ✅ BUG FIX: Pass the 3 score dimensions, not raw skill strings
        chart = create_radar_chart(
            candidate_row["Candidate Name"],
            skill_match=candidate_row.get("Skill Match", 0),
            semantic_match=candidate_row.get("Semantic Match", 0),
            experience_relevance=candidate_row.get("Experience Relevance", 0)
        )
        st.plotly_chart(chart, use_container_width=True)

        with st.expander("View Raw Data"):
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No evaluations to display yet. Upload resumes and run an evaluation to get started.")

with tab2:
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        selected_name = st.selectbox("Select Candidate to Optimize", df["Candidate Name"], key="opt_select")
        candidate_data = df[df["Candidate Name"] == selected_name].iloc[0]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**✅ Matched Skills:** {candidate_data.get('Matched Skills', 'N/A')}")
        with col_b:
            st.markdown(f"**❌ Missing Skills:** {candidate_data.get('Missing Skills', 'N/A')}")

        if st.button("✨ Generate Optimized Bullet Points"):
            with st.spinner("Coaching the candidate..."):
                missing_list = [s.strip() for s in str(candidate_data["Missing Skills"]).split(",") if s.strip() and s.strip().lower() != "nan"]
                matched_raw = candidate_data.get("Matched Skills", "")

                # ✅ BUG FIX: Pass actual matched skills as context (not "Candidate Context")
                suggestions = generate_optimized_bullets(
                    missing_skills=missing_list,
                    matched_skills=matched_raw,
                    candidate_name=selected_name
                )
                st.markdown("---")
                for i, tip in enumerate(suggestions):
                    st.success(f"**Suggestion {i+1}:** {tip}")
    else:
        st.info("Run an evaluation first to see optimization tips.")

# ✅ NEW: Tab 3 — Live streaming analysis
with tab3:
    st.header("⚡ Live Narrative Analysis")
    st.caption("Get a real-time, token-by-token evaluation streamed directly from the LLM — no waiting for the full response.")

    from src.chains import stream_evaluation
    from src.sanitizer import clean_pii as _clean

    stream_resume = st.text_area("Paste resume text here", height=200, key="stream_resume")
    stream_jd     = st.text_area("Paste job description here", height=150, key="stream_jd")

    if st.button("⚡ Stream Live Analysis", type="primary"):
        if not stream_resume.strip() or not stream_jd.strip():
            st.warning("⚠️ Please paste both a resume and a JD.")
        else:
            st.markdown("---")
            st.markdown(f"**Model:** `{selected_model}`")
            output_box = st.empty()
            full_text  = ""
            with st.spinner("Connecting to LLM stream..."):
                for token in stream_evaluation(
                    resume_text=_clean(stream_resume),
                    jd_text=stream_jd,
                    model_name=selected_model
                ):
                    full_text += token
                    output_box.markdown(full_text + "▌")   # blinking cursor effect
            output_box.markdown(full_text)                 # final render without cursor