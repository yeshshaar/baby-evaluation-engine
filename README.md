[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](https://yield-ai.streamlit.app)

# Yield.ai(B.A.B.Y. (Biometric & Ability Based Yield-engine))
### *A Privacy-First AI Evaluation Engine for Automated Resume Ranking*

**B.A.B.Y.** (under the product name **Yield.ai**) is a full-stack Machine Learning Engineering project designed to automate the evaluation of resumes against specific Job Descriptions (JDs). It bridges the gap between raw text extraction and actionable recruitment insights using LLMs, real-time telemetry, and secure session isolation.

---

## 🚀 **Key Features**
* **AI-Powered Scoring:** Uses **Llama 3.1 (8B/70B)** via the Groq API to extract skills and rank candidates based on JD relevance.
* **Real-time Telemetry:** Custom callback implementation provides live progress updates and file-name tracking during batch processing.
* **Privacy-by-Design:** Implements **UUID-based session isolation**, ensuring that user data is sandboxed and cleared upon session termination.
* **Interactive Analytics:** Visualizes candidate "Skill Gaps" using **Plotly Radar Charts** for multi-dimensional ability assessment.
* **AI Optimizer:** Generates context-aware resume bullet points to help candidates bridge the gap between their current profile and target roles.

---

## 🏗️ **System Architecture**
The system is designed with a decoupled architecture, separating the core AI logic from the UI layer to allow for future scalability.



### **Tech Stack**
* **LLM Orchestration:** Groq API (Llama 3.1)
* **Backend:** Python 3.10+
* **UI Framework:** Streamlit
* **Database:** SQLite (for historical persistence)
* **Data Science:** Pandas, Plotly (for analytics)
* **Security:** UUID Session Management & Streamlit Secrets

---

## 🛡️ **Data Governance & Privacy**
As a project handling PII (Personally Identifiable Information), B.A.B.Y. implements:
1.  **Session Isolation:** Every visitor is assigned a unique UUID. PDFs and analysis results are stored in temporary, isolated directories.
2.  **Admin Access Layer:** Historical data is protected via an authentication layer, preventing unauthorized access to candidate history.
3.  **Sanitized Extraction:** The engine is built to focus on skills and tools, reducing bias by ignoring non-relevant personal identifiers.

---

## 🛠️ **Installation & Setup**

1. **Clone the repository:**
   
   git clone [https://github.com/your-username/baby-evaluation-engine.git](https://github.com/yeshshaar/baby-evaluation-engine.git)
   cd baby-evaluation-engine

2. **Install dependencies:**

    pip install -r requirements.txt

3. **Configure Secrets:**

    Create a .streamlit/secrets.toml file and add:

    GROQ_API_KEY = "your_key_here"
    ADMIN_PASSWORD = "your_password_here"

4. **Run the Dashboard:**
    streamlit run app/dashboard.py