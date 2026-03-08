# 👶 B.A.B.Y. (Biometric & Ability Based Yield-engine)
**An AI-Powered Candidate Evaluation & Recommendation System**

## 📖 Project Overview
B.A.B.Y. is an end-to-end Machine Learning pipeline designed to automate and objectify the resume screening process. It ingests unstructured PDF resumes, leverages Large Language Models (LLMs) for entity extraction, and utilizes vector embeddings to semantically score candidates against Job Descriptions.

## 🏗️ System Architecture
1. **Data Ingestion:** `PyMuPDF` extracts raw text from PDF documents.
2. **Generative AI Parsing:** Groq API (Llama 3.1 8B) structures the raw text into highly accurate JSON objects using strict prompt engineering.
3. **Semantic Scoring Engine:** Hugging Face `SentenceTransformers` (`all-MiniLM-L6-v2`) converts extracted skills into vector embeddings. The engine calculates the **Cosine Similarity** between candidate vectors and target JD vectors to generate a fair, mathematical match percentage.
4. **Interactive UI:** A `Streamlit` dashboard provides recruiters with a real-time leaderboard and CSV export functionality.

## 🚀 Tech Stack
* **Language:** Python 3
* **AI/ML:** Llama 3.1 (Groq API), SentenceTransformers, Scikit-Learn
* **Data Engineering:** Pandas, PyMuPDF (fitz)
* **Frontend:** Streamlit

## ⚙️ Local Setup
To run this engine locally:
1. Clone the repository.
2. Create a virtual environment and install requirements: `pip install -r requirements.txt`
3. Create a `.env` file in the root directory and add your Groq API key: `GROQ_API_KEY=your_key_here`
4. Run the application: `streamlit run app/dashboard.py`