from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_skill_match(resume_skills, jd_skills):
    """Converts skills to vectors and calculates semantic similarity."""
    print("Loading embedding model (this might take a few seconds the first time)...")
    
    # 'all-MiniLM-L6-v2' is a lightweight, incredibly fast industry-standard model for this
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Convert the lists of skills into single strings of text
    resume_text = " ".join(resume_skills)
    jd_text = " ".join(jd_skills)

    # The ML Magic: Convert the text into mathematical vectors (lists of numbers)
    resume_vector = model.encode([resume_text])
    jd_vector = model.encode([jd_text])

    # Calculate the angle (cosine similarity) between the two vectors
    similarity = cosine_similarity(resume_vector, jd_vector)[0][0]

    # Convert the raw decimal score into a clean percentage
    match_percentage = round(similarity * 100, 2)
    return match_percentage

# --- Testing the Scoring Engine ---
if __name__ == "__main__":
    # Let's pretend this is a Job Description for an MLE role
    jd_skills = ["Python", "Machine Learning", "NLP", "AWS", "Docker", "Vector Databases"]
    
    # These are the exact skills your AI parser just pulled from your resume!
    candidate_skills = [
        "Python", "Java", "C++", "Scikit-Learn", "TensorFlow", "CNNs", 
        "Model Evaluation", "Semantic Matching", "Skill Extraction", 
        "Explainable AI", "LLaMA", "Prompt Engineering"
    ]

    print(f"Target JD Skills: {jd_skills}\n")
    print(f"Candidate Skills: {candidate_skills}\n")
    
    score = calculate_skill_match(candidate_skills, jd_skills)
    
    print(f"--- CANDIDATE MATCH SCORE: {score}% ---")