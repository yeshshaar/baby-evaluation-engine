from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the model outside the function so it doesn't reload for every single skill
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_skill_match(candidate_skills, jd_skills):
    """Calculates score and identifies matched/missing skills using semantic vectors."""
    if not candidate_skills or not jd_skills:
        return 0, [], jd_skills, "Missing data for evaluation."

    # 1. Overall Score Calculation
    resume_text = " ".join(candidate_skills)
    jd_text = " ".join(jd_skills)
    
    resume_vector = model.encode([resume_text])
    jd_vector = model.encode([jd_text])
    
    match_score = round(cosine_similarity(resume_vector, jd_vector)[0][0] * 100, 2)

    # 2. Explainable AI: Identify Matched vs. Missing Skills
    matched_skills = []
    missing_skills = []
    
    # Convert all candidate skills to vectors at once
    candidate_vectors = model.encode(candidate_skills)
    
    for jd_skill in jd_skills:
        jd_vec = model.encode([jd_skill])
        # Compare this one JD skill against ALL candidate skills
        similarities = cosine_similarity(jd_vec, candidate_vectors)[0]
        
        # If the highest similarity is above 0.4 (40%), we consider it a semantic match!
        if max(similarities) > 0.4: 
            matched_skills.append(jd_skill)
        else:
            missing_skills.append(jd_skill)

    # 3. Generate "How to Improve" Feedback
    if missing_skills:
        improvement = f"Focus on adding projects or experience related to: {', '.join(missing_skills)}"
    else:
        improvement = "Candidate is a highly aligned match for this role!"

    return match_score, matched_skills, missing_skills, improvement