import json
from groq import Groq
import os

def generate_optimized_bullets(missing_skills, resume_text):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    prompt = f"""
    You are an expert Technical Career Coach. 
    A candidate is missing these skills for a job: {", ".join(missing_skills)}
    
    Based on their existing resume text, suggest 3 high-impact resume bullet points 
    that subtly incorporate these missing skills or demonstrate transferable experience.
    Use the 'Action Verb + Task + Result' format.
    
    Resume Context: {resume_text[:2000]} 
    
    Return a JSON object with a key 'suggestions' which is a list of strings.
    """
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content).get("suggestions", [])