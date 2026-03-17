import json
import os
from groq import Groq


def generate_optimized_bullets(missing_skills, matched_skills, candidate_name="the candidate"):
    """
    Generates resume improvement suggestions based on missing skills.

    Args:
        missing_skills (list): Skills the candidate is lacking for the role.
        matched_skills (str|list): Skills the candidate already has — used as context
                                   so the LLM can suggest realistic bridging bullet points.
        candidate_name (str): Used for logging/context.
    """
    # ✅ BUG FIX: Build real context from matched skills instead of passing
    # the literal string "Candidate Context" to the LLM.
    if isinstance(matched_skills, list):
        matched_str = ", ".join(matched_skills) if matched_skills else "Not available"
    else:
        matched_str = str(matched_skills) if matched_skills not in [None, "", "nan"] else "Not available"

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    prompt = f"""
    You are an expert Technical Career Coach.
    
    The candidate already has experience with: {matched_str}
    They are missing these skills required for the target role: {", ".join(missing_skills)}
    
    Suggest exactly 3 high-impact resume bullet points that:
    - Leverage their existing strengths ({matched_str}) to bridge toward the missing skills
    - Follow the 'Action Verb + Task + Quantified Result' format
    - Sound authentic, not fabricated
    
    Return a JSON object with a single key 'suggestions' containing a list of 3 strings.
    Example: {{"suggestions": ["Led migration of X to Y, reducing Z by 40%", ...]}}
    """

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content).get("suggestions", [])