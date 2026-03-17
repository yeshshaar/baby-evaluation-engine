import os
from typing import List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. Define the exact structure we want back (This replaces your manual JSON parsing!)
class EvaluationResult(BaseModel):
    skill_match_score: int = Field(description="Score from 0-100 based on direct technical keyword overlap.")
    semantic_match_score: int = Field(description="Score from 0-100 based on contextual alignment of past projects.")
    experience_relevance_score: int = Field(description="Score from 0-100 based on career progression and tool seniority.")
    matched_skills: List[str] = Field(description="List of specific skills found in both the resume and JD.")
    missing_skills: List[str] = Field(description="List of critical skills in the JD that are missing from the resume.")

# 2. Build the LangChain Evaluator Function
def run_evaluation_chain(resume_text: str, jd_text: str) -> dict:
    """Runs the LangChain pipeline to evaluate a resume."""
    
    # Grab the API key safely
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing.")

    # Initialize the Groq LLM via LangChain
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )
    
    # Bind our Pydantic model to the LLM so it is forced to return this exact structure
    structured_llm = llm.with_structured_output(EvaluationResult)

    # 3. Create the Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Technical Recruiter evaluating a resume against a Job Description. You must analyze the candidate accurately and extract the requested fields."),
        ("human", "Job Description:\n{jd}\n\nCandidate Resume:\n{resume}")
    ])

    # 4. Create the Chain (Prompt -> LLM)
    chain = prompt | structured_llm

    # 5. Execute the Chain
    result: EvaluationResult = chain.invoke({"jd": jd_text, "resume": resume_text})
    
    # 6. Process the Math directly from the Pydantic object
    overall_score = round(
        (result.skill_match_score * 0.40) + 
        (result.semantic_match_score * 0.35) + 
        (result.experience_relevance_score * 0.25), 1
    )
    
    return {
        "overall_score": overall_score,
        "breakdown": {
            "Skill Match": result.skill_match_score,
            "Semantic Match": result.semantic_match_score,
            "Experience Relevance": result.experience_relevance_score
        },
        "matched_skills": result.matched_skills,
        "missing_skills": result.missing_skills
    }