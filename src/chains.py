import os
from typing import List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.tracers.context import tracing_v2_enabled
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ─── LangSmith tracing config ─────────────────────────────────────────────────
# Set in .env to enable:
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=ls__...
#   LANGCHAIN_PROJECT=yield-ai
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "false"))
os.environ.setdefault("LANGCHAIN_PROJECT",    os.getenv("LANGCHAIN_PROJECT", "yield-ai"))

# ─── Supported models ─────────────────────────────────────────────────────────
AVAILABLE_MODELS = {
    "llama-3.1-8b  (Fast)":     "llama-3.1-8b-instant",
    "llama-3.1-70b (Smart)":    "llama-3.3-70b-versatile",
    "mixtral-8x7b  (Balanced)": "mixtral-8x7b-32768",
}
DEFAULT_MODEL = "llama-3.1-8b-instant"

# ─── Input limits ─────────────────────────────────────────────────────────────
MAX_RESUME_CHARS = 12_000
MAX_JD_CHARS     = 4_000

# ─── Pydantic output schema ───────────────────────────────────────────────────
class EvaluationResult(BaseModel):
    skill_match_score:          int = Field(description="Score 0-100: direct technical keyword overlap.")
    semantic_match_score:       int = Field(description="Score 0-100: contextual alignment of past projects.")
    experience_relevance_score: int = Field(description="Score 0-100: career progression and tool seniority.")
    matched_skills: List[str]       = Field(description="Skills found in both resume and JD.")
    missing_skills: List[str]       = Field(description="Critical JD skills absent from the resume.")

# ─── Chain cache — one chain per model ────────────────────────────────────────
_chains: dict = {}

def _get_chain(model_name: str = DEFAULT_MODEL):
    if model_name not in _chains:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")

        llm = ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=0.2)
        structured_llm = llm.with_structured_output(EvaluationResult)

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert Technical Recruiter evaluating a resume against a Job Description. "
                "Analyze the candidate accurately and extract the requested fields."
            )),
            ("human", "Job Description:\n{jd}\n\nCandidate Resume:\n{resume}")
        ])

        _chains[model_name] = prompt | structured_llm
        print(f"🔧 Built chain for model: {model_name}")

    return _chains[model_name]

# ─── Retry wrapper ────────────────────────────────────────────────────────────
@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True
)
def _invoke_chain(resume_text: str, jd_text: str, model_name: str) -> EvaluationResult:
    return _get_chain(model_name).invoke({"jd": jd_text, "resume": resume_text})

def _zero_result(reason: str = "") -> dict:
    return {
        "overall_score": 0,
        "model_used": "",
        "breakdown": {"Skill Match": 0, "Semantic Match": 0, "Experience Relevance": 0},
        "matched_skills": [],
        "missing_skills": [],
        "error": reason or "LLM evaluation failed."
    }

# ─── Public: structured evaluation ────────────────────────────────────────────
def run_evaluation_chain(
    resume_text: str,
    jd_text: str,
    model_name: str = DEFAULT_MODEL
) -> dict:
    """
    Evaluates a resume against a JD using the selected Groq model.
    Traces every run to LangSmith when LANGCHAIN_TRACING_V2=true.
    """
    resume_text = resume_text[:MAX_RESUME_CHARS]
    jd_text     = jd_text[:MAX_JD_CHARS]

    project = os.environ.get("LANGCHAIN_PROJECT", "yield-ai")
    with tracing_v2_enabled(project_name=project):
        try:
            result: EvaluationResult = _invoke_chain(resume_text, jd_text, model_name)
        except OutputParserException as e:
            print(f"⚠️ OutputParserException: {e}")
            return _zero_result("LLM returned malformed structure.")
        except Exception as e:
            print(f"🚨 Chain failed after retries: {e}")
            return _zero_result(str(e))

    overall_score = round(
        (result.skill_match_score          * 0.40) +
        (result.semantic_match_score       * 0.35) +
        (result.experience_relevance_score * 0.25),
        1
    )

    return {
        "overall_score": overall_score,
        "model_used":    model_name,
        "breakdown": {
            "Skill Match":          result.skill_match_score,
            "Semantic Match":       result.semantic_match_score,
            "Experience Relevance": result.experience_relevance_score,
        },
        "matched_skills": result.matched_skills,
        "missing_skills":  result.missing_skills,
    }


# ─── Public: streaming evaluation ─────────────────────────────────────────────
def stream_evaluation(resume_text: str, jd_text: str, model_name: str = DEFAULT_MODEL):
    """
    Generator that streams raw LLM tokens for the /evaluate/stream endpoint.
    Returns a narrative evaluation (not structured JSON) suitable for live display.
    """
    resume_text = resume_text[:MAX_RESUME_CHARS]
    jd_text     = jd_text[:MAX_JD_CHARS]

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        yield "ERROR: GROQ_API_KEY missing."
        return

    llm = ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=0.2, streaming=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert Technical Recruiter. Analyze the resume vs the JD. "
            "Give a structured evaluation: overall fit, key strengths, skill gaps, "
            "and 3 actionable improvement suggestions. Be concise and specific."
        )),
        ("human", "Job Description:\n{jd}\n\nCandidate Resume:\n{resume}")
    ])

    project = os.environ.get("LANGCHAIN_PROJECT", "yield-ai")
    with tracing_v2_enabled(project_name=project):
        for chunk in (prompt | llm).stream({"jd": jd_text, "resume": resume_text}):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content
