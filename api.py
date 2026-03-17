from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.chains import run_evaluation_chain
from src.vector_store import save_candidate_to_vector_db, search_similar_candidates
import uvicorn

app = FastAPI(title="Yield-AI Advanced Engine", version="2.0")

# 👉 UPDATE 1: Added candidate_name so the DB knows who this is
class EvaluationRequest(BaseModel):
    candidate_name: str 
    resume_text: str
    jd_text: str

# 👉 UPDATE 2: Added the Search model
class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

@app.get("/")
def read_root():
    return {"status": "Yield-AI API is Online, powered by LangChain & ChromaDB", "version": "2.0"}

@app.post("/evaluate")
async def evaluate_resume(request: EvaluationRequest):
    try:
        # 1. Run the AI Evaluation
        results = run_evaluation_chain(request.resume_text, request.jd_text)
        
        # 2. 👉 UPDATE 3: Save the candidate to the Vector Database automatically
        save_candidate_to_vector_db(
            name=request.candidate_name, 
            resume_text=request.resume_text, 
            score=results["overall_score"]
        )
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 👉 UPDATE 4: Added the missing /search endpoint!
@app.post("/search")
async def semantic_search(request: SearchRequest):
    """Finds candidates matching a natural language query."""
    try:
        matches = search_similar_candidates(request.query, request.top_k)
        return {"query": request.query, "results": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 👉 UPDATE 5: Hot-reloading enabled
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)