import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_data")

vector_db = Chroma(
    collection_name="yield_candidates",
    embedding_function=embeddings,
    persist_directory=DB_DIR
)

def save_candidate_to_vector_db(name: str, resume_text: str, score: float):
    doc = Document(
        page_content=resume_text,
        metadata={"candidate_name": name, "overall_score": score}
    )
    vector_db.add_documents([doc])
    return True

def search_similar_candidates(query: str, top_k: int = 3):
    results = vector_db.similarity_search_with_score(query, k=top_k)
    search_results = []
    for doc, distance in results:
        search_results.append({
            "name": doc.metadata["candidate_name"],
            "score": doc.metadata["overall_score"],
            "match_distance": round(distance, 4) 
        })
    return search_results