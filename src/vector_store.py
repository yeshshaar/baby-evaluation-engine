"""
vector_store.py — Pluggable vector store with Chroma / Pinecone / Qdrant support.

Set VECTOR_DB in your .env to switch backends:
  VECTOR_DB=chroma    (default, local)
  VECTOR_DB=pinecone  (requires PINECONE_API_KEY, PINECONE_INDEX)
  VECTOR_DB=qdrant    (requires QDRANT_URL, QDRANT_API_KEY)
"""

import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from src.sanitizer import clean_pii

VECTOR_DB_BACKEND = os.environ.get("VECTOR_DB", "chroma").lower()

# ─── Lazy singletons ──────────────────────────────────────────────────────────
_embeddings = None
_db         = None

def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        print("🔧 Loading embedding model...")
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings

def _get_db():
    global _db
    if _db is None:
        backend = VECTOR_DB_BACKEND
        print(f"🔧 Initialising vector store: {backend}")

        if backend == "pinecone":
            _db = _init_pinecone()
        elif backend == "qdrant":
            _db = _init_qdrant()
        else:
            _db = _init_chroma()

    return _db

# ─── Backend initialisers ─────────────────────────────────────────────────────
def _init_chroma():
    from langchain_chroma import Chroma
    db_dir = os.environ.get(
        "CHROMA_DB_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_data")
    )
    return Chroma(
        collection_name="yield_candidates",
        embedding_function=_get_embeddings(),
        persist_directory=db_dir
    )

def _init_pinecone():
    """
    Requires:
      PINECONE_API_KEY — from app.pinecone.io
      PINECONE_INDEX   — name of your index (dimension=384 for all-MiniLM-L6-v2)
    """
    try:
        from langchain_pinecone import PineconeVectorStore
        import pinecone
    except ImportError:
        raise ImportError("Run: pip install langchain-pinecone pinecone-client")

    pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX"])
    return PineconeVectorStore(index=index, embedding=_get_embeddings())

def _init_qdrant():
    """
    Requires:
      QDRANT_URL     — e.g. https://xyz.qdrant.io:6333
      QDRANT_API_KEY — from cloud.qdrant.io
    """
    try:
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
    except ImportError:
        raise ImportError("Run: pip install langchain-qdrant qdrant-client")

    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY")
    )

    collection = "yield_candidates"
    # Create collection if it doesn't exist
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    return QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=_get_embeddings()
    )

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _l2_to_relevance(distance: float) -> float:
    relevance = max(0.0, 1.0 - (distance / 2.0))
    return round(relevance * 100, 1)

# ─── Public API ───────────────────────────────────────────────────────────────
def save_candidate_to_vector_db(name: str, resume_text: str, score: float) -> bool:
    """Sanitizes PII, deduplicates, and upserts a candidate into the vector store."""
    db = _get_db()
    clean_text = clean_pii(resume_text)

    # Deduplicate (Chroma supports filter-based delete; skip silently for others)
    if VECTOR_DB_BACKEND == "chroma":
        try:
            existing = db.get(where={"candidate_name": name})
            if existing and existing.get("ids"):
                db.delete(ids=existing["ids"])
                print(f"♻️  Replaced existing entry for: {name}")
        except Exception as e:
            print(f"⚠️ Dedup check failed: {e}")

    doc = Document(
        page_content=clean_text,
        metadata={"candidate_name": name, "overall_score": score}
    )
    db.add_documents([doc])
    return True


def search_similar_candidates(query: str, top_k: int = 3) -> list:
    """Returns candidates semantically similar to a query with 0-100 relevance scores."""
    db = _get_db()

    if VECTOR_DB_BACKEND == "chroma":
        results = db.similarity_search_with_score(query, k=top_k)
        output = []
        for doc, distance in results:
            output.append({
                "name":            doc.metadata.get("candidate_name", "Unknown"),
                "yield_score":     doc.metadata.get("overall_score", 0),
                "relevance_score": _l2_to_relevance(distance),
            })
    else:
        # Pinecone / Qdrant return (doc, score) where score is already a similarity
        results = db.similarity_search_with_score(query, k=top_k)
        output = []
        for doc, sim_score in results:
            output.append({
                "name":            doc.metadata.get("candidate_name", "Unknown"),
                "yield_score":     doc.metadata.get("overall_score", 0),
                "relevance_score": round(float(sim_score) * 100, 1),
            })

    output.sort(key=lambda x: x["relevance_score"], reverse=True)
    return output