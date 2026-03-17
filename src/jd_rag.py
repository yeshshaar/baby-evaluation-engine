"""
jd_rag.py — RAG pipeline for Job Description parsing.

Instead of dumping the raw JD into the LLM prompt every time (costly and noisy),
this module:
  1. Chunks the JD into semantic sections on first upload
  2. Embeds and stores the chunks in a dedicated ChromaDB collection
  3. At evaluation time, retrieves only the top-k most relevant chunks
     for a given candidate's skills — reducing prompt size and improving precision.
"""

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ─── Config ───────────────────────────────────────────────────────────────────
DB_DIR = os.environ.get(
    "CHROMA_DB_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_data")
)
JD_COLLECTION = "yield_jd_chunks"

# ─── Lazy singletons ──────────────────────────────────────────────────────────
_embeddings = None
_jd_db      = None

def _get_jd_db() -> Chroma:
    global _embeddings, _jd_db
    if _jd_db is None:
        print("🔧 Initialising JD RAG vector store...")
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _jd_db = Chroma(
            collection_name=JD_COLLECTION,
            embedding_function=_embeddings,
            persist_directory=DB_DIR
        )
    return _jd_db

# ─── Text splitter ────────────────────────────────────────────────────────────
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # ~120 tokens — small enough for precise retrieval
    chunk_overlap=80,     # overlap ensures skills mentioned across sentence
    separators=["\n\n", "\n", ". ", " "]
)

# ─── Public API ───────────────────────────────────────────────────────────────
def index_jd(jd_text: str, jd_id: str) -> int:
    """
    Chunks and embeds a Job Description into the JD vector store.
    Replaces any previously indexed version of the same Jd_id.

    Args:
        jd_text: Raw JD text
        jd_id:   Unique identifier (e.g. session_id or a hash of the JD)

    Returns:
        Number of chunks indexed
    """
    db = _get_jd_db()

    # Remove stale chunks for this JD
    try:
        existing = db.get(where={"jd_id": jd_id})
        if existing and existing.get("ids"):
            db.delete(ids=existing["ids"])
    except Exception:
        pass

    chunks = _splitter.split_text(jd_text)
    docs = [
        Document(page_content=chunk, metadata={"jd_id": jd_id, "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]
    db.add_documents(docs)
    print(f"📄 Indexed {len(docs)} JD chunks for jd_id={jd_id}")
    return len(docs)


def retrieve_relevant_jd_context(query: str, jd_id: str, top_k: int = 4) -> str:
    """
    Retrieves the most relevant JD chunks for a candidate's skill profile.

    Args:
        query:  The candidate's skills/resume summary used as the search query
        jd_id:  Which JD to search within
        top_k:  Number of chunks to retrieve

    Returns:
        A single string of the top-k chunks joined for use in an LLM prompt
    """
    db = _get_jd_db()

    try:
        results = db.similarity_search(
            query,
            k=top_k,
            filter={"jd_id": jd_id}
        )
        if not results:
            return ""
        return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        print(f"⚠️ JD RAG retrieval failed: {e}")
        return ""


def get_full_jd(jd_id: str) -> str:
    """
    Reconstructs the full JD from stored chunks (ordered by chunk_index).
    Useful as a fallback when RAG retrieval returns too little context.
    """
    db = _get_jd_db()
    try:
        results = db.get(where={"jd_id": jd_id}, include=["documents", "metadatas"])
        if not results or not results.get("documents"):
            return ""
        paired = sorted(
            zip(results["metadatas"], results["documents"]),
            key=lambda x: x[0].get("chunk_index", 0)
        )
        return "\n".join([doc for _, doc in paired])
    except Exception as e:
        print(f"⚠️ JD reconstruction failed: {e}")
        return ""

