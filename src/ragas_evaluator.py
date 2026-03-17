"""
ragas_evaluator.py — Objective evaluation of Yield-AI's scoring pipeline using RAGAS.

RAGAS measures the *quality of the AI system itself*, not the candidate.
It answers: "How reliable is Yield-AI's evaluation?"

Metrics used:
  - faithfulness:        Are the matched/missing skills actually in the resume?
  - answer_relevancy:   Does the evaluation address what the JD asked for?
  - context_precision:  Did the retrieved JD chunks contain the right info?
  - context_recall:     Did we retrieve all the relevant JD context?
"""

import os
from typing import List
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


def _get_ragas_llm():
    """RAGAS needs an LLM to judge faithfulness and relevancy."""
    return ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.0
    )

def _get_ragas_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def evaluate_pipeline(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str]
) -> dict:
    """
    Runs RAGAS evaluation over a batch of resume evaluations.

    Args:
        questions:     The evaluation query per candidate
                       e.g. "Does this candidate fit the ML Engineer JD?"
        answers:       Yield-AI's generated evaluation text per candidate
        contexts:      Retrieved JD chunks used to generate each answer
                       (list of lists — each candidate has multiple chunks)
        ground_truths: The ideal answer / expected key skills per candidate

    Returns:
        Dict of metric_name → score (0–1 scale, higher is better)
    """
    dataset = Dataset.from_dict({
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts,
        "ground_truth": ground_truths,
    })

    llm        = _get_ragas_llm()
    embeddings = _get_ragas_embeddings()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,     # gracefully skip broken rows
    )

    scores = result.to_pandas().mean(numeric_only=True).to_dict()

    # Round for readability
    return {k: round(float(v), 4) for k, v in scores.items() if not k.startswith("_")}


def build_ragas_sample(
    jd_text: str,
    resume_text: str,
    evaluation_result: dict,
    retrieved_jd_chunks: List[str]
) -> dict:
    """
    Converts a single Yield-AI evaluation into a RAGAS-compatible sample.
    Call this per candidate, collect all samples, then pass to evaluate_pipeline().

    Args:
        jd_text:              Full job description
        resume_text:          Candidate resume text
        evaluation_result:    Output from run_evaluation_chain()
        retrieved_jd_chunks:  JD chunks retrieved by jd_rag.retrieve_relevant_jd_context()

    Returns:
        Single RAGAS sample dict
    """
    matched = ", ".join(evaluation_result.get("matched_skills", []))
    missing = ", ".join(evaluation_result.get("missing_skills", []))
    score   = evaluation_result.get("overall_score", 0)

    question = "Does this candidate's resume match the requirements of the job description?"

    answer = (
        f"Yield-AI Score: {score}%. "
        f"Matched skills: {matched or 'none'}. "
        f"Missing skills: {missing or 'none'}."
    )

    # Ground truth: what the JD ideally expects (first 500 chars of JD as proxy)
    ground_truth = jd_text[:500]

    return {
        "question":    question,
        "answer":      answer,
        "contexts":    retrieved_jd_chunks if retrieved_jd_chunks else [jd_text[:500]],
        "ground_truth": ground_truth,
    }
