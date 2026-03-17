"""
tests/test_pipeline.py — Unit tests for the Yield-AI core pipeline.
Runs without a real GROQ_API_KEY by mocking the LLM call.
"""

import pytest
from unittest.mock import patch
from src.sanitizer import clean_pii
from src.chains import _zero_result


# ─── Sanitizer tests ──────────────────────────────────────────────────────────
class TestSanitizer:
    def test_removes_email(self):
        text = "Contact me at john.doe@gmail.com for more info."
        result = clean_pii(text)
        assert "@" not in result
        assert "[REDACTED EMAIL]" in result

    def test_removes_phone(self):
        text = "Call me at 415-555-1234 anytime."
        result = clean_pii(text)
        assert "415-555-1234" not in result
        assert "[REDACTED PHONE]" in result

    def test_removes_url(self):
        text = "Portfolio: https://johndoe.dev"
        result = clean_pii(text)
        assert "https://johndoe.dev" not in result

    def test_clean_text_unchanged(self):
        text = "Experienced Python developer with 3 years in ML."
        assert clean_pii(text) == text


# ─── Chain fallback tests ─────────────────────────────────────────────────────
class TestChains:
    def test_zero_result_structure(self):
        result = _zero_result("Test error")
        assert result["overall_score"] == 0
        assert "error" in result
        assert result["matched_skills"] == []
        assert result["missing_skills"] == []

    def test_zero_result_breakdown_keys(self):
        result = _zero_result()
        assert "Skill Match" in result["breakdown"]
        assert "Semantic Match" in result["breakdown"]
        assert "Experience Relevance" in result["breakdown"]

    @patch("src.chains._invoke_chain")
    def test_run_evaluation_chain_success(self, mock_invoke):
        from src.chains import run_evaluation_chain, EvaluationResult
        mock_result = EvaluationResult(
            skill_match_score=80,
            semantic_match_score=60,
            experience_relevance_score=40,
            matched_skills=["Python", "PyTorch"],
            missing_skills=["Kubernetes"]
        )
        mock_invoke.return_value = mock_result
        result = run_evaluation_chain("resume text", "jd text")
        assert result["overall_score"] == 63.0
        assert "Python" in result["matched_skills"]
        assert "Kubernetes" in result["missing_skills"]


# ─── Vector store helper tests (no heavy deps needed) ────────────────────────
def _l2_to_relevance(distance: float) -> float:
    relevance = max(0.0, 1.0 - (distance / 2.0))
    return round(relevance * 100, 1)


class TestVectorStore:
    def test_l2_to_relevance_identical(self):
        assert _l2_to_relevance(0.0) == 100.0

    def test_l2_to_relevance_max_distance(self):
        assert _l2_to_relevance(2.0) == 0.0

    def test_l2_to_relevance_midpoint(self):
        assert _l2_to_relevance(1.0) == 50.0

    def test_l2_to_relevance_no_negative(self):
        assert _l2_to_relevance(5.0) == 0.0
