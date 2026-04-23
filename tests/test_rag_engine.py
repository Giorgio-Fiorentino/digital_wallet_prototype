"""
tests/test_rag_engine.py

Covers RAGEngine:
  - is_document_question routes correctly (the bug that caused silent RAG failures)
  - _normalize produces unit vectors without relying on faiss.normalize_L2
    (the bug that caused AttributeError: module 'faiss' has no attribute 'normalize_L2')
"""

import numpy as np
import pytest
from models.rag_engine import RAGEngine, _normalize


# ── _normalize helper ─────────────────────────────────────────────────

class TestNormalize:
    def test_unit_vectors_after_normalize(self):
        arr = np.array([[3.0, 4.0], [1.0, 0.0]], dtype="float32")
        _normalize(arr)
        norms = np.linalg.norm(arr, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)

    def test_zero_vector_does_not_raise(self):
        arr = np.array([[0.0, 0.0]], dtype="float32")
        # Should not raise ZeroDivisionError
        _normalize(arr)

    def test_already_normalized_unchanged(self):
        arr = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        _normalize(arr)
        np.testing.assert_allclose(arr, [[1.0, 0.0], [0.0, 1.0]], atol=1e-6)

    def test_modifies_in_place(self):
        arr = np.array([[6.0, 8.0]], dtype="float32")
        original_id = id(arr)
        _normalize(arr)
        assert id(arr) == original_id  # same object
        np.testing.assert_allclose(np.linalg.norm(arr), 1.0, atol=1e-6)


# ── is_document_question routing ──────────────────────────────────────

class TestIsDocumentQuestion:
    """
    These tests guard the routing logic that decides whether a query goes
    to RAG (card terms) or tool-use (transaction data).
    A wrong routing means card-fee questions silently fall through to the
    tool-use loop which has no documents to answer from.
    """

    @pytest.fixture
    def rag(self):
        return RAGEngine()

    # Should route to RAG
    @pytest.mark.parametrize("query", [
        "What is the annual fee?",
        "Does this card have cashback?",
        "What are the foreign transaction fees?",
        "Tell me about the travel insurance benefit",
        "What is the credit limit?",
        "Are there any rewards points?",
        "What is the APR on this card?",
        "Is there a late fee if I miss a payment?",
        "What are the terms and conditions?",
        "Does it cover lounge access?",
    ])
    def test_document_question_returns_true(self, rag, query):
        assert rag.is_document_question(query) is True, \
            f"Expected True for: {query!r}"

    # Should route to tool-use
    @pytest.mark.parametrize("query", [
        "How much did I spend last month?",
        "Show me my top merchants",
        "What are my recent transactions?",
        "Any suspicious charges?",
        "Compare spending across my cards",
        "How much did I spend on food?",
        "What is my total spend this year?",
    ])
    def test_transaction_question_returns_false(self, rag, query):
        assert rag.is_document_question(query) is False, \
            f"Expected False for: {query!r}"
