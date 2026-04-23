"""
tests/test_categorizer.py

Covers TFIDFCategorizer:
  - predict returns correct tuple type
  - confidence is in [0, 1]
  - category is a non-empty string
  - known merchants hit the knowledge base
  - train() makes predict() return the trained category on next call
    (the bug that made AI Lab retraining a no-op)
"""

import pytest
from models.categorizer import TFIDFCategorizer, KNOWLEDGE_BASE


@pytest.fixture
def cat():
    return TFIDFCategorizer()


class TestTFIDFCategorizerPredict:

    def test_returns_three_tuple(self, cat):
        result = cat.predict("starbucks coffee")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_category_is_nonempty_string(self, cat):
        category, _, _ = cat.predict("netflix subscription")
        assert isinstance(category, str)
        assert len(category) > 0

    def test_confidence_in_range(self, cat):
        _, confidence, _ = cat.predict("amazon shopping")
        assert 0.0 <= confidence <= 1.0

    def test_needs_review_is_bool(self, cat):
        _, _, needs_review = cat.predict("random unknown merchant xyz")
        assert isinstance(needs_review, bool)

    @pytest.mark.parametrize("description,expected_category", [
        ("starbucks coffee",   "Food & Dining"),
        ("amazon",             "Shopping"),
        ("netflix",            "Entertainment"),
        ("uber",               "Gas & Transport"),
        ("gym membership",     "Health & Fitness"),
        ("pharmacy",           "Health & Fitness"),
    ])
    def test_known_merchants_hit_knowledge_base(self, cat, description, expected_category):
        category, confidence, _ = cat.predict(description)
        assert category == expected_category, \
            f"'{description}' → got '{category}', expected '{expected_category}'"
        assert confidence > 0.20  # above TFIDF_THRESHOLD

    def test_high_confidence_not_flagged(self, cat):
        _, _, needs_review = cat.predict("starbucks coffee")
        assert needs_review is False

    def test_unknown_merchant_flagged_for_review(self, cat):
        # A completely unrecognisable description should have low confidence
        _, _, needs_review = cat.predict("xyzabc12345qwerty")
        assert needs_review is True


class TestTFIDFCategorizerTrain:
    """
    These tests caught the bug where retraining in AI Lab appeared to succeed
    but had no effect because a fresh categorizer was created each scan.
    """

    def test_train_affects_subsequent_predict(self):
        cat = TFIDFCategorizer()
        cat.train("mynewmerchant2024", "Travel")
        category, _, _ = cat.predict("mynewmerchant2024")
        assert category == "Travel"

    def test_train_updates_knowledge_base(self):
        cat = TFIDFCategorizer()
        cat.train("exclusivedesc99", "Utilities")
        assert "exclusivedesc99" in cat.knowledge_base
        assert cat.knowledge_base["exclusivedesc99"] == "Utilities"

    def test_train_does_not_affect_other_instance(self):
        """Each TFIDFCategorizer instance has its own knowledge base."""
        cat_a = TFIDFCategorizer()
        cat_b = TFIDFCategorizer()
        cat_a.train("uniquemerchant999", "Travel")
        assert "uniquemerchant999" not in cat_b.knowledge_base

    def test_multiple_trains_all_take_effect(self):
        cat = TFIDFCategorizer()
        cat.train("merchantalpha", "Travel")
        cat.train("merchantbeta",  "Utilities")
        assert cat.knowledge_base["merchantalpha"] == "Travel"
        assert cat.knowledge_base["merchantbeta"]  == "Utilities"

    def test_train_overwrites_existing_entry(self):
        cat = TFIDFCategorizer()
        cat.train("amazon", "Travel")  # override the built-in "Shopping" mapping
        assert cat.knowledge_base["amazon"] == "Travel"
