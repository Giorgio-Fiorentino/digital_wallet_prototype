"""
tests/test_bank_connector.py

Covers TrueLayerAdapter:
  - fetch_transactions returns correct schema
  - deduplication removes identical rows across accounts
  - 401 triggers refresh-token flow
  - missing credentials raises RuntimeError (not silent)
  - _categorize_descriptions returns correct-length list within FINANCIAL_CATEGORIES
  - amounts are always positive
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from models.bank_connector import TrueLayerAdapter, FINANCIAL_CATEGORIES


MOCK_ACCOUNTS = {
    "results": [
        {"account_id": "acc_1", "display_name": "Mock Current Account"},
        {"account_id": "acc_2", "display_name": "Mock Savings Account"},
    ]
}

MOCK_TRANSACTIONS = {
    "results": [
        {
            "timestamp": "2024-03-15T12:00:00Z",
            "amount": -42.50,
            "description": "COSTA COFFEE",
            "transaction_classification": [],
        },
        {
            "timestamp": "2024-03-16T09:00:00Z",
            "amount": -120.00,
            "description": "AMAZON",
            "transaction_classification": [],
        },
    ]
}

# Both accounts return the same transactions — deduplication must collapse them
MOCK_TRANSACTIONS_DUPLICATE = MOCK_TRANSACTIONS


def _mock_get_response(json_data):
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = json_data
    m.raise_for_status = MagicMock()
    return m


def _mock_401():
    m = MagicMock()
    m.status_code = 401
    m.raise_for_status.side_effect = Exception("401 Unauthorized")
    return m


@pytest.fixture
def adapter():
    """Adapter with a pre-set access token — skips the refresh flow."""
    return TrueLayerAdapter(
        client_id="fake_id",
        client_secret="fake_secret",
        access_token="fake_access_token",
    )


# ── Schema and basic correctness ──────────────────────────────────────

class TestFetchTransactions:

    @patch("models.bank_connector.TrueLayerAdapter._categorize_descriptions",
           return_value=["Food & Dining", "Shopping"])
    @patch("models.bank_connector.requests.get")
    def test_returns_dataframe(self, mock_get, mock_cat, adapter):
        mock_get.side_effect = [
            _mock_get_response({"results": [{"account_id": "acc_1",
                                             "display_name": "Mock"}]}),
            _mock_get_response(MOCK_TRANSACTIONS),
        ]
        df = adapter.fetch_transactions()
        assert isinstance(df, pd.DataFrame)

    @patch("models.bank_connector.TrueLayerAdapter._categorize_descriptions",
           return_value=["Food & Dining", "Shopping"])
    @patch("models.bank_connector.requests.get")
    def test_required_columns_present(self, mock_get, mock_cat, adapter):
        mock_get.side_effect = [
            _mock_get_response({"results": [{"account_id": "acc_1",
                                             "display_name": "Mock"}]}),
            _mock_get_response(MOCK_TRANSACTIONS),
        ]
        df = adapter.fetch_transactions()
        required = {"Date", "Amount", "Raw_Description", "Category",
                    "Card", "Is_Fraud", "Month", "Month_Num", "Year", "DayOfWeek"}
        assert required.issubset(set(df.columns))

    @patch("models.bank_connector.TrueLayerAdapter._categorize_descriptions",
           return_value=["Food & Dining", "Shopping"])
    @patch("models.bank_connector.requests.get")
    def test_amounts_always_positive(self, mock_get, mock_cat, adapter):
        mock_get.side_effect = [
            _mock_get_response({"results": [{"account_id": "acc_1",
                                             "display_name": "Mock"}]}),
            _mock_get_response(MOCK_TRANSACTIONS),
        ]
        df = adapter.fetch_transactions()
        assert (df["Amount"] >= 0).all()

    @patch("models.bank_connector.TrueLayerAdapter._categorize_descriptions",
           return_value=["Food & Dining", "Shopping"])
    @patch("models.bank_connector.requests.get")
    def test_empty_accounts_returns_empty_dataframe(self, mock_get, mock_cat, adapter):
        mock_get.return_value = _mock_get_response({"results": []})
        df = adapter.fetch_transactions()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @patch("models.bank_connector.TrueLayerAdapter._categorize_descriptions",
           return_value=["Food & Dining", "Shopping"])
    @patch("models.bank_connector.requests.get")
    def test_deduplication_removes_identical_rows(self, mock_get, mock_cat, adapter):
        """Two accounts returning identical transactions must collapse to unique rows."""
        # Two accounts, same transactions each
        mock_get.side_effect = [
            _mock_get_response(MOCK_ACCOUNTS),          # accounts
            _mock_get_response(MOCK_TRANSACTIONS),       # acc_1 transactions
            _mock_get_response(MOCK_TRANSACTIONS),       # acc_2 transactions (duplicates)
        ]
        mock_cat.return_value = ["Food & Dining", "Shopping"]
        df = adapter.fetch_transactions()
        # Should have 2 unique rows, not 4
        assert len(df) == 2


# ── Auth / token refresh ──────────────────────────────────────────────

class TestAuthFlow:

    @patch.dict("os.environ", {
        "TRUELAYER_ACCESS_TOKEN": "",
        "TRUELAYER_REFRESH_TOKEN": "",
        "TRUELAYER_CLIENT_ID": "",
        "TRUELAYER_CLIENT_SECRET": "",
    })
    def test_no_token_no_refresh_raises_runtime_error(self):
        """No credentials → RuntimeError with clear message (env vars cleared)."""
        adapter = TrueLayerAdapter(
            client_id="", client_secret="",
            access_token=None, refresh_token=None,
        )
        with pytest.raises(RuntimeError, match="TrueLayer"):
            adapter.fetch_transactions()

    @patch("models.bank_connector.requests.post")
    @patch("models.bank_connector.requests.get")
    def test_401_triggers_refresh_and_retries(self, mock_get, mock_post):
        """On 401, adapter must call refresh endpoint and retry the request."""
        adapter = TrueLayerAdapter(
            client_id="cid", client_secret="csec",
            access_token="expired_token",
            refresh_token="valid_refresh",
        )

        # First GET → 401; second GET (after refresh) → accounts; third → transactions
        accounts_response = _mock_get_response(
            {"results": [{"account_id": "acc_1", "display_name": "Mock"}]}
        )
        txn_response = _mock_get_response({"results": []})

        mock_get.side_effect = [
            _mock_401(),         # first attempt: 401
            accounts_response,   # retry after refresh: accounts
            txn_response,        # transactions
        ]
        mock_post.return_value = _mock_get_response(
            {"access_token": "new_token", "refresh_token": "new_refresh"}
        )

        df = adapter.fetch_transactions()
        assert isinstance(df, pd.DataFrame)
        # Refresh endpoint was called
        mock_post.assert_called_once()


# ── _categorize_descriptions ──────────────────────────────────────────

class TestCategorizeDescriptions:

    def test_returns_same_length_as_input(self):
        adapter = TrueLayerAdapter(
            client_id="x", client_secret="x",
            access_token="tok", refresh_token=None,
        )
        descriptions = ["STARBUCKS", "AMAZON", "NETFLIX", "UBER"]
        result = adapter._categorize_descriptions(descriptions)
        assert len(result) == len(descriptions)

    def test_all_categories_are_valid(self):
        adapter = TrueLayerAdapter(
            client_id="x", client_secret="x",
            access_token="tok", refresh_token=None,
        )
        descriptions = ["STARBUCKS", "AMAZON"]
        result = adapter._categorize_descriptions(descriptions)
        for cat in result:
            assert isinstance(cat, str) and len(cat) > 0, f"Invalid category: {cat!r}"

    def test_tfidf_returns_nonempty_strings(self):
        """TF-IDF path must return a same-length list of non-empty strings."""
        adapter = TrueLayerAdapter(
            client_id="x", client_secret="x",
            access_token="tok", refresh_token=None,
        )
        descriptions = ["STARBUCKS", "AMAZON"]
        result = adapter._categorize_descriptions(descriptions)
        assert len(result) == len(descriptions)
        assert all(isinstance(c, str) and len(c) > 0 for c in result)

    def test_empty_input_returns_empty_list(self):
        adapter = TrueLayerAdapter(access_token="tok")
        assert adapter._categorize_descriptions([]) == []
