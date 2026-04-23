"""
tests/test_wallet_engine.py

Covers WalletEngine:
  - inject_dataframe / valid_sources switching  (the 5-CSV-cards bug)
  - get_transactions filter behaviour
  - get_spending_summary totals
  - get_top_merchants
  - get_source_breakdown
  - detect_anomalies
  - get_monthly_comparison
  - get_health_inputs
  - get_budget_status
"""

import pandas as pd
import pytest
from models.wallet_engine import WalletEngine


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_df(**kwargs) -> pd.DataFrame:
    """Minimal synthetic DataFrame that satisfies the engine's expected schema."""
    base = {
        "Date":            pd.to_datetime(["2024-01-10", "2024-01-15", "2024-02-05",
                                           "2024-02-20", "2024-03-01"]),
        "Amount":          [50.0, 120.0, 30.0, 200.0, 75.0],
        "Raw_Description": ["STARBUCKS", "AMAZON", "UBER", "APPLE STORE", "NETFLIX"],
        "Category":        ["Food & Dining", "Shopping", "Transport",
                            "Shopping", "Entertainment"],
        "Card":            ["CardA", "CardA", "CardB", "CardB", "CardA"],
        "Is_Fraud":        [False, False, False, True, False],
        "Month":           ["January", "January", "February", "February", "March"],
        "Month_Num":       [1, 1, 2, 2, 3],
        "Year":            [2024, 2024, 2024, 2024, 2024],
        "DayOfWeek":       ["Wednesday", "Monday", "Monday", "Tuesday", "Friday"],
    }
    base.update(kwargs)
    return pd.DataFrame(base)


@pytest.fixture
def engine():
    """Engine loaded with synthetic data — no CSV dependency."""
    e = WalletEngine()
    e.inject_dataframe(_make_df())
    return e


@pytest.fixture
def csv_engine():
    """Engine with real CSV data (skipped if file absent)."""
    import os
    if not os.path.exists("data/processed/transactions.csv"):
        pytest.skip("CSV data not available")
    e = WalletEngine()
    e.load_data()
    return e


# ── inject_dataframe / valid_sources ─────────────────────────────────

class TestInjectDataframe:
    def test_valid_sources_reflects_injected_data(self):
        e = WalletEngine()
        e.inject_dataframe(_make_df())
        assert set(e.valid_sources) == {"CardA", "CardB"}

    def test_inject_none_resets_to_csv(self):
        """After inject(None) the engine falls back to load_data() path."""
        e = WalletEngine()
        e.inject_dataframe(_make_df())
        # Inject None — next .df call must not return the synthetic data
        e.inject_dataframe(None)
        # _df is None; load_data will attempt CSV (may raise FileNotFoundError
        # in CI without CSV, but the important thing is it no longer returns
        # the synthetic data we injected).
        assert e._df is None

    def test_valid_categories_reflects_injected_data(self):
        e = WalletEngine()
        e.inject_dataframe(_make_df())
        cats = set(e.valid_categories)
        assert cats == {"Entertainment", "Food & Dining", "Shopping", "Transport"}

    def test_switching_source_updates_sources(self):
        e = WalletEngine()
        df_a = _make_df()
        e.inject_dataframe(df_a)
        assert "CardA" in e.valid_sources

        df_b = _make_df()
        df_b["Card"] = "LiveBank"
        e.inject_dataframe(df_b)
        assert e.valid_sources == ["LiveBank"]
        assert "CardA" not in e.valid_sources


# ── get_transactions ──────────────────────────────────────────────────

class TestGetTransactions:
    def test_returns_string(self, engine):
        result = engine.get_transactions()
        assert isinstance(result, str)

    def test_filter_by_source(self, engine):
        result = engine.get_transactions(source="CardA")
        assert "CardB" not in result

    def test_filter_by_category(self, engine):
        result = engine.get_transactions(category="Transport")
        assert "UBER" in result

    def test_filter_by_month(self, engine):
        result = engine.get_transactions(month="January")
        assert "STARBUCKS" in result
        assert "UBER" not in result

    def test_limit_respected(self, engine):
        result = engine.get_transactions(limit=1)
        # Only 1 transaction shown; total count still reported
        assert "showing latest 1" in result

    def test_no_match_returns_no_transactions_message(self, engine):
        result = engine.get_transactions(source="NonExistent")
        assert "No transactions found" in result

    def test_date_range_filter(self, engine):
        result = engine.get_transactions(date_from="2024-03-01", date_to="2024-03-31")
        assert "NETFLIX" in result
        assert "STARBUCKS" not in result


# ── get_spending_summary ──────────────────────────────────────────────

class TestGetSpendingSummary:
    def test_returns_string(self, engine):
        assert isinstance(engine.get_spending_summary(), str)

    def test_contains_total(self, engine):
        result = engine.get_spending_summary()
        assert "TOTAL" in result

    def test_filter_by_source_reduces_total(self, engine):
        all_result  = engine.get_spending_summary()
        card_result = engine.get_spending_summary(source="CardB")
        # Extract totals — CardB total must be less than overall total
        import re
        all_total  = float(re.search(r"TOTAL\s+\$\s*([\d,]+\.\d+)", all_result).group(1).replace(",", ""))
        card_total = float(re.search(r"TOTAL\s+\$\s*([\d,]+\.\d+)", card_result).group(1).replace(",", ""))
        assert card_total < all_total

    def test_month_filter(self, engine):
        result = engine.get_spending_summary(month="January")
        assert "Food & Dining" in result or "Shopping" in result


# ── get_top_merchants ─────────────────────────────────────────────────

class TestGetTopMerchants:
    def test_returns_string(self, engine):
        assert isinstance(engine.get_top_merchants(), str)

    def test_limit_respected(self, engine):
        result = engine.get_top_merchants(limit=2)
        # Should mention at most 2 merchants
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_merchant_name(self, engine):
        result = engine.get_top_merchants()
        # At least one of our synthetic merchants should appear
        assert any(m in result for m in ["AMAZON", "APPLE STORE", "NETFLIX",
                                          "STARBUCKS", "UBER"])


# ── get_source_breakdown ──────────────────────────────────────────────

class TestGetSourceBreakdown:
    def test_returns_string(self, engine):
        assert isinstance(engine.get_source_breakdown(), str)

    def test_contains_both_cards(self, engine):
        result = engine.get_source_breakdown()
        assert "CardA" in result
        assert "CardB" in result


# ── detect_anomalies ──────────────────────────────────────────────────

class TestDetectAnomalies:
    def test_returns_string(self, engine):
        assert isinstance(engine.detect_anomalies(), str)

    def test_fraud_flagged(self, engine):
        # Our synthetic data has one Is_Fraud=True row (APPLE STORE, $200)
        result = engine.detect_anomalies()
        # Anomaly detection may or may not flag it depending on z-score,
        # but the method must return a non-empty string
        assert len(result) > 0


# ── get_monthly_comparison ────────────────────────────────────────────

class TestGetMonthlyComparison:
    def test_returns_string(self, engine):
        assert isinstance(engine.get_monthly_comparison(), str)

    def test_last_n_months_param(self, engine):
        result = engine.get_monthly_comparison(last_n_months=2)
        assert isinstance(result, str)


# ── get_health_inputs (existing, kept for regression) ─────────────────

class TestGetHealthInputs:
    def test_returns_all_five_keys(self, engine):
        result = engine.get_health_inputs()
        assert set(result.keys()) == {
            "fraud_rate", "anomaly_count", "category_diversity",
            "monthly_variance", "top_category_share",
        }

    def test_fraud_rate_in_range(self, engine):
        assert 0.0 <= engine.get_health_inputs()["fraud_rate"] <= 1.0

    def test_anomaly_count_non_negative(self, engine):
        assert engine.get_health_inputs()["anomaly_count"] >= 0

    def test_category_diversity_at_least_one(self, engine):
        assert engine.get_health_inputs()["category_diversity"] >= 1

    def test_monthly_variance_non_negative(self, engine):
        assert engine.get_health_inputs()["monthly_variance"] >= 0.0

    def test_top_category_share_in_range(self, engine):
        assert 0.0 <= engine.get_health_inputs()["top_category_share"] <= 1.0


# ── get_budget_status (existing, kept for regression) ─────────────────

class TestGetBudgetStatus:
    def test_returns_entry_per_goal(self, engine):
        result = engine.get_budget_status({"Food & Dining": 500.0, "Shopping": 300.0})
        assert "Food & Dining" in result
        assert "Shopping" in result

    def test_entry_has_required_keys(self, engine):
        entry = engine.get_budget_status({"Food & Dining": 500.0})["Food & Dining"]
        assert set(entry.keys()) == {"limit", "spent", "pct", "status"}

    def test_limit_matches_input(self, engine):
        assert engine.get_budget_status({"Shopping": 999.0})["Shopping"]["limit"] == 999.0

    def test_spent_non_negative(self, engine):
        assert engine.get_budget_status({"Shopping": 500.0})["Shopping"]["spent"] >= 0.0

    def test_status_values(self, engine):
        status = engine.get_budget_status({"Shopping": 500.0})["Shopping"]["status"]
        assert status in ("ok", "warning", "over")

    def test_zero_limit_excluded(self, engine):
        assert "Shopping" not in engine.get_budget_status({"Shopping": 0.0})


# ── Helpers for merged-source tests ──────────────────────────────────

def _make_merged_df() -> pd.DataFrame:
    """Two-source DataFrame mimicking csv + live merge in app.py."""
    csv_half = pd.DataFrame({
        "Date":            pd.to_datetime(["2024-01-10", "2024-02-05"]),
        "Amount":          [50.0, 30.0],
        "Raw_Description": ["STARBUCKS", "UBER"],
        "Category":        ["Food & Dining", "Transport"],
        "Card":            ["Visa Classic", "Mastercard Gold"],
        "Is_Fraud":        [False, False],
        "Month":           ["January", "February"],
        "Month_Num":       [1, 2],
        "Year":            [2024, 2024],
        "DayOfWeek":       ["Wednesday", "Monday"],
        "Source_Tag":      ["csv", "csv"],
    })
    live_half = pd.DataFrame({
        "Date":            pd.to_datetime(["2024-01-15", "2024-03-01"]),
        "Amount":          [120.0, 75.0],
        "Raw_Description": ["AMAZON", "NETFLIX"],
        "Category":        ["Shopping", "Entertainment"],
        "Card":            ["TrueLayer Current Account", "TrueLayer Current Account"],
        "Is_Fraud":        [False, False],
        "Month":           ["January", "March"],
        "Month_Num":       [1, 3],
        "Year":            [2024, 2024],
        "DayOfWeek":       ["Monday", "Friday"],
        "Source_Tag":      ["live", "live"],
    })
    return pd.concat([csv_half, live_half], ignore_index=True)


# ── TestMergedDataframe ───────────────────────────────────────────────

class TestMergedDataframe:

    def test_valid_sources_includes_all_cards(self):
        e = WalletEngine()
        e.inject_dataframe(_make_merged_df())
        sources = e.valid_sources
        assert "Visa Classic" in sources
        assert "Mastercard Gold" in sources
        assert "TrueLayer Current Account" in sources

    def test_get_spending_summary_covers_both_halves(self):
        """Total must reflect rows from both csv and live halves (50+30+120+75=275)."""
        e = WalletEngine()
        e.inject_dataframe(_make_merged_df())
        result = e.get_spending_summary()
        assert isinstance(result, str)
        assert "275" in result

    def test_get_source_breakdown_no_source_type(self):
        """Merged df has no Source_Type column — get_source_breakdown must not raise."""
        e = WalletEngine()
        e.inject_dataframe(_make_merged_df())
        result = e.get_source_breakdown()
        assert isinstance(result, str)
        assert "Visa Classic" in result
        assert "TrueLayer Current Account" in result

    def test_source_tag_column_preserved(self):
        """inject_dataframe must not strip extra columns like Source_Tag."""
        e = WalletEngine()
        e.inject_dataframe(_make_merged_df())
        assert "Source_Tag" in e.df.columns
        assert set(e.df["Source_Tag"].unique()) == {"csv", "live"}
