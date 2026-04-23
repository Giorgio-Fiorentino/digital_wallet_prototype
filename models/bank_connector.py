"""
models/bank_connector.py

TrueLayer sandbox adapter. Fetches live transactions via OAuth2
authorization-code flow and normalises to the internal schema.

Required env vars (never hardcode):
    TRUELAYER_CLIENT_ID
    TRUELAYER_CLIENT_SECRET
    TRUELAYER_REFRESH_TOKEN   ← obtained once via the consent flow; lasts ~30 days
    TRUELAYER_ACCESS_TOKEN    ← optional; adapter auto-refreshes using refresh token
"""

import os
import pandas as pd
import requests
from dotenv import load_dotenv
from models.categorizer import TFIDFCategorizer

load_dotenv()

_AUTH_URL = "https://auth.truelayer-sandbox.com/connect/token"
_DATA_URL = "https://api.truelayer-sandbox.com/data/v1"

# Standard financial taxonomy used for all LLM-based classification.
# Consistent across TrueLayer and CSV data sources.
FINANCIAL_CATEGORIES = [
    "Food & Dining", "Groceries", "Shopping", "Transport",
    "Entertainment", "Health & Fitness", "Travel", "Utilities",
    "Finance & Investment", "Transfer", "Misc",
]


class TrueLayerAdapter:
    """
    Two-token strategy:
      1. Try TRUELAYER_ACCESS_TOKEN if present.
      2. On 401 (expired), or if no access token, call _refresh() using
         TRUELAYER_REFRESH_TOKEN to get a fresh access token.
      3. Retry the failed request once with the new token.

    The refresh token itself lasts ~30 days in the TrueLayer sandbox.
    After 30 days, repeat the one-time OAuth consent flow to get a new one.
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        access_token: str | None = None,
        refresh_token: str | None = None,
    ):
        self._client_id     = client_id     or os.getenv("TRUELAYER_CLIENT_ID", "")
        self._client_secret = client_secret or os.getenv("TRUELAYER_CLIENT_SECRET", "")
        self._token: str | None = (
            access_token or os.getenv("TRUELAYER_ACCESS_TOKEN", "") or None
        )
        self._refresh_token: str | None = (
            refresh_token or os.getenv("TRUELAYER_REFRESH_TOKEN", "") or None
        )

    # ── Private helpers ──────────────────────────────────────────────

    def _refresh(self) -> None:
        """
        Exchange the refresh token for a fresh access token.
        Updates self._token in-memory (valid for this session).
        Raises RuntimeError if no refresh token is available.
        """
        if not self._refresh_token:
            raise RuntimeError(
                "No TrueLayer tokens found. Set TRUELAYER_REFRESH_TOKEN (and optionally "
                "TRUELAYER_ACCESS_TOKEN) in .env. Obtain them once via the OAuth2 consent "
                "flow: visit the TrueLayer sandbox auth URL with your client_id, complete "
                "the mock bank login, then exchange the code via curl for access_token + "
                "refresh_token. The refresh token lasts ~30 days."
            )
        resp = requests.post(
            _AUTH_URL,
            data={
                "grant_type":    "refresh_token",
                "client_id":     self._client_id,
                "client_secret": self._client_secret,
                "refresh_token": self._refresh_token,
            },
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
        self._token = payload["access_token"]
        # TrueLayer may rotate the refresh token; keep it current in memory
        if "refresh_token" in payload:
            self._refresh_token = payload["refresh_token"]

    def _get(self, path: str) -> dict:
        """Authenticated GET against the TrueLayer Data API. Auto-refreshes on 401."""
        # If we have no access token at all, refresh first
        if self._token is None:
            self._refresh()

        resp = requests.get(
            f"{_DATA_URL}/{path}",
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=10,
        )

        # 401 = access token expired → refresh and retry once
        if resp.status_code == 401:
            self._refresh()
            resp = requests.get(
                f"{_DATA_URL}/{path}",
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=10,
            )

        resp.raise_for_status()
        return resp.json()

    def _categorize_descriptions(self, descriptions: list) -> list:
        """
        Classify transaction descriptions using TF-IDF cosine similarity.
        Instant, no API call — sufficient for sandbox mock data.
        """
        if not descriptions:
            return []
        try:
            cat = TFIDFCategorizer()
            return [cat.predict(d)[0] for d in descriptions]
        except Exception:
            return ["Misc"] * len(descriptions)

    # ── Public API ───────────────────────────────────────────────────

    def fetch_transactions(self) -> pd.DataFrame:
        """
        Fetch all sandbox transactions across all accounts.
        Returns a DataFrame with the same schema as transactions.csv:
        Date, Amount, Raw_Description, Category, Card, Is_Fraud,
        Month, Month_Num, Year, DayOfWeek.
        Raises RuntimeError on any network or auth failure.
        """
        try:
            accounts = self._get("accounts")["results"]
            rows = []
            for account in accounts:
                account_id   = account["account_id"]
                display_name = account.get("display_name", "TrueLayer")
                txns = self._get(f"accounts/{account_id}/transactions")["results"]
                for t in txns:
                    rows.append({
                        "Date":            pd.to_datetime(t.get("timestamp", "")[:10]),
                        "Amount":          abs(float(t.get("amount", 0))),
                        "Raw_Description": t.get("description", "Unknown"),
                        "Card":            display_name,
                        "Is_Fraud":        False,
                    })

            if not rows:
                return pd.DataFrame(
                    columns=["Date", "Amount", "Raw_Description",
                             "Category", "Card", "Is_Fraud",
                             "Month", "Month_Num", "Year", "DayOfWeek"]
                )

            df = pd.DataFrame(rows)

            # Deduplicate — TrueLayer mock returns identical transactions for
            # every account; keep the first occurrence per (date, amount, description)
            df = df.drop_duplicates(subset=["Date", "Amount", "Raw_Description"])
            df = df.reset_index(drop=True)

            # Categorise all descriptions via LLM batch call (robust for any
            # transaction set — no dependence on TrueLayer's own classification)
            df["Category"] = self._categorize_descriptions(
                df["Raw_Description"].tolist()
            )

            df["Month"]     = df["Date"].dt.month_name()
            df["Month_Num"] = df["Date"].dt.month
            df["Year"]      = df["Date"].dt.year
            df["DayOfWeek"] = df["Date"].dt.day_name()
            return df

        except Exception as exc:
            raise RuntimeError(f"TrueLayer fetch failed: {exc}") from exc
