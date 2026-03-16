"""
models/wallet_engine.py

Core computation layer. Every method = one tool the LLM can call.
The LLM never touches pandas directly.
Categories and sources derived from data — no hardcoding.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import os
from data.loader import SOURCES, SOURCE_TYPES


class WalletEngine:

    def __init__(self, data_path: str = "data/processed/transactions.csv"):
        self.data_path = data_path
        self._df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        if self._df is None:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(
                    f"Data not found at {self.data_path}. "
                    "Run: python3 data/loader.py"
                )
            df = pd.read_csv(self.data_path)
            df["Date"]      = pd.to_datetime(df["Date"])
            df["Month"]     = df["Date"].dt.month_name()
            df["Month_Num"] = df["Date"].dt.month
            df["Year"]      = df["Date"].dt.year
            df["DayOfWeek"] = df["Date"].dt.day_name()
            df["Amount"]    = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
            self._df = df
        return self._df

    @property
    def df(self) -> pd.DataFrame:
        return self.load_data()

    @property
    def valid_categories(self) -> list:
        return sorted(self.df["Category"].unique().tolist())

    @property
    def valid_sources(self) -> list:
        return sorted(self.df["Card"].unique().tolist())

    def get_transactions(
        self,
        source: Optional[str] = None,
        category: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        month: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        df = self.df.copy()
        if source and source != "All":
            df = df[df["Card"] == source]
        if category and category != "All":
            df = df[df["Category"] == category]
        if month:
            df = df[df["Month"].str.lower() == month.lower()]
        if date_from:
            df = df[df["Date"] >= pd.to_datetime(date_from)]
        if date_to:
            df = df[df["Date"] <= pd.to_datetime(date_to)]
        if df.empty:
            return "No transactions found for the specified filters."
        df_display = df.sort_values("Date", ascending=False).head(limit)
        lines = [f"Found {len(df)} transactions (showing latest {min(limit, len(df))}):\n"]
        for _, row in df_display.iterrows():
            lines.append(
                f"  {row['Date'].strftime('%d %b %Y')} | "
                f"{row['Raw_Description']:<30} | "
                f"${row['Amount']:>8.2f} | "
                f"{row['Card']} | "
                f"{row['Category']}"
            )
        total = df["Amount"].sum()
        lines.append(f"\nTotal: ${total:,.2f} across {len(df)} transactions.")
        return "\n".join(lines)

    def get_spending_summary(
        self,
        source: Optional[str] = None,
        month: Optional[str] = None,
        period: Optional[str] = None,
    ) -> str:
        df = self.df.copy()
        if source and source != "All":
            df = df[df["Card"] == source]
        if month:
            df = df[df["Month"].str.lower() == month.lower()]
        if period == "last_30_days":
            df = df[df["Date"] >= datetime.now() - timedelta(days=30)]
        elif period == "last_7_days":
            df = df[df["Date"] >= datetime.now() - timedelta(days=7)]
        if df.empty:
            return "No spending data found."
        summary     = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
        grand_total = summary.sum()
        lines = ["Spending breakdown by category:\n"]
        for cat, amt in summary.items():
            pct = (amt / grand_total * 100) if grand_total > 0 else 0
            lines.append(f"  {cat:<22} ${amt:>8,.2f}  ({pct:>5.1f}%)")
        lines.append(f"\n  {'TOTAL':<22} ${grand_total:>8,.2f}")
        lines.append(f"\nHighest category: {summary.index[0]} (${summary.iloc[0]:,.2f})")
        return "\n".join(lines)

    def get_monthly_comparison(
        self,
        source: Optional[str] = None,
        category: Optional[str] = None,
        last_n_months: int = 3,
    ) -> str:
        df = self.df.copy()
        if source and source != "All":
            df = df[df["Card"] == source]
        if category and category != "All":
            df = df[df["Category"] == category]
        monthly = (
            df.groupby(["Year", "Month_Num", "Month"])["Amount"]
            .sum().reset_index()
            .sort_values(["Year", "Month_Num"], ascending=False)
            .head(last_n_months)
            .sort_values(["Year", "Month_Num"])
        )
        if monthly.empty:
            return "No data available for monthly comparison."
        lines = [f"Monthly spending (last {last_n_months} months):\n"]
        prev = None
        for _, row in monthly.iterrows():
            trend = ""
            if prev is not None:
                diff = row["Amount"] - prev
                pct  = (diff / prev * 100) if prev > 0 else 0
                trend = f"  {'▲' if diff > 0 else '▼'} {abs(pct):.1f}% vs prior month"
            lines.append(
                f"  {row['Month']} {int(row['Year'])}: ${row['Amount']:>8,.2f}{trend}"
            )
            prev = row["Amount"]
        return "\n".join(lines)

    def get_top_merchants(
        self,
        source: Optional[str] = None,
        month: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        df = self.df.copy()
        if source and source != "All":
            df = df[df["Card"] == source]
        if month:
            df = df[df["Month"].str.lower() == month.lower()]
        df["Merchant"] = df["Raw_Description"].str.replace(
            r"\s+\d+$", "", regex=True
        ).str.strip()
        top = (
            df.groupby(["Merchant", "Category"])["Amount"]
            .agg(["sum", "count"]).reset_index()
            .sort_values("sum", ascending=False).head(limit)
        )
        if top.empty:
            return "No merchant data found."
        lines = [f"Top {limit} merchants by spend:\n"]
        for i, (_, row) in enumerate(top.iterrows(), 1):
            lines.append(
                f"  {i}. {row['Merchant']:<28} "
                f"${row['sum']:>8,.2f}  "
                f"({int(row['count'])} transactions, {row['Category']})"
            )
        return "\n".join(lines)

    def get_source_breakdown(self, month: Optional[str] = None) -> str:
        df = self.df.copy()
        if month:
            df = df[df["Month"].str.lower() == month.lower()]
        breakdown   = (
            df.groupby(["Card", "Source_Type"])["Amount"]
            .agg(["sum", "count"]).reset_index()
            .sort_values("sum", ascending=False)
        )
        grand_total = breakdown["sum"].sum()
        lines = ["Spending by payment source:\n"]
        lines.append(
            f"  {'Source':<20} {'Type':<20} {'Total':>10}  {'Txns':>6}  {'Share':>7}"
        )
        lines.append("  " + "─" * 65)
        for _, row in breakdown.iterrows():
            pct = (row["sum"] / grand_total * 100) if grand_total > 0 else 0
            lines.append(
                f"  {row['Card']:<20} {row['Source_Type']:<20} "
                f"${row['sum']:>8,.2f}  {int(row['count']):>6}  {pct:>6.1f}%"
            )
        lines.append("  " + "─" * 65)
        lines.append(f"  {'TOTAL':<20} {'':<20} ${grand_total:>8,.2f}")
        return "\n".join(lines)

    def detect_anomalies(
        self,
        source: Optional[str] = None,
        z_threshold: float = 2.0,
    ) -> str:
        df = self.df.copy()
        if source and source != "All":
            df = df[df["Card"] == source]
        anomalies = []
        for category, group in df.groupby("Category"):
            if len(group) < 3:
                continue
            mean = group["Amount"].mean()
            std  = group["Amount"].std()
            if std == 0:
                continue
            outliers = group[np.abs((group["Amount"] - mean) / std) > z_threshold]
            for _, row in outliers.iterrows():
                anomalies.append({
                    "date":        row["Date"].strftime("%d %b %Y"),
                    "description": row["Raw_Description"],
                    "amount":      row["Amount"],
                    "category":    category,
                    "card":        row["Card"],
                    "z_score":     abs((row["Amount"] - mean) / std),
                    "cat_avg":     mean,
                })
        if not anomalies:
            return "No anomalies detected. All transactions within normal ranges."
        anomalies.sort(key=lambda x: x["z_score"], reverse=True)
        lines = [f"Detected {len(anomalies)} anomalous transaction(s):\n"]
        for a in anomalies[:10]:
            lines.append(
                f"  {a['date']} | {a['description']:<30} | "
                f"${a['amount']:>8.2f} | {a['card']}\n"
                f"           Category avg: ${a['cat_avg']:.2f} | "
                f"Anomaly score: {a['z_score']:.1f}σ\n"
            )
        return "\n".join(lines)

    def get_data_context(self) -> dict:
        df = self.df
        return {
            "sources":            df["Card"].unique().tolist(),
            "categories":         df["Category"].unique().tolist(),
            "date_from":          df["Date"].min().strftime("%d %b %Y"),
            "date_to":            df["Date"].max().strftime("%d %b %Y"),
            "total_transactions": len(df),
            "total_spend":        round(df["Amount"].sum(), 2),
        }
