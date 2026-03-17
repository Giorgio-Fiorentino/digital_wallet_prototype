"""
app.py  ·  v2
Apple / Google Wallet-inspired multi-card intelligent dashboard.

Tabs
  1. 💳  My Wallet      — card visuals + recent transactions + passes
  2. 📊  Smart Insights — charts, monthly trends, anomaly detection
  3. 🤖  AI Assistant   — Cohere agentic chat (tool-use + RAG routing)
  4. 🔬  AI Lab         — model evaluation & human-in-the-loop training
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import date
from dotenv import load_dotenv

from models.wallet_engine import WalletEngine
from models.categorizer   import TFIDFCategorizer, EmbeddingCategorizer
from models.evaluator     import evaluate_categorizer, get_metrics_dataframe

load_dotenv()

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Wallet · AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# GLOBAL CSS — iOS / Apple Wallet aesthetic
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Page ── */
[data-testid="stAppViewContainer"] { background: #f2f2f7; }
[data-testid="stSidebar"]          { background: #1c1c1e !important; }
[data-testid="stSidebar"] *        { color: #f5f5f7 !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p        { color: #98989d !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #2c2c2e; border: 1px solid #3a3a3c; color: #f5f5f7 !important;
    border-radius: 10px;
}
[data-testid="stSidebar"] input[type="text"],
[data-testid="stSidebar"] input[type="date"],
[data-testid="stSidebar"] [data-testid="stDateInput"] input {
    color: #1c1c1e !important;
    background: #ffffff !important;
}

/* ── Credit-card component ── */
.cc-card {
    border-radius: 20px;
    padding: 22px 24px 18px;
    color: white;
    min-height: 170px;
    position: relative;
    font-family: -apple-system, "SF Pro Display", BlinkMacSystemFont, sans-serif;
    box-shadow: 0 14px 42px rgba(0,0,0,.26);
    overflow: hidden;
    margin-bottom: 14px;
}
.cc-card::after {
    content: '';
    position: absolute;
    top: -55px; right: -55px;
    width: 210px; height: 210px;
    border-radius: 50%;
    background: rgba(255,255,255,.09);
}
.cc-chip  { width: 36px; height: 26px; background: rgba(255,255,255,.32);
            border-radius: 5px; margin-bottom: 20px; }
.cc-num   { font-size: 15px; letter-spacing: 3px; font-weight: 600; opacity: .88; }
.cc-name  { font-size: 11px; opacity: .62; margin-top: 6px;
            letter-spacing: 1.2px; text-transform: uppercase; }
.cc-spend { position: absolute; top: 20px; right: 22px; text-align: right; }
.cc-sl    { font-size: 10px; opacity: .58; text-transform: uppercase; letter-spacing: .5px; }
.cc-sv    { font-size: 21px; font-weight: 700; }

/* ── KPI tile ── */
.kpi {
    background: white; border-radius: 16px;
    padding: 16px 20px; box-shadow: 0 2px 10px rgba(0,0,0,.06);
    margin-bottom: 12px;
}
.kpi-lbl { font-size: 11px; color: #6e6e73; font-weight: 500;
           text-transform: uppercase; letter-spacing: .5px; }
.kpi-val { font-size: 28px; font-weight: 700; color: #1c1c1e;
           letter-spacing: -1px; margin: 4px 0 2px; }
.kpi-dlt { font-size: 12px; font-weight: 600; }
.green   { color: #34c759; }
.red     { color: #ff3b30; }
.blue    { color: #007aff; }

/* ── Transaction row ── */
.txn {
    display: flex; align-items: center; gap: 12px;
    padding: 11px 0; border-bottom: 1px solid #f2f2f7;
}
.txn:last-child { border-bottom: none; }
.txn-ico {
    width: 38px; height: 38px; border-radius: 50%;
    background: #f2f2f7; display: flex; align-items: center;
    justify-content: center; font-size: 16px; flex-shrink: 0;
}
.txn-nm  { font-weight: 600; font-size: 14px; color: #1c1c1e; }
.txn-sub { font-size: 11px; color: #8e8e93; margin-top: 1px; }
.txn-amt { margin-left: auto; font-weight: 700; font-size: 15px;
           color: #1c1c1e; white-space: nowrap; }

/* ── Panel (white rounded card) ── */
.panel {
    background: white; border-radius: 16px;
    padding: 16px 20px; box-shadow: 0 2px 8px rgba(0,0,0,.05);
    margin-bottom: 14px;
}

/* ── Boarding pass ── */
.bp  { background: white; border-radius: 18px; overflow: hidden;
       box-shadow: 0 4px 20px rgba(0,0,0,.09); margin-bottom: 16px; }
.bp-hdr { background: #1a73e8; color: white; padding: 15px 22px; }
.bp-body { padding: 16px 22px; display: flex; gap: 28px; flex-wrap: wrap; }
.bp-fl  { font-size: 10px; color: #8e8e93; text-transform: uppercase; letter-spacing: .5px; }
.bp-fv  { font-size: 20px; font-weight: 700; color: #1c1c1e; }
.bp-qr  { padding: 14px 22px; border-top: 2px dashed #e5e5ea;
          display: flex; align-items: center; gap: 14px; }

/* ── Chat bubbles ── */
.msg-wrap-u { display: flex; justify-content: flex-end; margin: 5px 0; }
.msg-wrap-b { display: flex; justify-content: flex-start; margin: 5px 0; }
.msg-u {
    background: #007aff; color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 14px; max-width: 72%; font-size: 14px; line-height: 1.5;
}
.msg-b {
    background: white; color: #1c1c1e;
    border-radius: 18px 18px 18px 4px;
    padding: 10px 14px; max-width: 80%; font-size: 14px; line-height: 1.5;
    box-shadow: 0 2px 8px rgba(0,0,0,.07);
}
.msg-tag { font-size: 10px; color: #8e8e93; margin-top: 3px; padding-left: 4px; }

/* ── Anomaly alert ── */
.anom {
    background: #fff5f5; border-left: 4px solid #ff3b30;
    border-radius: 10px; padding: 10px 14px; margin: 5px 0;
}
.anom-d { font-weight: 600; font-size: 14px; color: #1c1c1e; }
.anom-m { font-size: 11px; color: #8e8e93; margin-top: 3px; }

/* ── Tab styling ── */
[data-testid="stTab"] > button { font-weight: 600; font-size: 14px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════
CARD_GRADIENTS = {
    "Visa Classic":   "linear-gradient(135deg,#1a1f71 0%,#4a90e2 100%)",
    "Mastercard Gold":"linear-gradient(135deg,#7b4f12 0%,#d4a017 100%)",
    "Revolut":        "linear-gradient(135deg,#191c1f 0%,#5a5e65 100%)",
    "PayPal":         "linear-gradient(135deg,#003087 0%,#009cde 100%)",
    "Trade Republic": "linear-gradient(135deg,#0a0a0a 0%,#1db954 100%)",
}
CARD_LAST4 = {
    "Visa Classic":"1234", "Mastercard Gold":"5678",
    "Revolut":"9012",      "PayPal":"3456",
    "Trade Republic":"7890",
}
CAT_ICONS = {
    "Food & Dining":"🍽️",  "Grocery":"🛒",        "Shopping":"🛍️",
    "Gas & Transport":"⛽", "Travel":"✈️",          "Entertainment":"🎬",
    "Health & Fitness":"💊","Home":"🏠",            "Personal Care":"💄",
    "Kids":"🧒",            "Misc":"💳",
}
PALETTE = [
    "#007aff","#34c759","#ff9500","#ff3b30","#af52de",
    "#5ac8fa","#ffcc00","#ff2d55","#4cd964","#5856d6",
]
QUICK_PROMPTS = [
    "How much did I spend in total?",
    "What are my top 5 merchants?",
    "Show spending by category",
    "Any suspicious charges?",
    "What are Revolut's foreign transaction fees?",
    "Compare spending across all my cards",
]


# ══════════════════════════════════════════════════════════════════════
# CACHED RESOURCES
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading wallet data…")
def load_engine():
    e = WalletEngine()
    e.load_data()
    return e


@st.cache_resource(show_spinner="Starting AI assistant…")
def load_ai():
    """Returns (WalletLLM, RAGEngine) or (None, None) if key is missing."""
    api_key = os.getenv("COHERE_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return None, None
    try:
        from models.rag_engine import RAGEngine
        from models.llm_engine import WalletLLM
        engine = load_engine()
        rag    = RAGEngine()
        llm    = WalletLLM(engine, rag)
        return llm, rag
    except Exception:
        return None, None


# ══════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ══════════════════════════════════════════════════════════════════════
def cc_html(source: str, spend: float) -> str:
    grad = CARD_GRADIENTS.get(source, "linear-gradient(135deg,#555,#888)")
    l4   = CARD_LAST4.get(source, "••••")
    return f"""
<div class="cc-card" style="background:{grad}">
  <div class="cc-chip"></div>
  <div class="cc-spend">
    <div class="cc-sl">Period Spend</div>
    <div class="cc-sv">${spend:,.0f}</div>
  </div>
  <div class="cc-num">•••• •••• •••• {l4}</div>
  <div class="cc-name">{source}</div>
</div>"""


def kpi_html(label: str, val: str, delta: str = "", color: str = "green") -> str:
    dlt = f'<div class="kpi-dlt {color}">{delta}</div>' if delta else ""
    return (
        f'<div class="kpi">'
        f'<div class="kpi-lbl">{label}</div>'
        f'<div class="kpi-val">{val}</div>'
        f'{dlt}</div>'
    )


def txn_html(merchant: str, category: str, amount: float, dt: str, card: str) -> str:
    ico = CAT_ICONS.get(category, "💳")
    return (
        f'<div class="txn">'
        f'<div class="txn-ico">{ico}</div>'
        f'<div><div class="txn-nm">{merchant}</div>'
        f'<div class="txn-sub">{dt} · {card}</div></div>'
        f'<div class="txn-amt">−${amount:,.2f}</div>'
        f'</div>'
    )


BOARDING_PASS_HTML = """
<div class="bp">
  <div class="bp-hdr">
    <b>✈️ Ryanair · FR8342</b>
    <span style="float:right;opacity:.8;font-size:12px">Boarding Pass</span>
  </div>
  <div class="bp-body">
    <div><div class="bp-fl">From</div><div class="bp-fv">BCN</div></div>
    <div style="align-self:center;font-size:22px;color:#c7c7cc">→</div>
    <div><div class="bp-fl">To</div><div class="bp-fv">PMO</div></div>
    <div><div class="bp-fl">Date</div><div class="bp-fv">17 Mar</div></div>
    <div><div class="bp-fl">Gate</div><div class="bp-fv">B22</div></div>
    <div><div class="bp-fl">Seat</div><div class="bp-fv">12F</div></div>
  </div>
  <div class="bp-qr">
    <div style="font-family:monospace;font-size:34px;letter-spacing:-2px;color:#1c1c1e">▌▐██▌▐▌</div>
    <div>
      <div style="font-weight:700;color:#1c1c1e;font-size:15px">GIORGIO FIORENTINO</div>
      <div style="font-size:12px;color:#34c759;margin-top:3px">✓ Check-in complete · Economy · 1 carry-on</div>
    </div>
  </div>
</div>"""


# ══════════════════════════════════════════════════════════════════════
# FILTER HELPER
# ══════════════════════════════════════════════════════════════════════
def apply_filters(engine: WalletEngine, cards: list, dr) -> pd.DataFrame:
    df = engine.df.copy()
    if cards:
        df = df[df["Card"].isin(cards)]
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        df = df[
            (df["Date"] >= pd.to_datetime(dr[0])) &
            (df["Date"] <= pd.to_datetime(dr[1]))
        ]
    return df


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💳 Smart Wallet")
    st.caption("Multi-card AI Financial Dashboard")
    st.divider()

    try:
        engine  = load_engine()
        sources = engine.valid_sources
        data_ok = True
    except FileNotFoundError:
        st.error(
            "⚠️ No transaction data found.\n\n"
            "Run:\n```\npython3 data/loader.py\n```"
        )
        sources = []
        data_ok = False

    selected_cards = st.multiselect(
        "Payment Sources",
        options=sources,
        default=sources,
        help="Select cards to include in the dashboard",
    )

    st.markdown("#### Date Range")
    if data_ok:
        _df = engine.df
        _min_date = _df["Date"].min().date()
        _max_date = _df["Date"].max().date()
    else:
        _min_date = date(2019, 1, 1)
        _max_date = date.today()
    dr = st.date_input(
        "Period",
        value=[_min_date, _max_date],
        min_value=_min_date,
        max_value=_max_date,
        format="YYYY/MM/DD",
        label_visibility="collapsed",
    )

    if st.button("⟳  Reset Filters", use_container_width=True):
        st.rerun()

    st.divider()
    if data_ok:
        ctx = engine.get_data_context()
        st.caption(f"📅 {ctx['date_from']} → {ctx['date_to']}")
        st.caption(f"🔢 {ctx['total_transactions']:,} transactions")
        st.caption(f"💰 ${ctx['total_spend']:,.2f} total")


# ══════════════════════════════════════════════════════════════════════
# GUARD — data must exist
# ══════════════════════════════════════════════════════════════════════
if not data_ok:
    st.error(
        "Transaction data not found. "
        "Please run `python3 data/loader.py` to generate it."
    )
    st.stop()

df = apply_filters(engine, selected_cards, dr)


# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════
tab_wallet, tab_insights, tab_chat, tab_lab = st.tabs([
    "💳  My Wallet",
    "📊  Smart Insights",
    "🤖  AI Assistant",
    "🔬  AI Lab",
])


# ──────────────────────────────────────────────────────────────────────
# TAB 1 — MY WALLET
# ──────────────────────────────────────────────────────────────────────
with tab_wallet:

    # ── Card stack
    st.markdown("### Payment Methods")

    per_card = (
        df.groupby("Card")["Amount"].sum()
        if not df.empty
        else pd.Series(dtype=float)
    )
    if selected_cards:
        n_cols = min(len(selected_cards), 3)
        cols   = st.columns(n_cols)
        for i, card in enumerate(selected_cards):
            with cols[i % n_cols]:
                st.markdown(
                    cc_html(card, float(per_card.get(card, 0))),
                    unsafe_allow_html=True,
                )
    else:
        st.info("Select at least one payment source in the sidebar.")

    st.divider()

    # ── Quick stats
    st.markdown("### Period Overview")
    total   = df["Amount"].sum()
    n_txn   = len(df)
    avg_txn = df["Amount"].mean() if n_txn else 0.0
    n_fraud = int(df["Is_Fraud"].sum()) if ("Is_Fraud" in df.columns and n_txn) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_html("Total Spent",     f"${total:,.2f}"),   unsafe_allow_html=True)
    c2.markdown(kpi_html("Transactions",    f"{n_txn:,}"),        unsafe_allow_html=True)
    c3.markdown(kpi_html("Avg Transaction", f"${avg_txn:,.2f}"),  unsafe_allow_html=True)
    c4.markdown(
        kpi_html(
            "Flagged Fraud", str(n_fraud),
            delta=f"⚠️ {n_fraud} flagged" if n_fraud else "✓ None detected",
            color="red" if n_fraud else "green",
        ),
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Recent transactions
    st.markdown("### Recent Transactions")
    if not df.empty:
        recent    = df.sort_values("Date", ascending=False).head(12)
        rows_html = "".join(
            txn_html(
                row["Raw_Description"],
                row.get("Category", "Misc"),
                row["Amount"],
                row["Date"].strftime("%d %b"),
                row["Card"],
            )
            for _, row in recent.iterrows()
        )
        st.markdown(f'<div class="panel">{rows_html}</div>', unsafe_allow_html=True)
    else:
        st.info("No transactions found for the selected period.")

    st.divider()

    # ── Passes & tickets
    st.markdown("### Passes & Tickets")
    col_pass, col_empty = st.columns(2)
    with col_pass:
        st.markdown(BOARDING_PASS_HTML, unsafe_allow_html=True)
    with col_empty:
        st.markdown("""
<div class="panel" style="min-height:160px;display:flex;flex-direction:column;
     align-items:center;justify-content:center;gap:8px;color:#c7c7cc;">
  <div style="font-size:40px">🎫</div>
  <div style="font-size:13px">Add more passes here</div>
</div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# TAB 2 — SMART INSIGHTS
# ──────────────────────────────────────────────────────────────────────
with tab_insights:
    if df.empty:
        st.warning("No data available for the selected filters.")
    else:
        # ── KPI row
        st.markdown("### Spending Overview")
        top_cat  = df.groupby("Category")["Amount"].sum().idxmax()
        top_card = df.groupby("Card")["Amount"].sum().idxmax()

        c1, c2, c3 = st.columns(3)
        c1.markdown(kpi_html("Total Spent",    f"${df['Amount'].sum():,.2f}"), unsafe_allow_html=True)
        c2.markdown(kpi_html("Top Category",   top_cat,  color="blue"),        unsafe_allow_html=True)
        c3.markdown(kpi_html("Most Used Card", top_card, color="blue"),        unsafe_allow_html=True)

        st.divider()

        # ── Donut + Source bar
        st.markdown("### Spending Breakdown")
        col_donut, col_bar = st.columns(2)

        cat_spend = (
            df.groupby("Category")["Amount"]
            .sum().reset_index()
            .sort_values("Amount", ascending=False)
        )
        with col_donut:
            fig_donut = px.pie(
                cat_spend, values="Amount", names="Category",
                hole=0.55, color_discrete_sequence=PALETTE,
                title="By Category",
            )
            fig_donut.update_traces(textposition="outside", textinfo="label+percent")
            fig_donut.update_layout(
                showlegend=False,
                margin=dict(t=50,b=10,l=10,r=10),
                paper_bgcolor="white", plot_bgcolor="white",
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        card_spend = (
            df.groupby("Card")["Amount"]
            .sum().reset_index()
            .sort_values("Amount")
        )
        with col_bar:
            fig_bar = px.bar(
                card_spend, x="Amount", y="Card", orientation="h",
                color="Card", color_discrete_sequence=PALETTE,
                title="By Payment Source",
                labels={"Amount": "Total ($)", "Card": ""},
            )
            fig_bar.update_layout(
                showlegend=False,
                margin=dict(t=50,b=10,l=10,r=10),
                paper_bgcolor="white", plot_bgcolor="#fafafa",
                yaxis=dict(tickfont=dict(size=13)),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # ── Monthly trend
        st.markdown("### Monthly Trend")
        available_cats = ["All"] + sorted(df["Category"].unique().tolist())
        sel_cats = st.multiselect(
            "Filter by category",
            available_cats,
            default=["All"],
            key="trend_cats",
        )

        trend_src = (
            df if ("All" in sel_cats or not sel_cats)
            else df[df["Category"].isin(sel_cats)]
        )
        monthly = (
            trend_src.groupby(["Year", "Month_Num", "Month"])["Amount"]
            .sum().reset_index()
            .sort_values(["Year", "Month_Num"])
        )
        monthly["Label"] = monthly["Month"] + " " + monthly["Year"].astype(str)

        fig_trend = px.line(
            monthly, x="Label", y="Amount", markers=True,
            labels={"Amount": "Total Spent ($)", "Label": "Month"},
            color_discrete_sequence=["#007aff"],
        )
        fig_trend.update_traces(line_width=2.5, marker_size=8)
        fig_trend.update_layout(
            margin=dict(t=10,b=10,l=10,r=10),
            paper_bgcolor="white", plot_bgcolor="#fafafa",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        st.divider()

        # ── Top merchants
        st.markdown("### Top Merchants")
        df_m = df.copy()
        df_m["Merchant"] = (
            df_m["Raw_Description"]
            .str.replace(r"\s+\d+$", "", regex=True)
            .str.strip()
        )
        top_merch = (
            df_m.groupby(["Merchant", "Category"])["Amount"]
            .agg(total="sum", count="count").reset_index()
            .sort_values("total", ascending=False).head(10)
        )
        fig_merch = px.bar(
            top_merch, x="Merchant", y="total",
            color="Category", color_discrete_sequence=PALETTE,
            labels={"total": "Total ($)", "Merchant": ""},
            title="Top 10 Merchants by Spend",
        )
        fig_merch.update_layout(
            xaxis_tickangle=-35,
            margin=dict(t=50,b=100,l=10,r=10),
            paper_bgcolor="white", plot_bgcolor="#fafafa",
        )
        st.plotly_chart(fig_merch, use_container_width=True)

        st.divider()

        # ── Anomaly detection
        st.markdown("### ⚠️ Anomaly Detection")
        st.caption(
            "Transactions that deviate significantly from their category average (z-score > 2σ)"
        )

        anomalies = []
        for category, group in df.groupby("Category"):
            if len(group) < 3:
                continue
            mean = group["Amount"].mean()
            std  = group["Amount"].std()
            if std == 0:
                continue
            outliers = group[np.abs((group["Amount"] - mean) / std) > 2.0]
            for _, row in outliers.iterrows():
                anomalies.append({
                    "date":   row["Date"].strftime("%d %b %Y"),
                    "desc":   row["Raw_Description"],
                    "amount": row["Amount"],
                    "cat":    category,
                    "card":   row["Card"],
                    "z":      abs((row["Amount"] - mean) / std),
                    "avg":    mean,
                })
        anomalies.sort(key=lambda x: x["z"], reverse=True)

        if not anomalies:
            st.success("✓ No anomalies detected — all transactions are within normal ranges.")
        else:
            st.warning(f"Detected **{len(anomalies)}** anomalous transaction(s)")
            for a in anomalies[:8]:
                st.markdown(f"""
<div class="anom">
  <div class="anom-d">
    {CAT_ICONS.get(a['cat'], '💳')} {a['desc']}
    <span style="float:right;color:#ff3b30">${a['amount']:,.2f}</span>
  </div>
  <div class="anom-m">
    {a['date']} · {a['card']} · {a['cat']}
    &nbsp;|&nbsp; Category avg: ${a['avg']:,.2f}
    &nbsp;|&nbsp; Anomaly score: {a['z']:.1f}σ
  </div>
</div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# TAB 3 — AI ASSISTANT
# ──────────────────────────────────────────────────────────────────────
with tab_chat:
    llm, rag = load_ai()

    if llm is None:
        st.warning("""
**AI Assistant requires a Cohere API key.**

1. Copy `.env.example` → `.env`
2. Add your key:  `COHERE_API_KEY=your_key_here`
3. Restart the app

The assistant answers questions about your **transactions** (via tool use) and
your **card terms & fees** (via RAG over the docs in `docs/card_terms/`).
""")
    else:
        # Session state
        if "chat_history"  not in st.session_state:
            st.session_state.chat_history  = []
        if "tool_log"      not in st.session_state:
            st.session_state.tool_log      = []
        if "display_msgs"  not in st.session_state:
            st.session_state.display_msgs  = []

        st.markdown("### 🤖 AI Financial Assistant")
        st.caption(
            "Ask about your spending, top merchants, monthly trends, or query "
            "card terms & fees. Routes automatically between **transaction tools** "
            "and **RAG document search**."
        )

        # Quick prompts
        st.markdown("**Quick prompts**")
        qcols = st.columns(3)
        for i, prompt in enumerate(QUICK_PROMPTS):
            if qcols[i % 3].button(prompt, key=f"qp_{i}", use_container_width=True):
                st.session_state["_pending"] = prompt

        st.divider()

        # Chat messages
        for msg in st.session_state.display_msgs:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="msg-wrap-u"><div class="msg-u">{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                route     = msg.get("route", "tool_use")
                tag_color = "#5ac8fa" if route == "rag" else "#34c759"
                route_lbl = "📚 RAG · Card Terms" if route == "rag" else "🔧 Tool Use · Transactions"
                st.markdown(f"""
<div class="msg-wrap-b">
  <div>
    <div class="msg-b">{msg["content"]}</div>
    <div class="msg-tag" style="color:{tag_color}">{route_lbl}</div>
  </div>
</div>""", unsafe_allow_html=True)

        # Input
        user_input = st.chat_input("Ask anything about your wallet…")
        pending    = None
        if "_pending" in st.session_state:
            pending = st.session_state["_pending"]
            del st.session_state["_pending"]
        query = pending or user_input

        if query:
            with st.spinner("Thinking…"):
                answer, new_hist, tools_used, route = llm.chat(
                    query, st.session_state.chat_history
                )
            st.session_state.chat_history = new_hist
            st.session_state.tool_log.extend(tools_used)
            st.session_state.display_msgs.append({"role": "user", "content": query})
            st.session_state.display_msgs.append({
                "role": "bot", "content": answer, "route": route
            })
            st.rerun()

        # Tool log
        if st.session_state.tool_log:
            with st.expander(
                f"🔧 Tool calls  ({len(st.session_state.tool_log)} total)",
                expanded=False,
            ):
                for name, args, result in reversed(st.session_state.tool_log[-6:]):
                    st.markdown(f"**`{name}`** · args: `{args}`")
                    st.code(str(result)[:500], language=None)
                    st.markdown("---")

        if st.button("🗑️  Clear conversation", key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.tool_log     = []
            st.session_state.display_msgs = []
            st.rerun()


# ──────────────────────────────────────────────────────────────────────
# TAB 4 — AI LAB
# ──────────────────────────────────────────────────────────────────────
with tab_lab:
    st.markdown("### 🔬 AI Lab — Model Evaluation & Training")
    st.caption(
        "Compare TF-IDF (v1 baseline) vs Cohere Embeddings categorization, "
        "review uncertain transactions, and manually correct the model."
    )

    # ── Section 1 — Model evaluation
    st.markdown("#### 1 · Model Comparison")
    col_cfg, col_info = st.columns([1, 2])

    with col_cfg:
        sample_size = st.slider("Evaluation sample size", 50, 300, 100, 25)
        tfidf_only  = st.checkbox("TF-IDF only  (no API calls)", value=True)
        run_eval    = st.button("▶  Run Evaluation", type="primary", use_container_width=True)

    with col_info:
        st.info("""
**What this does**
- Runs both categorisers on a stratified sample of your transactions
- Compares predictions against the Kaggle ground-truth labels
- Reports Precision, Recall and F1 per class (macro-averaged)

*Cohere Embeddings mode requires `COHERE_API_KEY` in your `.env`.*
""")

    if run_eval:
        with st.spinner("Evaluating TF-IDF…"):
            try:
                tfidf_res = evaluate_categorizer(TFIDFCategorizer(engine.df), engine.df, sample_size)
                st.session_state["tfidf_res"] = tfidf_res

                if not tfidf_only:
                    api_key = os.getenv("COHERE_API_KEY", "")
                    if api_key and not api_key.startswith("your_"):
                        with st.spinner("Evaluating Cohere Embeddings…"):
                            embed_res = evaluate_categorizer(
                                EmbeddingCategorizer(engine.df), engine.df, sample_size
                            )
                            st.session_state["embed_res"] = embed_res
                    else:
                        st.warning("Cohere API key not set — skipping embedding evaluation.")
                        st.session_state.pop("embed_res", None)
                else:
                    st.session_state.pop("embed_res", None)

                st.success("✓ Evaluation complete!")
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")

    if "tfidf_res" in st.session_state:
        tr = st.session_state["tfidf_res"]
        er = st.session_state.get("embed_res")

        col_t, col_e = st.columns(2)
        with col_t:
            st.markdown("**TF-IDF Baseline**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy",  f"{tr['accuracy']:.1%}")
            m2.metric("Macro F1",  f"{tr['macro_f1']:.1%}")
            m3.metric("Avg Conf.", f"{tr['avg_confidence']:.1%}")

        if er:
            with col_e:
                st.markdown("**Cohere Embeddings**")
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy",  f"{er['accuracy']:.1%}",
                          delta=f"{er['accuracy'] - tr['accuracy']:+.1%}")
                m2.metric("Macro F1",  f"{er['macro_f1']:.1%}",
                          delta=f"{er['macro_f1'] - tr['macro_f1']:+.1%}")
                m3.metric("Avg Conf.", f"{er['avg_confidence']:.1%}",
                          delta=f"{er['avg_confidence'] - tr['avg_confidence']:+.1%}")

        st.markdown("**Per-class Performance (TF-IDF)**")
        pc_df   = get_metrics_dataframe(tr)
        fig_pc  = px.bar(
            pc_df.melt(id_vars="Category", value_vars=["Precision", "Recall", "F1 Score"]),
            x="Category", y="value", color="variable", barmode="group",
            color_discrete_sequence=["#007aff", "#34c759", "#ff9500"],
            labels={"value": "Score", "variable": "Metric"},
        )
        fig_pc.update_layout(
            xaxis_tickangle=-35,
            margin=dict(t=10,b=90,l=10,r=10),
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig_pc, use_container_width=True)

    st.divider()

    # ── Section 2 — Uncertain transactions
    st.markdown("#### 2 · Uncertain Transactions  *(Human-in-the-Loop)*")
    st.caption(
        "Transactions where TF-IDF cosine similarity is below the confidence "
        "threshold — review and confirm or correct."
    )

    if st.button("🔍  Scan for Uncertain Transactions", key="scan_unc"):
        cat    = TFIDFCategorizer(engine.df)
        sample = engine.df.sample(min(200, len(engine.df)), random_state=42)
        found  = []
        for _, row in sample.iterrows():
            pred, conf, flagged = cat.predict(row["Raw_Description"])
            if flagged:
                found.append({
                    "desc":  row["Raw_Description"],
                    "truth": row["Category"],
                    "pred":  pred,
                    "conf":  conf,
                })
        st.session_state["uncertain"] = found

    if "uncertain" in st.session_state:
        unc = st.session_state["uncertain"]
        if unc:
            st.info(f"Found **{len(unc)}** uncertain transaction(s) in the sample.")
            for i, item in enumerate(unc[:8]):
                with st.expander(
                    f"⚠️  {item['desc']}  ·  predicted: **{item['pred']}**",
                    expanded=False,
                ):
                    c1, c2, c3 = st.columns(3)
                    c1.write(f"**Ground truth:** {item['truth']}")
                    c2.write(f"**AI prediction:** {item['pred']}")
                    conf_val = min(float(item["conf"]), 1.0)
                    c3.progress(conf_val, text=f"Confidence: {conf_val:.0%}")
                    if st.button("✅  Confirm prediction", key=f"conf_{i}"):
                        st.success("Training data updated!")
        else:
            st.success("✓ No uncertain transactions found in the sample.")

    st.divider()

    # ── Section 3 — Manual training
    st.markdown("#### 3 · Manual Training")
    st.caption("Override the model's classification for a specific merchant description.")

    all_cats  = sorted(engine.df["Category"].unique().tolist())
    all_descs = sorted(engine.df["Raw_Description"].unique().tolist())

    col_d, col_c, col_b = st.columns([2, 1, 1])
    with col_d:
        target_desc = st.selectbox("Select description", all_descs, key="train_desc")
    with col_c:
        manual_cat  = st.selectbox("Correct category",  all_cats,  key="train_cat")
    with col_b:
        st.write(" ")
        if st.button("🚀  Train Model", use_container_width=True, key="train_btn"):
            if "manual_tfidf" not in st.session_state:
                st.session_state.manual_tfidf = TFIDFCategorizer()
            st.session_state.manual_tfidf.train(target_desc, manual_cat)
            st.success(f"✓  '{target_desc}'  →  '{manual_cat}'")
            st.balloons()


# ══════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption("Smart Wallet · Prototype v2 · Powered by Cohere · Built with Streamlit")