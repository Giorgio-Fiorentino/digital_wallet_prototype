import streamlit as st
import pandas as pd
import os
import plotly.express as px
from models.categorizer import TransactionAI

# Initialize AI logic
ai = TransactionAI()

# Page Setup - Apple Style
st.set_page_config(page_title="Apple Wallet AI Extension", page_icon="💳", layout="wide")

# Sidebar - User Controls
st.sidebar.header("Wallet Settings")
card_filter = st.sidebar.multiselect(
    "Active Cards", 
    ["Visa ...1234", "Amex ...5678", "Apple Card"], 
    default=["Visa ...1234", "Amex ...5678"]
)
date_range = st.sidebar.date_input("Select Date Range", [])

# Data Pipeline: Loading and Filtering
csv_path = "data/transactions.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Apply Card Filter
    df = df[df['Card'].isin(card_filter)]
    
    # Apply Date Filter (if range is selected)
    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
else:
    st.error("Transaction data not found! Please run the generator.")
    st.stop()

# --- NAVIGATION TABS ---
tab_wallet, tab_home, tab_ai = st.tabs(["🗂️ My Wallet", "📊 Spending Analysis", "🤖 AI Training"])

# 1. TAB WALLET: Apple Wallet Experience
with tab_wallet:
    st.subheader("Your Digital Cards & Passes")
    col_cards, col_passes = st.columns(2)
    
    with col_cards:
        st.markdown("### Payment Methods")
        for card in card_filter:
            with st.container():
                st.info(f"💳 **{card}**\n\nStatus: Active | Region: Europe")
    
    with col_passes:
        st.markdown("### Boarding Passes")
        with st.container():
            # Representing the user's specific trip context
            st.warning("✈️ **Ryanair - Flight FR8342**\n\n**Barcelona (BCN) ➔ Palermo (PMO)**\n\nGate: B22 | Seat: 12F | Date: Tomorrow")

# 2. TAB HOME: Analytics (Pie & Bar Charts)
with tab_home:
    st.subheader("Unified Financial Insights")
    
    # Process categories using our AI model
    df[['Category', 'Conf', 'Feedback']] = df.apply(
        lambda row: pd.Series(ai.predict_category(row['Raw_Description'])), axis=1
    )
    
    # Layout for Charts
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Spending by Category (Share)")
        fig_pie = px.pie(df, values='Amount', names='Category', hole=0.5,
                         color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        st.markdown("#### Spending Volume (€)")
        fig_bar = px.bar(df.groupby('Category')['Amount'].sum().reset_index(), 
                         x='Category', y='Amount', color='Category',
                         color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### Detailed Transaction History")
    st.dataframe(df[['Date', 'Raw_Description', 'Amount', 'Card', 'Category']], use_container_width=True)

# 3. TAB AI TRAINING: The Human-in-the-Loop Feedback
with tab_ai:
    st.header("AI Feedback Center")
    needs_feedback = df[df['Feedback'] == True]
    if not needs_feedback.empty:
        st.write("The AI is uncertain about these transactions. Please manually verify:")
        # Widget usage: Data Editor for interactive feedback
        st.data_editor(needs_feedback[['Raw_Description', 'Category', 'Conf']])
        if st.button("Submit Corrections"):
            st.success("Model updated locally! (Prototype Simulation)")
    else:
        st.success("AI confidence is optimal across all filtered data.")