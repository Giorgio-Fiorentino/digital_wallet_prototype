import streamlit as st
import pandas as pd
import os
import plotly.express as px # Usiamo Plotly per grafici più belli e interattivi
from models.categorizer import TransactionAI

ai = TransactionAI()

st.set_page_config(page_title="Apple Wallet AI Extension", page_icon="💳", layout="wide")

# Sidebar - Filtri Temporali e Carte
st.sidebar.header("Wallet Settings")
card_filter = st.sidebar.multiselect(
    "Cards in Wallet", 
    ["Visa ...1234", "Amex ...5678", "Apple Card"], 
    default=["Visa ...1234", "Amex ...5678"]
)
date_range = st.sidebar.date_input("Filter by Date", [])

# Caricamento Dati
csv_path = "data/transactions.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date']) # Assicuriamoci che sia in formato data
    
    # Filtro per Carta
    df = df[df['Card'].isin(card_filter)]
    
    # Filtro per Data (se selezionata)
    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
else:
    st.error("CSV non trovato!")
    st.stop()

# --- TABS ---
tab_wallet, tab_home, tab_ai = st.tabs(["🗂️ My Wallet", "📊 Home Analysis", "🤖 AI Feedback"])

# 1. TAB WALLET: Estensione Apple Wallet
with tab_wallet:
    st.subheader("Your Digital Cards & Passes")
    
    col_cards, col_passes = st.columns(2)
    
    with col_cards:
        st.markdown("### Payment Cards")
        for card in card_filter:
            with st.container():
                st.info(f"💳 **{card}**\n\nActive and Ready for Apple Pay")
    
    with col_passes:
        st.markdown("### Passes & Boarding Passes")
        with st.container():
            st.warning("✈️ **Ryanair - FR8342**\n\n**Barcelona (BCN) ➔ Palermo (PMO)**\n\nGate: B22 | Seat: 12F")

# 2. TAB HOME ANALYSIS: Grafici e Transazioni
with tab_home:
    st.subheader("Spending Overview")
    
    # Prepariamo i dati categorizzati per i grafici
    df[['Category', 'Conf', 'Feedback']] = df.apply(
        lambda row: pd.Series(ai.predict_category(row['Raw_Description'])), axis=1
    )
    
    # Layout a due colonne per i grafici
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Category Distribution (Pie)")
        fig_pie = px.pie(df, values='Amount', names='Category', hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        st.markdown("#### Spending per Category (Bar)")
        fig_bar = px.bar(df.groupby('Category')['Amount'].sum().reset_index(), 
                         x='Category', y='Amount', color='Category',
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### Filtered Transactions")
    st.dataframe(df[['Date', 'Raw_Description', 'Amount', 'Card', 'Category']], use_container_width=True)

# 3. TAB AI FEEDBACK (Adattata)
with tab_ai:
    st.header("AI Training Center")
    needs_f = df[df['Feedback'] == True]
    if not needs_f.empty:
        st.write("Confirm categorization for low-confidence transactions:")
        st.data_editor(needs_f[['Raw_Description', 'Category', 'Conf']])
    else:
        st.success("All transactions accurately categorized by AI!")