import streamlit as st
import pandas as pd
import os
import plotly.express as px
from models.categorizer import TransactionAI
import datetime

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

# And date selection
months_map = {
    "January": 1, "February": 2, "March": 3, "April": 4, 
    "May": 5, "June": 6, "July": 7, "August": 8, 
    "September": 9, "October": 10, "November": 11, "December": 12
}
selected_month = st.sidebar.selectbox("Select Month", ["All"] + list(months_map.keys()))

# 2. Dynamic Date Range Default
current_year = datetime.datetime.now().year

if selected_month != "All":
    # Set the default calendar view to the 1st of the selected month
    month_num = months_map[selected_month]
    default_date = datetime.date(current_year, month_num, 1)
else:
    # Standard default (today)
    default_date = datetime.date.today()

# 3. Date Input Widget
# We use 'value' to force the calendar to the correct month
date_range = st.sidebar.date_input(
    "Select Date Range", 
    value=[default_date, default_date + datetime.timedelta(days=6)], # Default 1 week range
    format="YYYY/MM/DD"
)

# Data Pipeline: Loading and Filtering
csv_path = "data/transactions.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month_name()
    
    # Apply Card Filter
    df = df[df['Card'].isin(card_filter)]
    
    # Apply Month Filter
    if selected_month != "All":
        df = df[df['Month'] == selected_month]

    # Apply Date Filter (if range is selected)
    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
else:
    st.error("Transaction data not found! Please run the generator.")
    st.stop()

# --- NAVIGATION TABS ---
tab_wallet, tab_home, tab_ai = st.tabs(["🗂️ My Wallet", "📊 Spending Analysis", "🤖 Augment your wallet"])

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
    if not df.empty:
    # Use a lambda to apply your predict_category method
    # Ensure the method returns exactly 3 values: (Category, Confidence, Feedback_Req)
        df[['Category', 'Confidence', 'Feedback_Req']] = df.apply(
        lambda row: pd.Series(ai.predict_category(row['Raw_Description'])), axis=1
    )
    
    else:
    # Create empty columns if df is empty to prevent downstream errors in charts
        st.warning("No transactions found for the selected filters.")
        df['Category'] = None
        df['Confidence'] = None
        df['Feedback_Req'] = False
    
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

    st.divider()
    st.subheader("Monthly Comparison Analysis")
    st.write("Compare your spending habits across different months.")

    # Widget: Category Selector for the Trend Chart
    # We use all unique categories found after AI processing
    available_categories = df['Category'].unique().tolist()
    compare_cats = st.multiselect(
        "Select Categories to Compare", 
        options=available_categories, 
        default=available_categories[:3] # Default to first 3 to keep it clean
    )

    if compare_cats:
        # Filter data for selected categories
        trend_df = df[df['Category'].isin(compare_cats)]
        
        # Group by Month and Category to see the split in the line chart
        # We ensure months are sorted chronologically if possible
        monthly_trend = trend_df.groupby(['Month', 'Category'])['Amount'].sum().reset_index()
        
        # Plotly Line Chart with 'color' parameter to show different lines per category
        fig_trend = px.line(
            monthly_trend, 
            x='Month', 
            y='Amount', 
            color='Category', 
            markers=True,
            title="Spending Trend by Category",
            labels={"Amount": "Total Spent (€)", "Month": "Month"}
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("Please select at least one category to view the trend.")

# 3. TAB AI TRAINING: The Human-in-the-Loop Feedback
with tab_ai:
    st.header("AI Training Center")
    st.write("Customize your categories and refine the AI's accuracy through feedback.")

    # --- SUB-SECTION 1: Define New Categories ---
    with st.expander("➕ Define New Custom Category"):
        new_cat_name = st.text_input("Category Name (e.g., Hobbies)")
        new_cat_desc = st.text_area("Description (Help the AI understand context)")
        if st.button("Add Category"):
            # This demonstrates the 'Purpose of the prototype' (Slide 54/Lecture 1)
            st.success(f"Category '{new_cat_name}' registered!")

    st.divider()

    # --- SUB-SECTION 2: Active Learning (Automated Review) ---
    st.subheader("⚠️ Uncertain Transactions")
    st.caption("The AI flagged these for review based on a low confidence score.")
    
    # Filtering based on the AI logic in categorizer.py
    needs_feedback = df[df['Feedback_Req'] == True]
    
    if not needs_feedback.empty:
        for index, row in needs_feedback.iterrows():
            with st.expander(f"Review Required: {row['Raw_Description']}"):
                c1, c2 = st.columns(2)
                c1.write(f"AI Suggestion: **{row['Category']}**")
                c1.progress(row['Confidence'], text=f"Confidence: {int(row['Confidence']*100)}%")
                
                if c2.button("Confirm ✅", key=f"btn_{index}"):
                    st.success("Training data updated!")
    else:
        st.info("No uncertain transactions found. The model is performing well!")

    st.divider()

    # --- SUB-SECTION 3: Manual Model Training (User Input) ---
    st.subheader("✍️ Manual Model Training")
    st.write("Force a specific classification for a merchant or description.")
    
    col_input, col_cat, col_action = st.columns([2, 1, 1])
    
    with col_input:
        # User selects any transaction description from the dataset
        target_desc = st.selectbox("Select Description", df['Raw_Description'].unique())
    
    with col_cat:
        # Manual selection of the 'Ground Truth'
        manual_cat = st.selectbox("Correct Category", ["Food", "Shopping", "Transport", "Rent", "Utilities", "Others"])
        
    with col_action:
        st.write(" ") # Spacer for alignment
        if st.button("Train Model 🚀"):
            # This calls the predict_logic separated from the UI (Slide 20)
            ai.train_model(target_desc, manual_cat)
            st.success(f"AI updated: '{target_desc}' is now '{manual_cat}'")
            st.balloons()