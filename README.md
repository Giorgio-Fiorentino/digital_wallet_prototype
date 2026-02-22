💳 Apple Wallet AI Extension: Multi-Card Intelligence
This repository contains a functional AI prototype designed as an extension for the Apple Wallet ecosystem. The application aggregates transaction data from multiple cards, uses Natural Language Processing (NLP) for automated categorization, and implements an Active Learning loop to refine model accuracy through user interaction.

🚀 Live Demo
Access the deployed application here: [INSERT YOUR STREAMLIT URL HERE]

🎯 Project Vision
Traditional digital wallets often act as simple storage for cards. This prototype explores a "Smart Wallet" concept where the wallet understands spending behavior across different accounts. It addresses the three pillars of an AI prototype:

Appearance & UX: An Apple-inspired interface using custom .streamlit/config.toml styling and interactive Plotly visualizations.

Data/Model Pipeline: A robust pipeline that filters raw CSV data by card, month, and specific date ranges before processing.

Accuracy & Human-in-the-Loop: A dedicated "AI Training Center" that flags low-confidence predictions for manual verification.

🧠 AI & Technical Implementation
1. Categorization Engine
The core logic utilizes a Cosine Similarity model implemented via Scikit-Learn:

Vectorization: Raw transaction descriptions are converted into numerical vectors using TfidfVectorizer.

Similarity Scoring: New transactions are compared against a dynamic knowledge_base of known merchants (e.g., Amazon, Starbucks, Landlord).

Confidence Threshold: A threshold of 0.75 is used to distinguish between "Confident" predictions and "Uncertain" ones (categorized as Others).

2. Active Learning Flow
The prototype features a Feedback Loop:

Automated Review: Low-confidence transactions are surfaced in the AI Training tab.

Manual Training: Users can define new categories (e.g., "Rent" or "Utilities") and manually map transactions to them, simulating a model update.

📁 Repository Structure
Plaintext

digital_wallet_prototype/
├── .streamlit/
│   └── config.toml          # Global UI theme (Apple-style colors)
├── data/
│   └── transactions.csv     # Synthetic transaction dataset (Rent, Food, etc.)
├── models/
│   └── categorizer.py       # AI Categorization Logic & Training methods
├── app.py                   # Main Streamlit UI with Tabs and Graphs
├── requirements.txt         # Dependency manifest for Cloud deployment
└── README.md                # Project documentation
💻 Local Setup
To run this project locally on macOS:

Clone the repository:

Bash

git clone <your-repo-url>
cd digital_wallet_prototype
Create a Virtual Environment:

Bash

python3 -m venv wallet_venv
source wallet_venv/bin/activate
Install Dependencies:

Bash

pip install -r requirements.txt
Launch the App:

Bash

streamlit run app.py
🛠️ Advanced Features & Widgets Used
st.tabs: For clean navigation between Wallet, Home Analysis, and AI Training.

st.plotly_chart: Interactive line and pie charts for monthly spending comparisons.

st.data_editor: Allowing users to interact directly with the AI's data labels.

st.metric: Real-time KPI tracking for total spent and budget forecasts.

st.date_input: Dynamic date range selection linked to a month-selector state.