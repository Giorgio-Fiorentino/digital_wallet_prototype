# 💳 Apple Wallet AI Extension: Multi-Card Intelligence

This repository contains a **functional AI prototype** designed as an extension for the Apple Wallet ecosystem. The application aggregates transaction data from multiple cards, uses **Natural Language Processing (NLP)** for automated categorization, and implements an **Active Learning** loop to refine model accuracy through user interaction.

---

## 🎯 Project Vision
Traditional digital wallets often act as simple storage for cards. This prototype explores a "Smart Wallet" concept where the wallet understands spending behavior across different accounts. It addresses the three pillars of an AI prototype:

* **Appearance & UX**: An Apple-inspired interface using custom `.streamlit/config.toml` styling and interactive Plotly visualizations.
* **Data/Model Pipeline**: A robust pipeline that filters raw CSV data by card, month, and specific date ranges before processing.
* **Accuracy & Human-in-the-Loop**: A dedicated "AI Training Center" that flags low-confidence predictions for manual verification.

---

## 🧠 AI & Technical Implementation

### 1. Categorization Engine
The core logic utilizes a **Cosine Similarity** model implemented via `scikit-learn`:
* **Vectorization**: Raw transaction descriptions are converted into numerical vectors using `TfidfVectorizer`.
* **Similarity Scoring**: New transactions are compared against a dynamic `knowledge_base` of known merchants (e.g., Amazon, Starbucks, Landlord).
* **Confidence Threshold**: A threshold of **0.75** is used to distinguish between "Confident" predictions and "Uncertain" ones.

### 2. Active Learning Flow
The prototype features a **Human-in-the-loop** feedback system:
* **Automated Review**: Low-confidence transactions are surfaced in the AI Training tab.
* **Manual Training**: Users can define new categories (e.g., "Rent" or "Utilities") and manually map transactions to them, simulating a model update.

---

## 📊 Key Features
* **Unified Financial Insights**: A central dashboard showing total expenses across all selected cards.
* **Dynamic Filtering**: Filter by specific cards and custom date ranges (defaulting to the current month).
* **Monthly Comparison**: Interactive line charts to compare spending trends over time.
* **Wallet Experience**: Integration of digital passes, such as a Ryanair boarding pass, to simulate a real-world wallet environment.

---

## 📁 Repository Structure
```text
digital_wallet_prototype/
├── .streamlit/
│   └── config.toml          # Global UI theme (Apple-style colors)
├── data/
│   └── transactions.csv     # Synthetic transaction dataset
├── models/
│   └── categorizer.py       # AI Categorization Logic & Training methods
├── app.py                   # Main Streamlit UI with Tabs and Graphs
├── requirements.txt         # Dependency manifest for Cloud deployment
└── README.md                # Project documentation