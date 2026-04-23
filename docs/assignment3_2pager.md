# Smart Wallet AI — v3 Report
### ESADE MIBA · Prototyping & Digital AI · Assignment 3
**Giorgio Fiorentino · April 2026**

---

## What the Prototype Does

Smart Wallet is an AI-powered personal finance dashboard that aggregates transactions across multiple bank accounts and payment instruments into a single intelligent interface — the same paradigm as Google Pay or Apple Wallet, but with a reasoning layer on top.

The core architecture separates concerns cleanly into three layers. **WalletEngine** is the data computation layer: it exposes six structured tools (spending summary, top merchants, anomaly detection, source breakdown, monthly comparison, budget status) that the LLM can invoke but never access directly. **WalletLLM** is a Cohere ClientV2 agentic loop that runs up to five tool-call iterations, inspects `finish_reason`, and synthesises a natural-language answer from structured tool results. **RAGEngine** handles a separate document-grounded path: card-terms questions are routed via keyword detection to a FAISS index built over four card-terms documents, with `embed-english-v3.0` embeddings and cosine retrieval, bypassing the tool-use loop entirely.

The result is a system that can answer both *"How much did I spend on food in February?"* (tool-use path, real-time over the dataset) and *"Does Revolut charge foreign transaction fees?"* (RAG path, grounded answer from card-terms) — using the right pattern for each query type, with the full reasoning trace visible to the user.

The AI Lab tab adds a model evaluation harness: TFIDFCategorizer (TF-IDF + cosine, threshold 0.20) and EmbeddingCategorizer (Cohere embeddings, threshold 0.30) are benchmarked with per-class precision, recall, and F1 against Kaggle ground-truth labels. Low-confidence predictions surface in a human-in-the-loop correction interface.

---

## v3 Improvements

### 1. Live Bank API Integration

The most substantial addition is `models/bank_connector.py` — a production-grade `TrueLayerAdapter` implementing the full TrueLayer open-banking OAuth2 flow. The adapter manages a two-token strategy (access + refresh): on any 401 response it automatically exchanges the refresh token for a new access token and retries the request, with zero manual re-authentication. `fetch_transactions()` queries all accounts, paginates transaction lists, deduplicates by (date, amount, description), normalises amounts, and classifies descriptions via TF-IDF — returning the identical DataFrame schema as the Kaggle CSV so the rest of the system needs no changes.

This is a real HTTPS connection to a regulated open-banking sandbox. All credentials are loaded from environment variables; the app degrades gracefully to CSV-only mode when credentials are absent.

### 2. Multi-Account Wallet (3 Simultaneous Sources)

v2 forced a binary choice: CSV *or* live data. v3 eliminates this constraint. When TrueLayer is active, `app.py` merges the live bank account with two Kaggle CSV accounts (Revolut, Visa Classic) and injects the combined DataFrame into WalletEngine. Each row carries a `Source_Tag` (`"csv"` or `"live"`) added at merge time — the engine itself remains source-agnostic.

The user experience reflects this: the sidebar multiselect shows all three accounts simultaneously, a **🔴 Live** badge appears on the TrueLayer card in the Payment Methods panel, and the sidebar reports "✓ 1 live account · 2 CSV accounts." Every downstream feature — AI queries, Smart Insights charts, anomaly detection — operates over the full merged dataset. This is the prototype's core value proposition made real.

### 3. AI Confidence Scoring

Every tool call in the AI Assistant tab now carries a colour-coded confidence badge alongside the existing trace expander. The heuristic is: **High** (green) when the model reaches `COMPLETE` in a single tool call; **Medium** (amber) when multiple iterations are needed; **Low** (red) when no tool is called and the model answers from priors alone. This adds a quantitative signal to the tool-use transparency the professor praised in v2.

### 4. AI Strategist Feedback Loop

After each strategy output, thumbs up / thumbs down buttons let the user rate the response. Ratings accumulate in session state with a `strategy_id` and timestamp; a running "% of strategies rated helpful" metric is displayed in real time. This directly implements the professor's suggestion to validate the Strategist over time and closes the human-AI feedback loop.

---

## Test-Driven Development

v3 was built test-first. The `tests/` directory contains **95 automated pytest tests** covering every public method of WalletEngine and TrueLayerAdapter — including edge cases (empty accounts, duplicate transactions, 401 token refresh, merged-source DataFrames, budget status thresholds). All network calls are mocked; the full suite runs in under five seconds. No feature was shipped before its tests passed. This is a material shift from v2, where correctness was verified manually — TDD caught several subtle bugs (regex format mismatch in spending totals, `KeyError` on missing `Source_Type` column in injected data, overbroad RAG routing keyword) before they could reach the app.

---

*Smart Wallet · v3 · Cohere command-r-plus · TrueLayer open-banking · FAISS RAG · 95 tests · Streamlit Cloud*
