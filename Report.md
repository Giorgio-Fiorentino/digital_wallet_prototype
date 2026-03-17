
              DIGITAL WALLET PROTOTYPE v2: MULTI-SOURCE AI DASHBOARD
                    Prototype Development Report — Assignment 2
                                Giorgio Fiorentino


1. QUESTIONS ADDRESSED IN THIS PROTOTYPE
-----------------------------------------
The second iteration was guided by two core design questions that directly
extended the v1 baseline:

QUESTION 1 — How do you implement multi-tool use and RAG in the same assistant?

The v1 prototype demonstrated that a TF-IDF classifier could categorize
transactions. The gap it left was interaction: there was no way for a user
to ask questions about their own data in natural language and get reliable,
computed answers. The v2 AI Assistant closes this gap through two complementary
mechanisms operating from a single chat interface.

Multi-tool use is implemented via a Cohere agentic loop (ClientV2 API,
command-r-08-2024). Six structured tools are exposed to the model —
get_transactions, get_spending_summary, get_monthly_comparison,
get_top_merchants, get_source_breakdown, and detect_anomalies. Each tool
is a typed Python function operating on a pandas DataFrame; the model
selects and calls tools, receives results, and only generates a final
natural-language answer when it has the real data in hand. This architecture
guarantees that every figure in the assistant's response is computed from
the actual dataset, not hallucinated.

Retrieval-Augmented Generation handles a different category of question:
card fees, foreign transaction charges, cashback rules, and coverage limits.
These live in policy documents, not transaction rows. A FAISS index is
built over four card-terms documents (Revolut, PayPal, Trade Republic,
Visa/Mastercard) using Cohere embeddings. A lightweight keyword router
inspects each query before any API call; if it matches terms like "fee",
"cashback", or "interest", the query goes to the RAG branch; otherwise
it goes to the tool-use branch. The user experiences one unified chat.

QUESTION 2 — What other user needs do these functions satisfy beyond chat?

The same infrastructure powers the analytics dashboard independently of
the chat: anomaly detection (Z-score > 2σ per spending category), monthly
trend charts, per-source breakdowns across five payment instruments, and
a model evaluation harness that compares TF-IDF against Cohere Embeddings
with precision, recall, and F1 per class. The human-in-the-loop training
interface surfaces low-confidence predictions for manual correction. These
features show that the underlying tool and embedding stack is general-purpose
— the chat assistant is one surface, not the only one.


2. MAIN DIFFICULTIES ENCOUNTERED
----------------------------------
DIFFICULTY 1 — Making the prototype reproducible and deployable

The largest operational gap between v1 and v2 was the distance between
"runs on my machine" and "deploys cleanly on Streamlit Cloud." Several
compounding issues had to be resolved in sequence:

- faiss-cpu requires the swig compiler on macOS but installs from pre-built
  wheels on Linux; the platform difference was invisible locally.
- The Kaggle fraud-detection dataset (1.85 M rows) cannot be bundled in the
  repo. A stratified-sampling pipeline was written to produce a 5,000-row
  processed CSV that can be committed to GitHub and loaded on the cloud
  without Kaggle credentials.
- The processed data directory had been added to .gitignore, causing the
  cloud app to fail silently with no transaction data to load.
- A corrupted virtual environment (invalid version string in a package's
  metadata) masked pip errors for hours until the environment was rebuilt.
- A real Cohere API key had been committed accidentally in .env.example
  and had to be rotated and replaced with a placeholder.

Each failure was invisible in development but fatal in deployment. The fix
was systematic: audit every runtime assumption, pin every dependency, and
test the full cold-start sequence on a clean environment before pushing.

DIFFICULTY 2 — Aligning user needs with the technical backend

Designing the tool schema required a translation exercise that went in both
directions. On one side: what does a user mean when they ask "Am I spending
more this month?" and what does that map to in a pandas query? The tool
descriptions had to be precise enough that the model would select the right
function and pass the right arguments, but not so restrictive that it could
not generalize to paraphrased questions. Several iterations were needed.

On the other side: the professor's feedback required reported performance
metrics, but surfacing precision/recall tables to an end user is noise.
The AI Lab tab resolves this tension by separating model comparison (a
technical view for evaluation purposes) from the main dashboard (designed
for a non-technical wallet user). Both audiences are served without
either being degraded.

A further alignment problem came from the Cohere SDK itself: the v5 ClientV2
API introduced breaking changes to finish-reason strings and message
serialization that were not documented in the official guides. Diagnosing
these required reading SDK source files directly — a reminder that LLM APIs
are still unstable surfaces requiring pinned versions and isolated tests.


3. HOW AI WAS LEVERAGED
------------------------
AI assistance was present at every layer of the project, but in clearly
distinct roles at different stages of development.

GENERATION AND SCAFFOLDING — The six-tool wallet engine, the FAISS RAG
pipeline, the Streamlit four-tab structure, and the HTML card components
(credit card gradients, KPI tiles, chat bubbles, anomaly alerts) were all
produced through AI-assisted code generation. This compressed the path from
concept to running prototype significantly, allowing focus to stay on
architecture and product decisions rather than boilerplate.

DEBUGGING AND SDK NAVIGATION — When the agentic loop produced only "Reached
maximum tool iterations" regardless of the query, AI assistance was used to
form hypotheses (wrong finish-reason strings, missing tool-call fields in
the message context) and locate the relevant SDK source files. What would
otherwise have been an opaque multi-hour debugging session became a
structured investigation with a clear fix.

DESIGN ITERATION — The visual theme was developed iteratively through
natural-language design prompts, with each round producing working CSS and
Streamlit components rather than mockups requiring separate translation.

LIMITS AND HUMAN OVERSIGHT — AI-generated code required review at every
integration point. Tool schemas needed manual tuning to align with the
model's selection behavior; the RAG routing keywords needed adjustment to
avoid misrouting transaction questions that happened to contain the word
"limit." The prototype works because the human decision layer caught and
corrected the places where generated code was technically correct but
behaviorally wrong. AI provided the velocity; judgment provided the
direction.


4. CONCLUSION
--------------
This prototype demonstrates that a language model with structured tools and
a document index can cover a meaningful portion of what a personal finance
app does — and deliver it through a conversational interface that requires
no knowledge of the underlying data schema. The remaining gap between
prototype and product is primarily one of data connectivity (live bank APIs
instead of a static CSV) and trust (auditability and error handling at
production scale). The architecture built here — tool dispatch, RAG routing,
human-in-the-loop correction, model evaluation — is the right shape for
addressing both, and provides a credible technical foundation for what an
intelligent layer inside Apple Wallet or Google Wallet could look like.
