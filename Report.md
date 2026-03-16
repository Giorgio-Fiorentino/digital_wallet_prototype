               
                    DIGITAL WALLET PROTOTYPE: MULTI-CARD AI EXTENSION
                              Project Documentation Report
                                  Giorgio Fiorentino

Link to the github repo: [link](https://github.com/Giorgio-Fiorentino/digital_wallet_prototype)

Link to the app deployed on streamlit cloud: [link](https://digitalwalletprototype-26r8y6nikossaxzzff47if.streamlit.app/)


1. EXECUTIVE SUMMARY & CONCEPTUAL ORIGIN
----------------------------------------
The Digital Wallet Prototype stems from a personal observation of a 
functional gap within the current Fintech ecosystem. While individual 
neobanking applications (e.g., Revolut, Trade Republic) offer high-level 
transaction categorization and spending analysis, centralized digital 
wallets like Apple Wallet and Google Wallet which aggregate multiple 
cards—do not yet provide a unified, state of the art analytical layer.

This project proposes an "Intelligence Extension" for multi-card 
digital wallets. It provides users with a single, synchronized interface 
to analyze spending habits across different financial institutions, 
bridging the gap between payment convenience and financial oversight.

2. DEVELOPMENT METHODOLOGY & PROCESS
------------------------------------
Before actually starting the process, 
github and streamlit galleries have been checked to find similar projects
to take inspiration from, but any of them was applicable to this case.
The development followed an agile, iterative "Look and Feel" 
methodology, characterized by a tight feedback loop between local 
development and cloud deployment.

* Environment & Reproducibility: To ensure high engineering standards, 
  the project was initialized within a virtual environment (wallet_venv) with 
  a clear repository structure. This was synchronized with a GitHub 
  repository to facilitate continuous deployment via Streamlit Cloud.

* Data Strategy: A custom data generation pipeline (generate_data.py) 
  was developed to create a synthetic but representative dataset. This 
  ensured that the prototype could be tested against diverse transaction 
  types, including complex cases like rent payments and low-confidence 
  merchant descriptions.

* Iterative UI/UX Design: Utilizing the Streamlit framework, the 
  interface was refined through multiple iterations. This allowed for 
  the translation of conceptual visual ideas into functional widgets 
  (st.tabs, st.metric, st.plotly_chart), focusing on an Apple inspired 
  minimalist aesthetic defined in the global .streamlit/config.toml.

3. AI ARCHITECTURE & DATA PIPELINE
----------------------------------
While the front-end remains the primary focus, the back-end integrates 
a functional machine learning pipeline to demonstrate "Active Learning" 
capabilities:

* Natural Language Processing (NLP): The system utilizes Scikit-Learn’s 
  TfidfVectorizer and Cosine Similarity to classify raw transaction 
  strings into categories (Food, Shopping, Rent, etc...).
  
* Active Learning Feedback Loop: A core feature of the prototype is 
  the "AI Training / Augment your wallet". When the model encounters low-confidence 
  similarity scores (threshold < 0.75), it flags the transaction for 
  user review. This "Human-in-the-Loop" design allows the user to 
  manually train the model, simulating a system that evolves and 
  improves accuracy with use.

* Data Pipeline Management: The app implements robust filtering 
  logic that dynamically handles empty data states and resets date 
  ranges based on the user's selected month, preventing common 
  runtime errors in the data pipeline.

4. TOOLS & COLLABORATIVE AI DEVELOPMENT
---------------------------------------
The prototype was built using a modern Python stack:
- Frontend: Streamlit
- Data Manipulation: Pandas
- Machine Learning: Scikit-Learn
- Visualizations: Plotly Express
- Version Control: Git/GitHub

AI assistance played a crucial role in the development phase, acting 
as a "Senior Cloud Architect" peer. The AI helped translate complex 
UI requirements into clean code, optimized the logic for state 
management between widgets, and provided troubleshooting for cloud 
deployment hurdles (such as TOML configuration and hidden folder 
management).

5. CONCLUSION
-------------
This project successfully demonstrates that a unified spending analysis 
feature is not only technically feasible but adds significant value 
to the multi-card wallet experience. Furthermore it gave me the opportunity 
to experience the high potential of the use of this cloud platform for creating,
developing and deploying ideas in hours!

