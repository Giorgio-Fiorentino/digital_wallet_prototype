import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TransactionAI:
    def __init__(self):
        # Database espanso per coprire più categorie
        self.knowledge_base = {
            "AMZN MKTPLACE": "Shopping",
            "STARBUCKS COFFEE": "Food",
            "UBER TRIP": "Transport",
            "ESSELUNGA MILANO": "Groceries",
            "NETFLIX.COM": "Subscriptions",
            "SHELL REFUEL": "Transport",
            "LANDLORD RENT": "Rent",
            "ENEL ENERGY": "Utilities"
        }
        self.vectorizer = TfidfVectorizer()

    def predict_category(self, description):
        known_desc = list(self.knowledge_base.keys())
        known_cats = list(self.knowledge_base.values())
        
        all_texts = known_desc + [description]
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # Similarity calculus
        similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        max_sim = similarity.max()
        
        # FEEDBACK LOOP
        if max_sim > 0.75: # Soglia di confidenza alta
            return known_cats[similarity.argmax()], max_sim, False
        else:
            return "Others", max_sim, True

    # NUOVA FEATURE: Analisi Budget (Prediction)
    def predict_monthly_burn(self, df):
        """Semplice modello predittivo: calcola la media e prevede la spesa fine mese"""
        if df.empty: return 0.0
        daily_avg = df['Amount'].sum() / 30 # Semplificazione per prototipo
        return round(daily_avg * 30, 2)
    
    def train_model(self, description, category):
        self.knowledge_base[description.upper()] = category