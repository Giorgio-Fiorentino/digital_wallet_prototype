"""
models/categorizer.py

Two categorisation approaches:
    1. TF-IDF + Cosine Similarity  (Assignment 1 baseline)
    2. Cohere Embeddings + Cosine Similarity  (Assignment 2 upgrade)

Knowledge base built dynamically from dataset categories at runtime.
"""

import os
import numpy as np
import cohere
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# Seed knowledge base — extended at runtime from actual dataset categories
KNOWLEDGE_BASE = {
    "restaurant": "Food & Dining", "cafe": "Food & Dining",
    "coffee": "Food & Dining", "pizza": "Food & Dining",
    "burger": "Food & Dining", "food delivery": "Food & Dining",
    "fast food": "Food & Dining", "bar": "Food & Dining",
    "grocery": "Grocery", "supermarket": "Grocery",
    "market": "Grocery", "whole foods": "Grocery",
    "amazon": "Shopping", "shop": "Shopping",
    "store": "Shopping", "retail": "Shopping",
    "clothing": "Shopping", "electronics": "Shopping",
    "uber": "Gas & Transport", "lyft": "Gas & Transport",
    "taxi": "Gas & Transport", "gas station": "Gas & Transport",
    "airline": "Travel", "train": "Travel",
    "netflix": "Entertainment", "spotify": "Entertainment",
    "cinema": "Entertainment", "movie": "Entertainment",
    "gaming": "Entertainment", "streaming": "Entertainment",
    "pharmacy": "Health & Fitness", "gym": "Health & Fitness",
    "fitness": "Health & Fitness", "hospital": "Health & Fitness",
    "clinic": "Health & Fitness", "doctor": "Health & Fitness",
    "home depot": "Home", "ikea": "Home",
    "furniture": "Home", "hardware": "Home",
    "insurance": "Misc", "bank": "Misc", "atm": "Misc",
}

TFIDF_THRESHOLD     = 0.20
EMBEDDING_THRESHOLD = 0.30


class TFIDFCategorizer:
    def __init__(self):
        self.knowledge_base = KNOWLEDGE_BASE.copy()
        self.vectorizer     = TfidfVectorizer(ngram_range=(1, 2))

    def predict(self, description: str) -> tuple:
        known_descs  = list(self.knowledge_base.keys())
        known_cats   = list(self.knowledge_base.values())
        all_texts    = known_descs + [description.lower()]
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        similarity   = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        max_sim      = float(similarity.max())
        best_cat     = known_cats[similarity.argmax()]
        if max_sim > TFIDF_THRESHOLD:
            return best_cat, max_sim, False
        return best_cat, max_sim, True

    def train(self, description: str, category: str) -> None:
        self.knowledge_base[description.lower()] = category


class EmbeddingCategorizer:
    def __init__(self):
        self.client         = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        self.knowledge_base = KNOWLEDGE_BASE.copy()
        self._kb_embeddings = None
        self._kb_texts      = []
        self._kb_categories = []

    def _build_kb_embeddings(self) -> None:
        self._kb_texts      = list(self.knowledge_base.keys())
        self._kb_categories = list(self.knowledge_base.values())
        response = self.client.embed(
            texts=self._kb_texts,
            model="embed-english-v3.0",
            input_type="search_document",
        )
        self._kb_embeddings = np.array(response.embeddings, dtype="float32")

    def predict(self, description: str) -> tuple:
        if self._kb_embeddings is None:
            self._build_kb_embeddings()
        response  = self.client.embed(
            texts=[description.lower()],
            model="embed-english-v3.0",
            input_type="search_query",
        )
        query_vec    = np.array(response.embeddings, dtype="float32")
        similarities = cosine_similarity(query_vec, self._kb_embeddings)[0]
        max_idx      = int(similarities.argmax())
        max_sim      = float(similarities[max_idx])
        best_cat     = self._kb_categories[max_idx]
        if max_sim > EMBEDDING_THRESHOLD:
            return best_cat, max_sim, False
        return best_cat, max_sim, True

    def train(self, description: str, category: str) -> None:
        self.knowledge_base[description.lower()] = category
        self._kb_embeddings = None
