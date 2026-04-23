"""
models/rag_engine.py

RAG engine for financial document Q&A.
Covers Lecture 6: Cohere embeddings, FAISS vector store, chunking strategy,
search_document vs search_query input types.

Used for card terms, fees, benefits questions — not transaction data.
"""

import os
import numpy as np
import cohere
import faiss
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _normalize(arr: np.ndarray) -> None:
    """In-place L2 normalization. Works with any faiss version."""
    try:
        faiss.normalize_L2(arr)
    except AttributeError:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr /= np.maximum(norms, 1e-10)


class RAGEngine:

    DOCS_DIR      = "docs/card_terms"
    EMBED_MODEL   = "embed-english-v3.0"
    CHUNK_SIZE    = 150
    CHUNK_OVERLAP = 30
    TOP_K         = 3

    def __init__(self):
        self.client        = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        self.chunks        = []
        self.chunk_sources = []
        self.index         = None
        self._is_built     = False

    def build_index(self) -> None:
        docs_path = Path(self.DOCS_DIR)
        if not docs_path.exists():
            raise FileNotFoundError(
                f"Docs directory not found: {self.DOCS_DIR}"
            )

        raw_docs = []
        for filepath in sorted(docs_path.glob("*.txt")):
            text = filepath.read_text(encoding="utf-8")
            raw_docs.append((filepath.name, text))
            print(f"  Loaded: {filepath.name} ({len(text.split())} words)")

        self.chunks        = []
        self.chunk_sources = []

        for source_name, text in raw_docs:
            words = text.split()
            i = 0
            while i < len(words):
                chunk_text = " ".join(words[i: i + self.CHUNK_SIZE])
                self.chunks.append(chunk_text)
                self.chunk_sources.append(source_name)
                i += self.CHUNK_SIZE - self.CHUNK_OVERLAP

        print(f"  Total chunks: {len(self.chunks)}")

        response   = self.client.embed(
            texts=self.chunks,
            model=self.EMBED_MODEL,
            input_type="search_document",
        )
        embeddings = np.array(response.embeddings, dtype="float32")
        _normalize(embeddings)

        dim        = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self._is_built = True
        print(f"  FAISS index: {self.index.ntotal} vectors, dim={dim}")

    def retrieve(self, query: str) -> list:
        if not self._is_built:
            self.build_index()
        response  = self.client.embed(
            texts=[query],
            model=self.EMBED_MODEL,
            input_type="search_query",
        )
        query_vec = np.array(response.embeddings, dtype="float32")
        _normalize(query_vec)
        scores, indices = self.index.search(query_vec, self.TOP_K)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text":   self.chunks[idx],
                "source": self.chunk_sources[idx],
                "score":  float(score),
            })
        return results

    def answer(self, query: str) -> tuple:
        chunks = self.retrieve(query)
        if not chunks:
            return "I couldn't find relevant information in the card terms.", []

        context_blocks = []
        for i, chunk in enumerate(chunks, 1):
            label = chunk["source"].replace("_", " ").replace(".txt", "").title()
            context_blocks.append(f"[Source {i} — {label}]\n{chunk['text']}")
        context = "\n\n".join(context_blocks)

        prompt = f"""You are a financial assistant helping a user understand their card terms. Use ONLY the information in the sources below. Quote fees and limits exactly as stated. If the answer is not in the sources, say "I don't have that information in the card terms."

SOURCES:
{context}

QUESTION: {query}

Answer:"""

        response = self.client.chat(
            model="command-r-08-2024",
            message=prompt,
            temperature=0.1,
        )
        return response.text, chunks

    def is_document_question(self, query: str) -> bool:
        keywords = [
            "annual fee", "cashback", "reward", "interest", "apr",
            "foreign transaction", "travel insurance", "lounge", "benefit",
            "limit", "withdrawal", "fee", "finance charge", "terms", "condition",
            "coverage", "protection", "minimum payment", "late fee",
            "credit limit", "points", "miles", "membership", "plan",
        ]
        return any(kw in query.lower() for kw in keywords)
