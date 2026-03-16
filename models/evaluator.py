"""
models/evaluator.py

Precision, recall, F1 for both categorisers against Kaggle ground truth.
Addresses professor feedback: "performance metrics not reported."
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from models.categorizer import TFIDFCategorizer, EmbeddingCategorizer


def evaluate_categorizer(
    categorizer,
    df: pd.DataFrame,
    sample_size: int = 200,
) -> dict:
    eval_df = (
        df.groupby("Category", group_keys=False)
        .apply(lambda x: x.sample(
            min(len(x), max(1, int(sample_size * len(x) / len(df)))),
            random_state=42
        ))
        .reset_index(drop=True)
    )

    true_labels, pred_labels, confidences = [], [], []
    for _, row in eval_df.iterrows():
        pred_cat, confidence, _ = categorizer.predict(row["Raw_Description"])
        true_labels.append(row["Category"])
        pred_labels.append(pred_cat)
        confidences.append(confidence)

    labels    = sorted(set(true_labels))
    precision = precision_score(
        true_labels, pred_labels, labels=labels, average=None, zero_division=0
    )
    recall    = recall_score(
        true_labels, pred_labels, labels=labels, average=None, zero_division=0
    )
    f1        = f1_score(
        true_labels, pred_labels, labels=labels, average=None, zero_division=0
    )

    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": round(float(precision[i]), 3),
            "recall":    round(float(recall[i]), 3),
            "f1":        round(float(f1[i]), 3),
        }

    return {
        "accuracy":        round(
            float(np.mean([t == p for t, p in zip(true_labels, pred_labels)])), 3
        ),
        "avg_confidence":  round(float(np.mean(confidences)), 3),
        "macro_precision": round(float(np.mean(precision)), 3),
        "macro_recall":    round(float(np.mean(recall)), 3),
        "macro_f1":        round(float(np.mean(f1)), 3),
        "n_evaluated":     len(eval_df),
        "per_class":       per_class,
        "true_labels":     true_labels,
        "pred_labels":     pred_labels,
    }


def compare_models(df: pd.DataFrame, sample_size: int = 100) -> dict:
    print("Evaluating TF-IDF...")
    tfidf_results = evaluate_categorizer(TFIDFCategorizer(), df, sample_size)
    print("Evaluating Embeddings...")
    embed_results = evaluate_categorizer(EmbeddingCategorizer(), df, sample_size)
    return {
        "tfidf":       tfidf_results,
        "embedding":   embed_results,
        "improvement": {
            "accuracy": round(
                embed_results["accuracy"] - tfidf_results["accuracy"], 3
            ),
            "macro_f1": round(
                embed_results["macro_f1"] - tfidf_results["macro_f1"], 3
            ),
        },
    }


def get_metrics_dataframe(results: dict) -> pd.DataFrame:
    rows = []
    for category, metrics in results["per_class"].items():
        rows.append({
            "Category":  category,
            "Precision": metrics["precision"],
            "Recall":    metrics["recall"],
            "F1 Score":  metrics["f1"],
        })
    return pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
