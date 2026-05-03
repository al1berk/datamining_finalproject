import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import Normalizer
from umap import UMAP
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "bertopic_experiment" / "results"
DB_PATH = PROJECT_ROOT / "data" / "CompSciencePub.sqlite"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import load_data
from src.preprocessing import preprocess_dataframe

DOMAIN_STOP_WORDS = {
    "analysis",
    "approach",
    "approaches",
    "application",
    "applications",
    "article",
    "based",
    "computer",
    "data",
    "information",
    "method",
    "methods",
    "model",
    "models",
    "paper",
    "problem",
    "problems",
    "propose",
    "proposed",
    "research",
    "result",
    "results",
    "science",
    "study",
    "system",
    "systems",
    "use",
    "used",
    "using",
}


def build_embeddings(texts, n_components=100):
    vectorizer = TfidfVectorizer(
        stop_words=sorted(set(ENGLISH_STOP_WORDS).union(DOMAIN_STOP_WORDS)),
        max_features=15000,
        min_df=5,
        max_df=0.35,
        ngram_range=(1, 2),
    )
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    normalizer = Normalizer(copy=False)

    term_matrix = vectorizer.fit_transform(texts)
    embeddings = svd.fit_transform(term_matrix)
    embeddings = normalizer.fit_transform(embeddings)
    return embeddings


def build_sentence_transformer_embeddings(texts, model_name):
    model = SentenceTransformer(model_name, device="cpu")
    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def build_topic_model():
    stop_words = sorted(set(ENGLISH_STOP_WORDS).union(DOMAIN_STOP_WORDS))
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=150,
        min_samples=20,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
        core_dist_n_jobs=1,
    )
    vectorizer_model = CountVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.35,
    )

    return BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True,
        low_memory=True,
    )


def summarize_topics(topic_model, topics, docs_df):
    info = topic_model.get_topic_info().copy()
    topic_counts = pd.Series(topics).value_counts().sort_index()

    representative = topic_model.get_representative_docs()
    rows = []
    for _, row in info.iterrows():
        topic_id = int(row["Topic"])
        docs_in_topic = docs_df.loc[np.array(topics) == topic_id, "journal_name"].value_counts().head(5)
        rep_docs = representative.get(topic_id, [])[:2]
        rows.append(
            {
                "topic_id": topic_id,
                "count": int(row["Count"]),
                "name": row["Name"],
                "representation": ", ".join(row["Representation"]),
                "top_journals": "; ".join(f"{j} ({c})" for j, c in docs_in_topic.items()),
                "representative_doc_1": rep_docs[0] if len(rep_docs) > 0 else "",
                "representative_doc_2": rep_docs[1] if len(rep_docs) > 1 else "",
            }
        )
    return info, pd.DataFrame(rows), topic_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--embedding-backend",
        choices=("tfidf_svd", "sentence_transformer"),
        default="tfidf_svd",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    args = parser.parse_args()

    output_dir = RESULTS_DIR / args.embedding_backend
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(output_dir / "hf_cache"))

    df = load_data(str(DB_PATH))
    df = preprocess_dataframe(df)
    if args.sample_size:
        df = df.sample(args.sample_size, random_state=42).reset_index(drop=True)

    texts = df["clean_text"].tolist()
    if args.embedding_backend == "sentence_transformer":
        embeddings = build_sentence_transformer_embeddings(texts, args.model_name)
    else:
        embeddings = build_embeddings(texts, n_components=100)
    topic_model = build_topic_model()
    topics, _ = topic_model.fit_transform(texts, embeddings)

    info, summary_df, topic_counts = summarize_topics(topic_model, topics, df)

    n_topics = int(sum(1 for topic in set(topics) if topic != -1))
    outliers = int(np.sum(np.array(topics) == -1))
    summary = {
        "embedding_backend": args.embedding_backend,
        "model_name": args.model_name if args.embedding_backend == "sentence_transformer" else "tfidf_svd",
        "documents": int(len(df)),
        "assigned_topics_excluding_outliers": n_topics,
        "outlier_documents": outliers,
        "outlier_ratio": float(outliers / len(df)),
        "largest_topic_size": int(topic_counts.drop(labels=[-1], errors="ignore").max()),
        "smallest_topic_size": int(topic_counts.drop(labels=[-1], errors="ignore").min()),
    }

    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    info.to_csv(output_dir / "topic_info.csv", index=False)
    summary_df.to_csv(output_dir / "topic_summary.csv", index=False)

    print("SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nTOPICS")
    printable = summary_df[summary_df["topic_id"] != -1][["topic_id", "count", "representation", "top_journals"]]
    print(printable.to_string(index=False))


if __name__ == "__main__":
    main()
