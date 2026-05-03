import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.clustering import TopicClusterer
from src.database import load_data
from src.final_config import (
    ACCEPTED_KNEEDLE_K,
    ALTERNATIVE_METHOD_RESULTS,
    FINAL_CLUSTER_K,
    FINAL_K_COMPARISON_VALUES,
    K_SELECTION_VALUES,
    MANUAL_K_CANDIDATES,
    SAMPLE_ABSTRACT,
    SAMPLE_SIZE_SILHOUETTE,
)
from src.preprocessing import preprocess_dataframe
from src.recommender import JournalRecommender


OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"


def resolve_database_path():
    candidates = [
        PROJECT_ROOT / "data" / "CompSciencePub.sqlite",
        PROJECT_ROOT / "CompSciencePub.sqlite",
        PROJECT_ROOT.parent / "CompSciencePub.sqlite",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("CompSciencePub.sqlite could not be located.")


def import_knee_locator():
    try:
        from kneed import KneeLocator
        return KneeLocator
    except ModuleNotFoundError:
        pass

    for site_packages in (PROJECT_ROOT / "bertopic_experiment" / ".venv" / "lib").glob("python*/site-packages"):
        if str(site_packages) not in sys.path:
            sys.path.insert(0, str(site_packages))
        try:
            from kneed import KneeLocator
            return KneeLocator
        except ModuleNotFoundError:
            continue
    return None


def resolve_kneedle_k(results_df):
    knee_locator = import_knee_locator()
    if knee_locator is None:
        return ACCEPTED_KNEEDLE_K, False

    knee = knee_locator(
        results_df["k"].tolist(),
        results_df["inertia"].tolist(),
        curve="convex",
        direction="decreasing",
    ).knee
    return int(knee) if knee is not None else ACCEPTED_KNEEDLE_K, knee is not None


def plot_metric(x, y, title, ylabel, save_path, selected_k=None, selected_label=None):
    plt.figure(figsize=(9, 5))
    plt.plot(x, y, marker="o", linewidth=1.8, markersize=4)
    if selected_k is not None:
        plt.axvline(
            selected_k,
            linestyle="--",
            linewidth=1.5,
            color="tab:red",
            label=selected_label or f"Selected K={selected_k}",
        )
        plt.legend()
    plt.xlabel("Number of Clusters K")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def load_alternative_method_summary():
    rows = []
    for config in ALTERNATIVE_METHOD_RESULTS:
        summary_path = config["summary_path"]
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        rows.append(
            {
                "method": config["label"],
                "backend": config["backend"],
                "documents": int(summary["documents"]),
                "assigned_topics_excluding_outliers": int(summary["assigned_topics_excluding_outliers"]),
                "outlier_documents": int(summary["outlier_documents"]),
                "outlier_ratio_pct": float(summary["outlier_ratio"]) * 100,
                "largest_topic_size": int(summary["largest_topic_size"]),
                "smallest_topic_size": int(summary["smallest_topic_size"]),
            }
        )
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)

    print("=== Data Mining Final Project: Journal Recommendation System ===")

    db_path = resolve_database_path()
    print(f"Using database: {db_path}")

    print("\n--- Phase 1: Data Loading ---")
    df = load_data(str(db_path))
    print(f"Loaded {len(df)} articles.")

    print("\n--- Phase 2: Preprocessing ---")
    df = preprocess_dataframe(df)

    print("\n--- Phase 3: Exploratory Data Analysis ---")
    plt.figure(figsize=(10, 6))
    df["journal_name"].value_counts().head(10).plot(kind="bar")
    plt.title("Top 10 Journals by Number of Articles")
    plt.ylabel("Number of Articles")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "top_10_journals.png")
    plt.close()

    df["abstract_word_count"] = df["abstract"].apply(lambda value: len(str(value).split()))
    plt.figure(figsize=(10, 6))
    df["abstract_word_count"].plot(kind="hist", bins=50, color="skyblue", edgecolor="black")
    plt.title("Distribution of Abstract Lengths (Words)")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "abstract_length_dist.png")
    plt.close()

    dataset_summary = pd.DataFrame(
        [
            {
                "articles": len(df),
                "journals": df["journal_name"].nunique(),
                "mean_abstract_words": df["abstract_word_count"].mean(),
                "median_abstract_words": df["abstract_word_count"].median(),
                "missing_author_keywords_pct": df["author_keywords"].eq("").mean() * 100,
                "missing_subjects_pct": df["subjects"].eq("").mean() * 100,
                "missing_keyword_plus_pct": df["keyword_plus"].eq("").mean() * 100,
            }
        ]
    )
    dataset_summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)

    print("\n--- Phase 4: Journal Recommendation ---")
    recommender = JournalRecommender()
    recommender.fit(df)

    recs = recommender.recommend_journals(SAMPLE_ABSTRACT, top_n=5)
    print("\nTop 5 Recommendations:")
    print(recs[["journal_name", "score", "max_similarity", "avg_similarity"]])
    recs.to_csv(OUTPUT_DIR / "top_journals_sample.csv", index=False)

    print("\nRunning hold-out evaluation...")
    metrics = recommender.evaluate(
        df=df,
        test_size=0.2,
        min_journal_articles=5,
        random_state=42,
    )
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(OUTPUT_DIR / "recommendation_metrics.csv", index=False)
    print(metrics_df.to_string(index=False))

    print("\n--- Phase 5: Topic Clustering and K Selection ---")
    probe_clusterer = TopicClusterer(random_state=42)
    k_selection_results = probe_clusterer.evaluate_k_values(
        df,
        text_column="clean_text",
        k_values=K_SELECTION_VALUES,
        sample_size=SAMPLE_SIZE_SILHOUETTE,
    )
    k_selection_results.to_csv(OUTPUT_DIR / "k_selection_results.csv", index=False)

    shortlist = (
        k_selection_results[k_selection_results["k"].isin(FINAL_K_COMPARISON_VALUES)]
        .sort_values("k")
        .reset_index(drop=True)
    )
    shortlist.to_csv(OUTPUT_DIR / "final_k_comparison.csv", index=False)
    print(shortlist.to_string(index=False))

    kneedle_k, used_true_kneedle = resolve_kneedle_k(k_selection_results)
    best_silhouette_k = int(k_selection_results.loc[k_selection_results["silhouette_score"].idxmax(), "k"])
    best_db_k = int(k_selection_results.loc[k_selection_results["davies_bouldin_score"].idxmin(), "k"])
    best_ch_k = int(k_selection_results.loc[k_selection_results["calinski_harabasz_score"].idxmax(), "k"])

    plot_metric(
        k_selection_results["k"],
        k_selection_results["inertia"],
        "Elbow Method with Accepted Final K",
        "Inertia",
        FIGURE_DIR / "kneedle_elbow.png",
        selected_k=FINAL_CLUSTER_K,
        selected_label=f"Accepted elbow / final K={FINAL_CLUSTER_K}",
    )
    plot_metric(
        k_selection_results["k"],
        k_selection_results["silhouette_score"],
        "Silhouette Score by K",
        "Silhouette Score",
        FIGURE_DIR / "silhouette_scores.png",
        selected_k=best_silhouette_k,
        selected_label=f"Best silhouette K={best_silhouette_k}",
    )
    plot_metric(
        k_selection_results["k"],
        k_selection_results["davies_bouldin_score"],
        "Davies-Bouldin Score by K",
        "Davies-Bouldin Score",
        FIGURE_DIR / "davies_bouldin_scores.png",
        selected_k=best_db_k,
        selected_label=f"Best DB K={best_db_k}",
    )
    plot_metric(
        k_selection_results["k"],
        k_selection_results["calinski_harabasz_score"],
        "Calinski-Harabasz Score by K",
        "Calinski-Harabasz Score",
        FIGURE_DIR / "calinski_harabasz_scores.png",
        selected_k=best_ch_k,
        selected_label=f"Best CH K={best_ch_k}",
    )

    manual_candidates = (
        k_selection_results[k_selection_results["k"].isin(MANUAL_K_CANDIDATES)]
        .sort_values("k")
        .reset_index(drop=True)
    )
    print(
        "\nAccepted final K selection summary:\n"
        f"- Kneedle / elbow candidate: K={kneedle_k}"
        f"{'' if used_true_kneedle else ' (accepted fallback because kneed is unavailable)'}\n"
        f"- Best silhouette: K={best_silhouette_k}\n"
        f"- Best Davies-Bouldin: K={best_db_k}\n"
        f"- Best Calinski-Harabasz: K={best_ch_k}\n"
        f"- Final accepted K: K={FINAL_CLUSTER_K}"
    )
    print("\nManual candidate comparison:")
    print(manual_candidates.to_string(index=False))

    clusterer = TopicClusterer(n_clusters=FINAL_CLUSTER_K, random_state=42)
    clustered_df = clusterer.fit_predict(df, text_column="clean_text")
    cluster_summary = clusterer.get_cluster_summary(clustered_df, top_n=10)
    cluster_summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)

    print("\nTop Terms per Cluster:")
    clusterer.print_top_terms(top_n=10)

    _, explained_variance = clusterer.visualize_clusters(
        clustered_df,
        save_path=FIGURE_DIR / "cluster_visualization.png",
        show=False,
    )
    print(f"SVD explained variance (2 components): {explained_variance:.4f}")

    clustered_df[["article_id", "journal_name", "cluster"]].to_csv(
        OUTPUT_DIR / "cluster_results.csv",
        index=False,
    )

    plt.figure(figsize=(10, 6))
    clustered_df["cluster"].value_counts().sort_index().plot(
        kind="bar",
        color="coral",
        edgecolor="black",
    )
    plt.title(f"Distribution of Articles across {FINAL_CLUSTER_K} Clusters")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Articles")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "cluster_distribution.png")
    plt.close()

    print("\n--- Phase 6: Alternative Method Summary ---")
    alternative_summary = load_alternative_method_summary()
    if alternative_summary.empty:
        print("No BERTopic experiment summaries were found.")
    else:
        alternative_summary.to_csv(OUTPUT_DIR / "alternative_method_summary.csv", index=False)
        print(alternative_summary.to_string(index=False))

    print("\n=== Pipeline execution complete! ===")


if __name__ == "__main__":
    main()
