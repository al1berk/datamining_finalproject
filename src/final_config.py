from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_CLUSTER_K = 46
ACCEPTED_KNEEDLE_K = 46
K_SELECTION_VALUES = tuple(sorted(set(range(2, 101, 2)).union({45})))
FINAL_K_COMPARISON_VALUES = (36, 40, 44, 45, 46, 48, 50)
MANUAL_K_CANDIDATES = (36, 40, 45)
SAMPLE_SIZE_SILHOUETTE = 2000

SAMPLE_ABSTRACT = (
    "This paper proposes a machine learning based method for detecting "
    "anomalies in network traffic using deep neural networks and "
    "feature selection techniques."
)

ALTERNATIVE_METHOD_RESULTS = (
    {
        "label": "BERTopic (TF-IDF + TruncatedSVD)",
        "backend": "tfidf_svd",
        "summary_path": PROJECT_ROOT
        / "bertopic_experiment"
        / "results"
        / "experiment_summary.json",
    },
    {
        "label": "BERTopic (sentence-transformer/all-MiniLM-L6-v2)",
        "backend": "sentence_transformer",
        "summary_path": PROJECT_ROOT
        / "bertopic_experiment"
        / "results"
        / "sentence_transformer"
        / "experiment_summary.json",
    },
)
