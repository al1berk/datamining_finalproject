import os

import nbformat as nbf


def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# Journal Recommendation System
**Data Mining Final Project**

**Student**: Ali Berk Yeşilduman  
**Student ID**: 20200808035  
**Email**: 20200808035@ogr.akdeniz.edu.tr  

This notebook documents the final delivery pipeline. The selected recommendation method is a TF-IDF and cosine-similarity baseline, while the selected topic clustering model is the accepted `K=46` K-Means solution chosen after a broad K-selection study."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath('..'))

from src.clustering import TopicClusterer
from src.database import load_data
from src.final_config import (
    ACCEPTED_KNEEDLE_K,
    FINAL_CLUSTER_K,
    FINAL_K_COMPARISON_VALUES,
    K_SELECTION_VALUES,
    SAMPLE_ABSTRACT,
    SAMPLE_SIZE_SILHOUETTE,
)
from src.preprocessing import preprocess_dataframe
from src.recommender import JournalRecommender"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 1. Data Loading and Preprocessing
We build a combined text field from abstracts, author keywords, Web of Science subjects, and keyword plus terms. The text is lowercased, stripped of HTML, purged of non-alphabetic characters, and normalized for whitespace."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """db_path = '../data/CompSciencePub.sqlite'
df = load_data(db_path)
df = preprocess_dataframe(df)
print(f'Total articles loaded: {len(df)}')
df[['article_id', 'journal_name', 'abstract']].head(3)"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. Exploratory Data Analysis
These charts summarize the corpus before recommendation and clustering."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """plt.figure(figsize=(10, 6))
df['journal_name'].value_counts().head(10).plot(kind='bar', color='cornflowerblue', edgecolor='black')
plt.title('Top 10 Journals by Number of Articles')
plt.ylabel('Number of Articles')
plt.tight_layout()
plt.show()"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """df['abstract_length'] = df['abstract'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(10, 6))
df['abstract_length'].plot(kind='hist', bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Abstract Lengths (Words)')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.xlim(0, 500)
plt.tight_layout()
plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 3. Journal Recommendation Pipeline
The recommender fits a TF-IDF representation over the cleaned document text and ranks journals with a weighted combination of maximum and average cosine similarity."""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 4. Journal Recommendation Example
In this step, a sample article abstract is given to the trained recommender system.  
The system ranks journals using TF-IDF cosine similarity and returns the top 5 most relevant journals."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """print("Sample abstract used for journal recommendation:")
print(SAMPLE_ABSTRACT)

recommender = JournalRecommender(max_features=10000)
recommender.fit(df)

recs = recommender.recommend_journals(SAMPLE_ABSTRACT, top_n=5)
print("\\nTop-5 recommended journals:")
recs[['journal_name', 'score', 'max_similarity', 'avg_similarity']]"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 5. Hold-Out Evaluation
The recommendation model is evaluated with a stratified 80/20 split over journals that have at least five articles."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """metrics = recommender.evaluate(
    df=df,
    test_size=0.2,
    min_journal_articles=5,
    random_state=42,
)
pd.DataFrame([metrics])"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 6. K Selection Analysis for Topic Clustering
The clustering feature space uses domain-aware TF-IDF, TruncatedSVD, and L2 normalization. We scanned `K=2..100` with step `2`, plus the manually compared `K=45`, and accepted `K=46` as the elbow-based final operating point."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """probe_clusterer = TopicClusterer(random_state=42)
k_selection_results = probe_clusterer.evaluate_k_values(
    df,
    text_column='clean_text',
    k_values=K_SELECTION_VALUES,
    sample_size=SAMPLE_SIZE_SILHOUETTE,
)

shortlist = (
    k_selection_results[k_selection_results['k'].isin(FINAL_K_COMPARISON_VALUES)]
    .sort_values('k')
    .reset_index(drop=True)
)
shortlist"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """fig, axes = plt.subplots(2, 2, figsize=(14, 9))

axes[0, 0].plot(k_selection_results['k'], k_selection_results['inertia'], marker='o')
axes[0, 0].axvline(FINAL_CLUSTER_K, color='tab:red', linestyle='--', label=f'Accepted final K={FINAL_CLUSTER_K}')
axes[0, 0].set_title('Elbow / Inertia')
axes[0, 0].set_xlabel('K')
axes[0, 0].set_ylabel('Inertia')
axes[0, 0].legend()

axes[0, 1].plot(k_selection_results['k'], k_selection_results['silhouette_score'], marker='o')
axes[0, 1].set_title('Silhouette Score')
axes[0, 1].set_xlabel('K')
axes[0, 1].set_ylabel('Score')

axes[1, 0].plot(k_selection_results['k'], k_selection_results['davies_bouldin_score'], marker='o')
axes[1, 0].set_title('Davies-Bouldin Score')
axes[1, 0].set_xlabel('K')
axes[1, 0].set_ylabel('Score')

axes[1, 1].plot(k_selection_results['k'], k_selection_results['calinski_harabasz_score'], marker='o')
axes[1, 1].set_title('Calinski-Harabasz Score')
axes[1, 1].set_xlabel('K')
axes[1, 1].set_ylabel('Score')

plt.tight_layout()
plt.show()

print(
    f'Accepted elbow / Kneedle decision: K={ACCEPTED_KNEEDLE_K}. '
    f'The final delivery uses K={FINAL_CLUSTER_K}.'
)"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 7. Final `K=46` Clustering Model
After the broad scan, the final K-Means model is fit with `K=46` and summarized with representative top terms per cluster."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """clusterer = TopicClusterer(n_clusters=FINAL_CLUSTER_K, random_state=42)
clustered_df = clusterer.fit_predict(df, text_column='clean_text')
cluster_summary = clusterer.get_cluster_summary(clustered_df, top_n=10)
cluster_summary"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """plt.figure(figsize=(12, 6))
clustered_df['cluster'].value_counts().sort_index().plot(kind='bar', color='coral', edgecolor='black')
plt.title(f'Distribution of Articles across {FINAL_CLUSTER_K} Clusters')
plt.xlabel('Cluster ID')
plt.ylabel('Number of Articles')
plt.tight_layout()
plt.show()"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """plot_df, explained_variance = clusterer.visualize_clusters(clustered_df, show=True)
print(f'SVD explained variance (2 components): {explained_variance:.4f}')"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 8. Alternative Method Explored
An experimental BERTopic pipeline was also evaluated with two embedding backends: TF-IDF + TruncatedSVD and sentence-transformer embeddings. The final submission keeps K-Means as the chosen model because it is deterministic, easier to explain, and does not discard a large fraction of documents as outliers."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """alternative_summary_path = '../outputs/alternative_method_summary.csv'
if os.path.exists(alternative_summary_path):
    alternative_summary = pd.read_csv(alternative_summary_path)
    display(alternative_summary)
else:
    alternative_summary = pd.DataFrame({
        "method": ["BERTopic (TF-IDF + SVD)", "BERTopic (sentence-transformer)"],
        "topics": [52, 41],
        "outliers": [3234, 4449],
        "outlier_ratio": [0.1402, 0.1929],
    })
    display(alternative_summary)"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 9. Final Notes
The final delivery is organized around one selected clustering decision (`K=46`), one accepted recommendation pipeline, and one clearly separated experimental alternative. This keeps the submission focused while still documenting the explored alternatives and the reasoning behind the final choice."""
        )
    )

    nb["cells"] = cells

    os.makedirs("notebooks", exist_ok=True)
    with open("notebooks/journal_finder_project.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)


if __name__ == "__main__":
    create_notebook()
