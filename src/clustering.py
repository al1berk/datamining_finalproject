import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import Normalizer


DEFAULT_CLUSTER_STOP_WORDS = {
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


class TopicClusterer:
    def __init__(
        self,
        n_clusters=46,
        random_state=42,
        max_features=15000,
        min_df=5,
        max_df=0.35,
        ngram_range=(1, 2),
        n_components=50,
        extra_stop_words=None,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.n_components = n_components

        stop_words = sorted(
            set(ENGLISH_STOP_WORDS).union(DEFAULT_CLUSTER_STOP_WORDS).union(extra_stop_words or set())
        )
        self.vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
        )
        self.feature_svd = TruncatedSVD(
            n_components=n_components,
            random_state=random_state,
        )
        self.normalizer = Normalizer(copy=False)
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )
        self.visualization_svd = TruncatedSVD(n_components=2, random_state=random_state)

        self.term_matrix = None
        self.cluster_matrix = None
        self.cluster_labels = None

    def prepare_features(self, texts):
        """
        Build a clustering-oriented text representation.

        A dedicated vectorizer removes generic research vocabulary and then
        compresses the sparse TF-IDF space with TruncatedSVD before K-Means.
        """
        print("Building clustering features...")
        self.term_matrix = self.vectorizer.fit_transform(texts)
        reduced = self.feature_svd.fit_transform(self.term_matrix)
        self.cluster_matrix = self.normalizer.fit_transform(reduced)
        return self.cluster_matrix

    def fit_predict(self, df, text_column="clean_text"):
        """
        Fit K-Means on the prepared clustering features and append labels.
        """
        features = self.prepare_features(df[text_column])
        print(f"Clustering into {self.n_clusters} topics using K-Means...")
        self.cluster_labels = self.kmeans.fit_predict(features)

        result_df = df.copy()
        result_df["cluster"] = self.cluster_labels
        return result_df

    def evaluate_k_values(self, df, text_column="clean_text", k_values=(20, 22, 24), sample_size=2000):
        """
        Compare multiple K values on the same feature representation.
        """
        features = self.prepare_features(df[text_column])

        records = []
        for k in k_values:
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = model.fit_predict(features)
            cluster_sizes = np.bincount(labels)
            records.append(
                {
                    "k": int(k),
                    "silhouette_score": float(
                        silhouette_score(
                            features,
                            labels,
                            sample_size=sample_size,
                            random_state=self.random_state,
                        )
                    ),
                    "inertia": float(model.inertia_),
                    "davies_bouldin_score": float(davies_bouldin_score(features, labels)),
                    "calinski_harabasz_score": float(calinski_harabasz_score(features, labels)),
                    "min_cluster_size": int(cluster_sizes.min()),
                    "median_cluster_size": float(np.median(cluster_sizes)),
                    "max_cluster_size": int(cluster_sizes.max()),
                    "clusters_under_150": int((cluster_sizes < 150).sum()),
                    "clusters_under_100": int((cluster_sizes < 100).sum()),
                }
            )
        return pd.DataFrame(records).sort_values("k").reset_index(drop=True)

    def get_top_terms(self, top_n=15):
        """
        Return top terms per cluster from mean TF-IDF weights in the original term space.
        """
        if self.term_matrix is None or self.cluster_labels is None:
            raise ValueError("Clusterer must be fitted before requesting top terms.")

        terms = self.vectorizer.get_feature_names_out()
        cluster_terms = {}

        for cluster_id in range(self.n_clusters):
            indices = np.flatnonzero(self.cluster_labels == cluster_id)
            centroid = np.asarray(self.term_matrix[indices].mean(axis=0)).ravel()
            top_indices = centroid.argsort()[::-1][:top_n]
            cluster_terms[cluster_id] = [terms[ind] for ind in top_indices]

        return cluster_terms

    def get_cluster_summary(self, df, top_n=10):
        """
        Build a dataframe with cluster sizes and representative keywords.
        """
        counts = df["cluster"].value_counts().sort_index()
        top_terms = self.get_top_terms(top_n=top_n)
        return pd.DataFrame(
            {
                "cluster": counts.index.astype(int),
                "size": counts.values.astype(int),
                "top_terms": [", ".join(top_terms[int(i)]) for i in counts.index],
            }
        )

    def print_top_terms(self, top_n=15):
        """
        Print top terms for each fitted cluster.
        """
        cluster_terms = self.get_top_terms(top_n)
        for i in range(self.n_clusters):
            print(f"\nCluster {i}:")
            print(", ".join(cluster_terms[i]))

    def visualize_clusters(self, df, save_path=None, show=False):
        """
        Project the original TF-IDF term matrix to 2D for visualization.
        """
        if self.term_matrix is None or self.cluster_labels is None:
            raise ValueError("Clusterer must be fitted before visualization.")

        print("Reducing dimensions for visualization...")
        reduced = self.visualization_svd.fit_transform(self.term_matrix)

        plot_df = df.copy()
        plot_df["x"] = reduced[:, 0]
        plot_df["y"] = reduced[:, 1]

        plt.figure(figsize=(10, 7))
        cmap = plt.get_cmap("turbo", self.n_clusters)
        scatter = plt.scatter(
            plot_df["x"],
            plot_df["y"],
            c=plot_df["cluster"],
            cmap=cmap,
            vmin=-0.5,
            vmax=self.n_clusters - 0.5,
            alpha=0.6,
            s=18,
        )
        plt.title(f"Topic Clustering of Computer Science Articles (K={self.n_clusters})")
        plt.xlabel("Component 1 (TruncatedSVD)")
        plt.ylabel("Component 2 (TruncatedSVD)")

        colorbar = plt.colorbar(scatter, pad=0.02, ticks=range(self.n_clusters))
        colorbar.set_label("Cluster ID")
        colorbar.ax.tick_params(labelsize=7)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Cluster visualization saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        explained_variance = float(self.visualization_svd.explained_variance_ratio_.sum())
        return plot_df, explained_variance
