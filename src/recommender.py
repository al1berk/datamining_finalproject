import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from .preprocessing import clean_text


class JournalRecommender:
    def __init__(self, max_features=10000, min_df=2, ngram_range=(1, 2)):
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
        )
        self.tfidf_matrix = None
        self.df = None
        self.journal_indices = {}

    def fit(self, df):
        """
        Fit the vectorizer on the cleaned article text and cache row indices per journal.
        """
        print("Fitting TF-IDF Vectorizer...")
        self.df = df.reset_index(drop=True).copy()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["clean_text"])
        self.journal_indices = {
            journal: np.flatnonzero(self.df["journal_name"].to_numpy() == journal)
            for journal in self.df["journal_name"].unique()
        }
        print(f"TF-IDF Matrix shape: {self.tfidf_matrix.shape}")
        return self

    def _score_journals(self, similarities):
        if self.tfidf_matrix is None or self.df is None:
            raise ValueError("Recommender must be fitted before scoring journals.")

        records = []
        for journal_name, indices in self.journal_indices.items():
            journal_sims = similarities[indices]
            records.append(
                {
                    "journal_name": journal_name,
                    "avg_similarity": float(journal_sims.mean()),
                    "max_similarity": float(journal_sims.max()),
                    "article_count": int(len(indices)),
                }
            )

        journal_scores = pd.DataFrame(records)
        journal_scores["score"] = (
            0.7 * journal_scores["max_similarity"] +
            0.3 * journal_scores["avg_similarity"]
        )
        return journal_scores.sort_values("score", ascending=False).reset_index(drop=True)

    def recommend_journals(self, user_abstract, top_n=5):
        """
        Recommend the top journals for a user-provided abstract.
        """
        if self.tfidf_matrix is None or self.df is None:
            raise ValueError("Recommender must be fitted before calling recommend_journals.")

        cleaned = clean_text(user_abstract)
        user_vec = self.vectorizer.transform([cleaned])
        similarities = cosine_similarity(user_vec, self.tfidf_matrix).flatten()
        return self._score_journals(similarities).head(top_n)

    def evaluate(
        self,
        df=None,
        test_size=0.2,
        min_journal_articles=5,
        random_state=42,
        batch_size=200,
    ):
        """
        Evaluate the recommender on a stratified hold-out split.

        Only journals with at least `min_journal_articles` records are included so
        every evaluated journal has representation in both train and test splits.
        """
        source_df = df if df is not None else self.df
        if source_df is None:
            raise ValueError("Provide a dataframe or fit the recommender before evaluation.")

        counts = source_df["journal_name"].value_counts()
        eligible_journals = counts[counts >= min_journal_articles].index
        eligible_df = (
            source_df[source_df["journal_name"].isin(eligible_journals)]
            .reset_index(drop=True)
            .copy()
        )
        if eligible_df.empty:
            raise ValueError("No journals satisfy the minimum article threshold.")

        train_df, test_df = train_test_split(
            eligible_df,
            test_size=test_size,
            random_state=random_state,
            stratify=eligible_df["journal_name"],
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        evaluator = JournalRecommender(
            max_features=self.max_features,
            min_df=self.min_df,
            ngram_range=self.ngram_range,
        )
        evaluator.fit(train_df)

        test_matrix = evaluator.vectorizer.transform(test_df["clean_text"])

        correct_top1 = 0
        correct_top3 = 0
        correct_top5 = 0
        reciprocal_rank_total = 0.0

        for start in range(0, test_matrix.shape[0], batch_size):
            end = min(start + batch_size, test_matrix.shape[0])
            similarity_batch = cosine_similarity(
                test_matrix[start:end],
                evaluator.tfidf_matrix,
            )

            for offset, similarities in enumerate(similarity_batch):
                row_idx = start + offset
                ranked = evaluator._score_journals(similarities)
                ranked_journals = ranked["journal_name"].tolist()
                true_journal = test_df.iloc[row_idx]["journal_name"]

                if true_journal == ranked_journals[0]:
                    correct_top1 += 1
                if true_journal in ranked_journals[:3]:
                    correct_top3 += 1
                if true_journal in ranked_journals[:5]:
                    correct_top5 += 1

                reciprocal_rank_total += 1.0 / (ranked_journals.index(true_journal) + 1)

        test_count = len(test_df)
        return {
            "eligible_rows": int(len(eligible_df)),
            "eligible_journals": int(eligible_df["journal_name"].nunique()),
            "train_rows": int(len(train_df)),
            "test_rows": int(test_count),
            "top1_accuracy": correct_top1 / test_count,
            "top3_accuracy": correct_top3 / test_count,
            "top5_accuracy": correct_top5 / test_count,
            "mrr": reciprocal_rank_total / test_count,
        }
