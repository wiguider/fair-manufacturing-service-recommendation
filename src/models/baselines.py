"""Baseline recommendation models: Random, Popularity, Content-Based."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BaseRecommender(ABC):
    """Abstract base class for all recommenders."""

    @abstractmethod
    def fit(self, train_interactions: pd.DataFrame, item_features: pd.DataFrame = None):
        pass

    @abstractmethod
    def predict(self, user_id: int, candidate_items: np.ndarray) -> np.ndarray:
        """Return scores for candidate items for the given user."""
        pass

    def recommend(self, user_id: int, candidate_items: np.ndarray, top_k: int = 10) -> np.ndarray:
        """Return top-k item IDs sorted by predicted score."""
        scores = self.predict(user_id, candidate_items)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return candidate_items[top_indices]

    def recommend_all(
        self, user_ids: np.ndarray, candidate_items: np.ndarray, top_k: int = 10
    ) -> dict[int, np.ndarray]:
        """Return top-k recommendations for each user."""
        results = {}
        for uid in user_ids:
            results[uid] = self.recommend(uid, candidate_items, top_k)
        return results


class RandomRecommender(BaseRecommender):
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def fit(self, train_interactions: pd.DataFrame, item_features: pd.DataFrame = None):
        pass

    def predict(self, user_id: int, candidate_items: np.ndarray) -> np.ndarray:
        return self.rng.rand(len(candidate_items))


class PopularityRecommender(BaseRecommender):
    def __init__(self):
        self.item_popularity = {}

    def fit(self, train_interactions: pd.DataFrame, item_features: pd.DataFrame = None):
        counts = train_interactions["item_id"].value_counts()
        max_count = counts.max() if len(counts) > 0 else 1
        self.item_popularity = (counts / max_count).to_dict()

    def predict(self, user_id: int, candidate_items: np.ndarray) -> np.ndarray:
        return np.array([self.item_popularity.get(i, 0.0) for i in candidate_items])


class ContentBasedRecommender(BaseRecommender):
    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.item_vectors = None
        self.user_profiles = {}
        self.item_id_to_idx = {}

    def fit(self, train_interactions: pd.DataFrame, item_features: pd.DataFrame = None):
        if item_features is None:
            raise ValueError("ContentBasedRecommender requires item_features")

        # Build text representation for each item
        text_cols = [c for c in item_features.columns if c != "item_id" and item_features[c].dtype == object]
        if not text_cols:
            # Use all non-ID columns as string
            text_cols = [c for c in item_features.columns if c != "item_id"]

        item_features = item_features.copy()
        item_features["_text"] = item_features[text_cols].astype(str).agg(" ".join, axis=1)

        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(item_features["item_id"].values)}
        self.item_vectors = self.vectorizer.fit_transform(item_features["_text"])

        # Build user profiles from training interactions
        for uid, group in train_interactions.groupby("user_id"):
            item_indices = [self.item_id_to_idx[iid] for iid in group["item_id"] if iid in self.item_id_to_idx]
            if item_indices:
                self.user_profiles[uid] = self.item_vectors[item_indices].mean(axis=0)

    def predict(self, user_id: int, candidate_items: np.ndarray) -> np.ndarray:
        if user_id not in self.user_profiles:
            return np.zeros(len(candidate_items))

        user_vec = self.user_profiles[user_id]
        candidate_indices = [self.item_id_to_idx.get(i, -1) for i in candidate_items]

        scores = np.zeros(len(candidate_items))
        for idx, ci in enumerate(candidate_indices):
            if ci >= 0:
                scores[idx] = cosine_similarity(user_vec, self.item_vectors[ci]).item()
        return scores
