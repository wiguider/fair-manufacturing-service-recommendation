"""SentenceBERT-based recommender using semantic embeddings for item representation."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.models.baselines import BaseRecommender

logger = logging.getLogger(__name__)


def _build_item_text(item_features: pd.DataFrame) -> list[str]:
    """Construct a single text string per item from all non-ID feature columns.

    String columns are used as-is; numeric/categorical columns are serialised
    as ``column_name: value`` pairs so that field names contribute semantic
    signal when no free-text description column is present.
    """
    id_col = "item_id"
    feature_cols = [c for c in item_features.columns if c != id_col]

    if not feature_cols:
        return ["" for _ in range(len(item_features))]

    text_cols = [c for c in feature_cols if item_features[c].dtype == object]
    other_cols = [c for c in feature_cols if c not in text_cols]

    parts: list[pd.Series] = []
    if text_cols:
        parts.append(item_features[text_cols].fillna("").astype(str).agg(" ".join, axis=1))
    if other_cols:
        # Represent numeric/categorical columns as "key: value" tokens so that
        # field names carry semantic meaning during encoding.
        kv_series = item_features[other_cols].apply(
            lambda row: " ".join(f"{col}: {val}" for col, val in row.items()),
            axis=1,
        )
        parts.append(kv_series)

    combined = parts[0]
    for extra in parts[1:]:
        combined = combined + " " + extra

    return combined.tolist()


class SentenceBERTRecommender(BaseRecommender):
    """Content-based recommender driven by SentenceBERT semantic embeddings.

    Item descriptions are encoded once at fit-time with the
    ``all-MiniLM-L6-v2`` model (or any other compatible sentence-transformers
    model).  User profiles are constructed as the (optionally rating-weighted)
    mean of the embeddings of their interacted items.  At inference time,
    candidates are ranked by cosine similarity between the user profile vector
    and each candidate item embedding.

    Parameters
    ----------
    model_name:
        Any ``sentence-transformers`` model identifier.  Defaults to the
        lightweight ``all-MiniLM-L6-v2`` (384-dim, ~22 M params).
    batch_size:
        Encoding batch size forwarded to ``SentenceTransformer.encode()``.
    normalize_embeddings:
        Whether to L2-normalise embeddings before storing them.  When
        ``True`` cosine similarity reduces to a dot product, which is
        slightly faster for large candidate sets.
    seed:
        Random seed kept for interface consistency; not used for stochastic
        decisions within this model.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.seed = seed

        # Populated in fit()
        self._encoder = None
        self._item_embeddings: Optional[np.ndarray] = None  # (n_items, dim)
        self._item_id_to_idx: dict[int, int] = {}
        self._user_profiles: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_encoder(self) -> None:
        """Lazily import and instantiate the SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceBERTRecommender. "
                "Install it with:  pip install 'sentence-transformers>=2.2.0'"
            ) from exc

        logger.info("Loading sentence-transformers model '%s'", self.model_name)
        self._encoder = SentenceTransformer(self.model_name)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a list of strings; returns float32 array of shape (n, dim)."""
        if self._encoder is None:
            self._load_encoder()

        embeddings: np.ndarray = self._encoder.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    # ------------------------------------------------------------------
    # BaseRecommender interface
    # ------------------------------------------------------------------

    def fit(
        self,
        train_interactions: pd.DataFrame,
        item_features: Optional[pd.DataFrame] = None,
    ) -> None:
        """Encode item descriptions and build mean-embedding user profiles.

        Parameters
        ----------
        train_interactions:
            DataFrame with columns ``user_id``, ``item_id``, and optionally
            ``rating``.
        item_features:
            DataFrame with column ``item_id`` plus any number of feature
            columns.  When ``None`` or empty, items are represented by their
            ID string only, which degrades gracefully to a uniform signal.
        """
        # ---- Build item text corpus ----------------------------------------
        if item_features is not None and len(item_features) > 0:
            items_df = item_features.copy().reset_index(drop=True)
        else:
            # Fallback: one row per unique item_id seen in interactions
            unique_items = train_interactions["item_id"].unique()
            items_df = pd.DataFrame({"item_id": unique_items})

        item_ids: np.ndarray = items_df["item_id"].values
        self._item_id_to_idx = {int(iid): idx for idx, iid in enumerate(item_ids)}

        texts = _build_item_text(items_df)
        logger.info(
            "Encoding %d item descriptions with model '%s'",
            len(texts),
            self.model_name,
        )
        self._item_embeddings = self._encode_texts(texts)  # (n_items, dim)

        # ---- Build user profiles as mean of interacted item embeddings ------
        self._user_profiles = {}
        for uid, group in train_interactions.groupby("user_id"):
            # Collect indices of items that appear in our embedding matrix
            interacted_indices: list[int] = [
                self._item_id_to_idx[int(iid)]
                for iid in group["item_id"]
                if int(iid) in self._item_id_to_idx
            ]
            if not interacted_indices:
                continue

            item_vecs = self._item_embeddings[interacted_indices]  # (k, dim)

            if "rating" in group.columns:
                # Rating-weighted mean: higher-rated items contribute more
                rated_mask = group["item_id"].apply(
                    lambda iid: int(iid) in self._item_id_to_idx
                )
                weights = group.loc[rated_mask, "rating"].values.astype(np.float32)
                weight_sum = weights.sum()
                if weight_sum > 0:
                    profile = np.average(item_vecs, axis=0, weights=weights)
                else:
                    profile = item_vecs.mean(axis=0)
            else:
                profile = item_vecs.mean(axis=0)

            # Re-normalise the profile so cosine similarity stays well-scaled
            if self.normalize_embeddings:
                norm = np.linalg.norm(profile)
                if norm > 0:
                    profile = profile / norm

            self._user_profiles[int(uid)] = profile.astype(np.float32)

        logger.info(
            "Built profiles for %d users (%d items indexed)",
            len(self._user_profiles),
            len(self._item_id_to_idx),
        )

    def predict(self, user_id: int, candidate_items: np.ndarray) -> np.ndarray:
        """Score candidate items for a user via cosine similarity.

        Parameters
        ----------
        user_id:
            Integer user identifier.
        candidate_items:
            1-D array of integer item IDs to score.

        Returns
        -------
        np.ndarray
            Float32 scores aligned with ``candidate_items``.  Cold-start
            users (not seen during training) receive uniform zero scores.
        """
        if self._item_embeddings is None:
            return np.zeros(len(candidate_items), dtype=np.float32)

        uid = int(user_id)
        if uid not in self._user_profiles:
            return np.zeros(len(candidate_items), dtype=np.float32)

        user_vec = self._user_profiles[uid].reshape(1, -1)  # (1, dim)
        dim = self._item_embeddings.shape[1]

        # Gather candidate embedding matrix; unknown items stay as zero vectors
        candidate_vecs = np.zeros((len(candidate_items), dim), dtype=np.float32)
        valid_mask = np.zeros(len(candidate_items), dtype=bool)

        for pos, iid in enumerate(candidate_items):
            idx = self._item_id_to_idx.get(int(iid), -1)
            if idx >= 0:
                candidate_vecs[pos] = self._item_embeddings[idx]
                valid_mask[pos] = True

        scores = np.zeros(len(candidate_items), dtype=np.float32)
        if valid_mask.any():
            # Single batched cosine similarity call for all valid candidates
            sims = cosine_similarity(user_vec, candidate_vecs[valid_mask]).flatten()
            scores[valid_mask] = sims.astype(np.float32)

        return scores
