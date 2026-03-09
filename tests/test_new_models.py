"""Unit tests for new recommendation models: SentenceBERTRecommender and UltraGCNRecommender."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.models.llm_recommender import SentenceBERTRecommender, _build_item_text
from src.models.gnn_advanced import UltraGCNRecommender, UltraGCNModel


# =============================================================================
# Fixtures: minimal synthetic datasets for fast tests
# =============================================================================


@pytest.fixture
def small_interactions():
    """Create a minimal user-item interaction dataset."""
    return pd.DataFrame({
        "user_id": [0, 0, 1, 1, 2, 2],
        "item_id": [0, 1, 1, 2, 0, 2],
        "rating": [5.0, 4.0, 3.0, 5.0, 4.0, 2.0],
    })


@pytest.fixture
def small_item_features():
    """Create minimal item features for content-based methods."""
    return pd.DataFrame({
        "item_id": [0, 1, 2, 3],
        "description": [
            "high quality manufacturing services",
            "precision engineering solutions",
            "industrial fabrication experts",
            "advanced supply chain partner",
        ],
        "size_group": ["small", "small", "large", "large"],
        "geo_group": ["northeast", "south", "west", "midwest"],
    })


@pytest.fixture
def medium_interactions():
    """Create a larger interaction dataset for testing scalability."""
    np.random.seed(42)
    n_users, n_items, n_interactions = 20, 15, 100
    users = np.random.randint(0, n_users, n_interactions)
    items = np.random.randint(0, n_items, n_interactions)
    ratings = np.random.randint(1, 6, n_interactions)
    return pd.DataFrame({
        "user_id": users,
        "item_id": items,
        "rating": ratings,
    })


@pytest.fixture
def medium_item_features():
    """Create medium-scale item features."""
    return pd.DataFrame({
        "item_id": range(15),
        "description": [
            f"manufacturer type {i % 3} with certifications" for i in range(15)
        ],
        "size_group": ["small"] * 5 + ["medium"] * 5 + ["large"] * 5,
        "geo_group": ["northeast", "south", "west", "midwest"] * 4,
    })


# =============================================================================
# Tests for SentenceBERTRecommender
# =============================================================================


class TestBuildItemText:
    """Test the text-building utility function."""

    def test_no_features(self):
        """Empty item_features should produce empty strings."""
        df = pd.DataFrame({"item_id": [0, 1, 2]})
        result = _build_item_text(df)
        assert len(result) == 3
        assert all(t == "" for t in result)

    def test_only_text_columns(self):
        """Text columns only should be concatenated with spaces."""
        df = pd.DataFrame({
            "item_id": [0, 1],
            "desc": ["high quality", "precision engineering"],
            "tags": ["service", "product"],
        })
        result = _build_item_text(df)
        assert len(result) == 2
        assert "high quality" in result[0]
        assert "service" in result[0]

    def test_numeric_and_text_columns(self):
        """Mixed columns should produce "key: value" format for numerics."""
        df = pd.DataFrame({
            "item_id": [0, 1],
            "desc": ["maker", "supplier"],
            "size": [10, 20],
        })
        result = _build_item_text(df)
        assert len(result) == 2
        # Text columns appear first, then numeric fields as "key: value"
        assert "maker" in result[0]
        assert "size:" in result[0]


class TestSentenceBERTRecommenderInit:
    """Test initialization of SentenceBERTRecommender."""

    def test_default_init(self):
        """Default initialization should set standard hyperparameters."""
        rec = SentenceBERTRecommender()
        assert rec.model_name == "all-MiniLM-L6-v2"
        assert rec.batch_size == 64
        assert rec.normalize_embeddings is True
        assert rec.seed == 42
        assert rec._encoder is None
        assert rec._item_embeddings is None

    def test_custom_init(self):
        """Custom parameters should be stored correctly."""
        rec = SentenceBERTRecommender(
            model_name="distilbert-base-uncased",
            batch_size=32,
            normalize_embeddings=False,
            seed=123,
        )
        assert rec.model_name == "distilbert-base-uncased"
        assert rec.batch_size == 32
        assert rec.normalize_embeddings is False
        assert rec.seed == 123


class TestSentenceBERTRecommenderFit:
    """Test the fit() method with mocked encoder."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_fit_with_item_features(self, mock_st_class, small_interactions, small_item_features):
        """fit() should encode items and build user profiles."""
        # Mock the encoder
        mock_encoder = MagicMock()
        mock_st_class.return_value = mock_encoder

        # Mock embeddings: 4 items, 8-dim
        embeddings = np.random.randn(4, 8).astype(np.float32)
        mock_encoder.encode.return_value = embeddings

        rec = SentenceBERTRecommender()
        rec.fit(small_interactions, small_item_features)

        # Verify encoder was loaded and called
        mock_st_class.assert_called_once_with("all-MiniLM-L6-v2")
        mock_encoder.encode.assert_called_once()

        # Check internal state
        assert rec._item_embeddings is not None
        assert rec._item_embeddings.shape == (4, 8)
        assert len(rec._item_id_to_idx) == 4
        assert len(rec._user_profiles) > 0

    @patch("sentence_transformers.SentenceTransformer")
    def test_fit_without_item_features(self, mock_st_class, small_interactions):
        """fit() should work if item_features is None, falling back to item IDs."""
        mock_encoder = MagicMock()
        mock_st_class.return_value = mock_encoder

        # 3 unique items in small_interactions
        embeddings = np.random.randn(3, 8).astype(np.float32)
        mock_encoder.encode.return_value = embeddings

        rec = SentenceBERTRecommender()
        rec.fit(small_interactions, item_features=None)

        assert rec._item_embeddings is not None
        assert rec._item_embeddings.shape == (3, 8)
        assert len(rec._item_id_to_idx) == 3

    @patch("sentence_transformers.SentenceTransformer")
    def test_fit_with_ratings(self, mock_st_class, small_interactions, small_item_features):
        """fit() should use ratings to weight user profiles."""
        mock_encoder = MagicMock()
        mock_st_class.return_value = mock_encoder
        embeddings = np.random.randn(4, 8).astype(np.float32)
        mock_encoder.encode.return_value = embeddings

        rec = SentenceBERTRecommender()
        rec.fit(small_interactions, small_item_features)

        # User profiles should exist
        assert len(rec._user_profiles) > 0
        # Each profile should have the same dimension as embeddings
        for profile in rec._user_profiles.values():
            assert profile.shape == (8,)

    @patch("sentence_transformers.SentenceTransformer")
    def test_fit_missing_sentence_transformers(self, mock_st_class, small_interactions, small_item_features):
        """fit() should raise ImportError if sentence-transformers is missing."""
        mock_st_class.side_effect = ImportError("No module named 'sentence_transformers'")

        rec = SentenceBERTRecommender()
        with pytest.raises(ImportError, match="sentence-transformers is required"):
            rec.fit(small_interactions, small_item_features)


class TestSentenceBERTRecommenderPredict:
    """Test the predict() method."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_predict_known_user(self, mock_st_class, small_interactions, small_item_features):
        """predict() should return scores for a known user."""
        mock_encoder = MagicMock()
        mock_st_class.return_value = mock_encoder
        embeddings = np.random.randn(4, 8).astype(np.float32)
        mock_encoder.encode.return_value = embeddings

        rec = SentenceBERTRecommender()
        rec.fit(small_interactions, small_item_features)

        scores = rec.predict(user_id=0, candidate_items=np.array([0, 1, 2, 3]))

        assert scores is not None
        assert len(scores) == 4
        assert scores.dtype == np.float32

    @patch("sentence_transformers.SentenceTransformer")
    def test_predict_unknown_user(self, mock_st_class, small_interactions, small_item_features):
        """predict() should return zero scores for unknown users (cold-start)."""
        mock_encoder = MagicMock()
        mock_st_class.return_value = mock_encoder
        embeddings = np.random.randn(4, 8).astype(np.float32)
        mock_encoder.encode.return_value = embeddings

        rec = SentenceBERTRecommender()
        rec.fit(small_interactions, small_item_features)

        scores = rec.predict(user_id=999, candidate_items=np.array([0, 1, 2]))

        assert len(scores) == 3
        assert np.allclose(scores, 0.0)

    @patch("sentence_transformers.SentenceTransformer")
    def test_predict_before_fit(self, mock_st_class):
        """predict() called before fit() should return zeros."""
        rec = SentenceBERTRecommender()

        scores = rec.predict(user_id=0, candidate_items=np.array([0, 1, 2]))

        assert len(scores) == 3
        assert np.allclose(scores, 0.0)

    @patch("sentence_transformers.SentenceTransformer")
    def test_predict_cold_start_items(self, mock_st_class, small_interactions, small_item_features):
        """predict() should return zeros for unknown items while preserving others."""
        mock_encoder = MagicMock()
        mock_st_class.return_value = mock_encoder
        embeddings = np.random.randn(4, 8).astype(np.float32)
        mock_encoder.encode.return_value = embeddings

        rec = SentenceBERTRecommender()
        rec.fit(small_interactions, small_item_features)

        # Mix of known (0, 1) and unknown (999) items
        scores = rec.predict(user_id=0, candidate_items=np.array([0, 999, 1]))

        assert len(scores) == 3
        assert scores[0] != 0.0  # Known item
        assert scores[1] == 0.0  # Unknown item
        assert scores[2] != 0.0  # Known item

    @patch("sentence_transformers.SentenceTransformer")
    def test_predict_output_shape(self, mock_st_class, small_interactions, small_item_features):
        """predict() output shape should match candidate_items length."""
        mock_encoder = MagicMock()
        mock_st_class.return_value = mock_encoder
        embeddings = np.random.randn(4, 8).astype(np.float32)
        mock_encoder.encode.return_value = embeddings

        rec = SentenceBERTRecommender()
        rec.fit(small_interactions, small_item_features)

        for k in [1, 2, 5, 10]:
            candidates = np.arange(min(k, 4))
            scores = rec.predict(0, candidates)
            assert scores.shape == (len(candidates),)


class TestSentenceBERTRecommenderRecommend:
    """Test high-level recommend() and recommend_all() methods."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_recommend(self, mock_st_class, small_interactions, small_item_features):
        """recommend() should return top-k items sorted by score."""
        mock_encoder = MagicMock()
        mock_st_class.return_value = mock_encoder

        # Fixed embeddings for reproducibility
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)
        mock_encoder.encode.return_value = embeddings

        rec = SentenceBERTRecommender()
        rec.fit(small_interactions, small_item_features)

        recs = rec.recommend(user_id=0, candidate_items=np.array([0, 1, 2, 3]), top_k=2)

        assert len(recs) == 2
        assert all(r in [0, 1, 2, 3] for r in recs)

    @patch("sentence_transformers.SentenceTransformer")
    def test_recommend_all(self, mock_st_class, small_interactions, small_item_features):
        """recommend_all() should return top-k for each user."""
        mock_encoder = MagicMock()
        mock_st_class.return_value = mock_encoder
        embeddings = np.random.randn(4, 8).astype(np.float32)
        mock_encoder.encode.return_value = embeddings

        rec = SentenceBERTRecommender()
        rec.fit(small_interactions, small_item_features)

        results = rec.recommend_all(
            user_ids=np.array([0, 1, 2]),
            candidate_items=np.array([0, 1, 2, 3]),
            top_k=2,
        )

        assert len(results) == 3
        for uid in [0, 1, 2]:
            assert uid in results
            assert len(results[uid]) <= 2


# =============================================================================
# Tests for UltraGCNRecommender
# =============================================================================


class TestUltraGCNModelInit:
    """Test UltraGCNModel PyTorch module initialization."""

    def test_model_init(self):
        """UltraGCNModel should initialize embeddings with Xavier uniform."""
        model = UltraGCNModel(num_users=10, num_items=5, embedding_dim=16)

        assert model.num_users == 10
        assert model.num_items == 5
        assert model.user_emb.weight.shape == (10, 16)
        assert model.item_emb.weight.shape == (5, 16)

    def test_predict_all_items_shape(self):
        """predict_all_items() should return scores for all items."""
        model = UltraGCNModel(num_users=10, num_items=5, embedding_dim=16)

        scores = model.predict_all_items(user_id=0)

        assert scores.shape == (5,)
        assert scores.dtype == np.float64 or scores.dtype == np.float32

    def test_predict_all_items_different_users(self):
        """predict_all_items() should return different scores for different users."""
        model = UltraGCNModel(num_users=10, num_items=5, embedding_dim=16)

        scores_u0 = model.predict_all_items(user_id=0)
        scores_u1 = model.predict_all_items(user_id=1)

        # Scores should differ between users (with very high probability)
        assert not np.allclose(scores_u0, scores_u1)


class TestUltraGCNRecommenderInit:
    """Test UltraGCNRecommender initialization."""

    def test_default_init(self):
        """Default initialization should set standard hyperparameters."""
        rec = UltraGCNRecommender()

        assert rec.embedding_dim == 64
        assert rec.ii_topk == 10
        assert rec.lambda_constraint == 1e-3
        assert rec.w1 == 1.0
        assert rec.neg_sample_ratio == 1
        assert rec.batch_size == 1024
        assert rec.num_epochs == 100
        assert rec.patience == 10
        assert rec.model is None

    def test_custom_init(self):
        """Custom hyperparameters should be stored."""
        rec = UltraGCNRecommender(
            embedding_dim=32,
            ii_topk=5,
            lambda_constraint=0.01,
            lr=0.01,
            batch_size=512,
            num_epochs=50,
        )

        assert rec.embedding_dim == 32
        assert rec.ii_topk == 5
        assert rec.lambda_constraint == 0.01
        assert rec.lr == 0.01
        assert rec.batch_size == 512
        assert rec.num_epochs == 50


class TestUltraGCNRecommenderFit:
    """Test the fit() method."""

    def test_fit_small_dataset(self, small_interactions):
        """fit() should complete successfully on small data."""
        rec = UltraGCNRecommender(
            embedding_dim=8,
            num_epochs=2,
            batch_size=2,
            patience=100,  # High patience to avoid early stopping
        )

        # Should not raise any exceptions
        rec.fit(small_interactions)

        assert rec.model is not None
        assert rec.num_users == small_interactions["user_id"].max() + 1
        assert rec.num_items == small_interactions["item_id"].max() + 1

    def test_fit_with_item_features(self, small_interactions, small_item_features):
        """fit() should accept (but ignore) item_features."""
        rec = UltraGCNRecommender(
            embedding_dim=8,
            num_epochs=2,
            batch_size=2,
            patience=100,
        )

        # Should not raise any exceptions
        rec.fit(small_interactions, small_item_features)

        assert rec.model is not None

    def test_fit_medium_dataset(self, medium_interactions):
        """fit() should scale to slightly larger datasets."""
        rec = UltraGCNRecommender(
            embedding_dim=16,
            ii_topk=5,
            num_epochs=3,
            batch_size=16,
            patience=100,
        )

        rec.fit(medium_interactions)

        assert rec.model is not None
        assert rec.num_users == medium_interactions["user_id"].max() + 1
        assert rec.num_items == medium_interactions["item_id"].max() + 1

    def test_fit_initializes_omega(self, small_interactions):
        """fit() should compute omega weights for interactions."""
        rec = UltraGCNRecommender(
            embedding_dim=8,
            num_epochs=1,
            batch_size=2,
            patience=100,
        )

        rec.fit(small_interactions)

        # Should have omega for at least one interaction
        assert len(rec._omega) > 0
        # All omega values should be positive
        assert all(v > 0 for v in rec._omega.values())

    def test_fit_initializes_beta_neighbours(self, small_interactions):
        """fit() should compute item-item neighbours."""
        rec = UltraGCNRecommender(
            embedding_dim=8,
            ii_topk=2,
            num_epochs=1,
            batch_size=2,
            patience=100,
        )

        rec.fit(small_interactions)

        # Should have neighbours for items
        assert len(rec._ii_neighbours) > 0


class TestUltraGCNRecommenderPredict:
    """Test the predict() method."""

    def test_predict_known_user(self, small_interactions):
        """predict() should return scores for a known user."""
        rec = UltraGCNRecommender(
            embedding_dim=8,
            num_epochs=2,
            batch_size=2,
            patience=100,
        )
        rec.fit(small_interactions)

        candidate_items = np.array([0, 1, 2])
        scores = rec.predict(user_id=0, candidate_items=candidate_items)

        assert len(scores) == len(candidate_items)
        assert scores.dtype in [np.float32, np.float64]

    def test_predict_before_fit(self):
        """predict() called before fit() should return zeros."""
        rec = UltraGCNRecommender()

        scores = rec.predict(user_id=0, candidate_items=np.array([0, 1, 2]))

        assert len(scores) == 3
        assert np.allclose(scores, 0.0)

    def test_predict_output_shape(self, small_interactions):
        """predict() output shape should match candidate_items."""
        rec = UltraGCNRecommender(
            embedding_dim=8,
            num_epochs=2,
            batch_size=2,
            patience=100,
        )
        rec.fit(small_interactions)

        for k in [1, 2, 3]:
            candidates = np.arange(k)
            scores = rec.predict(0, candidates)
            assert scores.shape == (k,)

    def test_predict_valid_items(self, small_interactions):
        """predict() should return numeric scores for valid items."""
        rec = UltraGCNRecommender(
            embedding_dim=8,
            num_epochs=2,
            batch_size=2,
            patience=100,
        )
        rec.fit(small_interactions)

        scores = rec.predict(user_id=0, candidate_items=np.array([0, 1, 2]))

        assert all(np.isfinite(scores))


class TestUltraGCNRecommenderRecommend:
    """Test high-level recommend() methods."""

    def test_recommend(self, small_interactions):
        """recommend() should return top-k items."""
        rec = UltraGCNRecommender(
            embedding_dim=8,
            num_epochs=2,
            batch_size=2,
            patience=100,
        )
        rec.fit(small_interactions)

        candidates = np.array([0, 1, 2])
        recs = rec.recommend(user_id=0, candidate_items=candidates, top_k=2)

        assert len(recs) <= 2
        assert all(r in candidates for r in recs)

    def test_recommend_all(self, small_interactions):
        """recommend_all() should return top-k for each user."""
        rec = UltraGCNRecommender(
            embedding_dim=8,
            num_epochs=2,
            batch_size=2,
            patience=100,
        )
        rec.fit(small_interactions)

        user_ids = np.array([0, 1, 2])
        candidates = np.array([0, 1, 2])
        results = rec.recommend_all(
            user_ids=user_ids,
            candidate_items=candidates,
            top_k=2,
        )

        assert len(results) == 3
        for uid in user_ids:
            assert uid in results
            assert len(results[uid]) <= 2


# =============================================================================
# Integration tests comparing both models
# =============================================================================


class TestModelComparison:
    """Test properties of both models together."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_both_models_fit(self, mock_st_class, small_interactions, small_item_features):
        """Both models should fit successfully on the same data."""
        # Setup SentenceBERT mock
        mock_encoder = MagicMock()
        mock_st_class.return_value = mock_encoder
        embeddings = np.random.randn(4, 8).astype(np.float32)
        mock_encoder.encode.return_value = embeddings

        # Fit SentenceBERT
        sbert_rec = SentenceBERTRecommender()
        sbert_rec.fit(small_interactions, small_item_features)

        # Fit UltraGCN
        ultragcn_rec = UltraGCNRecommender(
            embedding_dim=8,
            num_epochs=2,
            batch_size=2,
            patience=100,
        )
        ultragcn_rec.fit(small_interactions, small_item_features)

        assert sbert_rec._item_embeddings is not None
        assert ultragcn_rec.model is not None

    @patch("sentence_transformers.SentenceTransformer")
    def test_both_models_predict(self, mock_st_class, small_interactions, small_item_features):
        """Both models should produce predictions."""
        mock_encoder = MagicMock()
        mock_st_class.return_value = mock_encoder
        embeddings = np.random.randn(4, 8).astype(np.float32)
        mock_encoder.encode.return_value = embeddings

        sbert_rec = SentenceBERTRecommender()
        sbert_rec.fit(small_interactions, small_item_features)

        ultragcn_rec = UltraGCNRecommender(
            embedding_dim=8,
            num_epochs=2,
            batch_size=2,
            patience=100,
        )
        ultragcn_rec.fit(small_interactions, small_item_features)

        candidates = np.array([0, 1, 2])

        sbert_scores = sbert_rec.predict(0, candidates)
        ultragcn_scores = ultragcn_rec.predict(0, candidates)

        assert len(sbert_scores) == len(candidates)
        assert len(ultragcn_scores) == len(candidates)

    @patch("sentence_transformers.SentenceTransformer")
    def test_both_models_recommend(self, mock_st_class, small_interactions, small_item_features):
        """Both models should produce recommendations."""
        mock_encoder = MagicMock()
        mock_st_class.return_value = mock_encoder
        embeddings = np.random.randn(4, 8).astype(np.float32)
        mock_encoder.encode.return_value = embeddings

        sbert_rec = SentenceBERTRecommender()
        sbert_rec.fit(small_interactions, small_item_features)

        ultragcn_rec = UltraGCNRecommender(
            embedding_dim=8,
            num_epochs=2,
            batch_size=2,
            patience=100,
        )
        ultragcn_rec.fit(small_interactions, small_item_features)

        candidates = np.array([0, 1, 2])

        sbert_recs = sbert_rec.recommend(0, candidates, top_k=2)
        ultragcn_recs = ultragcn_rec.recommend(0, candidates, top_k=2)

        assert len(sbert_recs) <= 2
        assert len(ultragcn_recs) <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
