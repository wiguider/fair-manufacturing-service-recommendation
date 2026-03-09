"""Unit tests for evaluation and fairness metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.evaluation.ranking_metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mrr,
    average_precision,
    evaluate_ranking,
)
from src.fairness.metrics import (
    demographic_parity_ratio,
    equity_of_exposure,
    expected_exposure_loss,
    compute_all_fairness_metrics,
)


# --- Ranking metrics ---

class TestPrecisionAtK:
    def test_all_relevant(self):
        ranking = np.array([1, 2, 3, 4, 5])
        relevant = {1, 2, 3, 4, 5}
        assert precision_at_k(ranking, relevant, 5) == 1.0

    def test_none_relevant(self):
        ranking = np.array([1, 2, 3, 4, 5])
        relevant = {6, 7, 8}
        assert precision_at_k(ranking, relevant, 5) == 0.0

    def test_partial(self):
        ranking = np.array([1, 2, 3, 4, 5])
        relevant = {1, 3, 5}
        assert precision_at_k(ranking, relevant, 5) == 3 / 5

    def test_at_k_smaller(self):
        ranking = np.array([1, 2, 3, 4, 5])
        relevant = {1, 2}
        assert precision_at_k(ranking, relevant, 2) == 1.0


class TestRecallAtK:
    def test_all_found(self):
        ranking = np.array([1, 2, 3])
        relevant = {1, 2, 3}
        assert recall_at_k(ranking, relevant, 3) == 1.0

    def test_none_found(self):
        ranking = np.array([4, 5, 6])
        relevant = {1, 2, 3}
        assert recall_at_k(ranking, relevant, 3) == 0.0

    def test_empty_relevant(self):
        ranking = np.array([1, 2, 3])
        assert recall_at_k(ranking, set(), 3) == 0.0


class TestNDCG:
    def test_perfect(self):
        ranking = np.array([1, 2, 3])
        relevant = {1, 2, 3}
        assert ndcg_at_k(ranking, relevant, 3) == pytest.approx(1.0)

    def test_worst(self):
        ranking = np.array([4, 5, 6])
        relevant = {1, 2, 3}
        assert ndcg_at_k(ranking, relevant, 3) == 0.0

    def test_single_relevant_at_top(self):
        ranking = np.array([1, 4, 5])
        relevant = {1}
        assert ndcg_at_k(ranking, relevant, 3) == pytest.approx(1.0)


class TestMRR:
    def test_first(self):
        ranking = np.array([1, 2, 3])
        assert mrr(ranking, {1}) == 1.0

    def test_second(self):
        ranking = np.array([2, 1, 3])
        assert mrr(ranking, {1}) == 0.5

    def test_not_found(self):
        ranking = np.array([2, 3, 4])
        assert mrr(ranking, {1}) == 0.0


class TestAveragePrecision:
    def test_perfect(self):
        ranking = np.array([1, 2, 3])
        relevant = {1, 2, 3}
        assert average_precision(ranking, relevant) == pytest.approx(1.0)

    def test_empty_relevant(self):
        ranking = np.array([1, 2, 3])
        assert average_precision(ranking, set()) == 0.0


# --- Fairness metrics ---

def _make_protected_attrs():
    return pd.DataFrame({
        "item_id": [0, 1, 2, 3, 4, 5],
        "size_group": ["small", "small", "small", "large", "large", "large"],
        "geo_group": ["northeast", "south", "west", "northeast", "south", "west"],
    })


class TestDPR:
    def test_balanced(self):
        attrs = _make_protected_attrs()
        ranking = np.array([0, 3, 1, 4])  # 2 small, 2 large
        assert demographic_parity_ratio(ranking, attrs, "size_group") == pytest.approx(1.0)

    def test_imbalanced(self):
        attrs = _make_protected_attrs()
        ranking = np.array([3, 4, 5, 0])  # 3 large, 1 small
        dpr = demographic_parity_ratio(ranking, attrs, "size_group")
        assert dpr < 1.0
        assert dpr > 0.0


class TestEquityOfExposure:
    def test_returns_float(self):
        attrs = _make_protected_attrs()
        ranking = np.array([0, 1, 2, 3, 4, 5])
        result = equity_of_exposure(ranking, attrs, "size_group")
        assert isinstance(result, float)
        assert result >= 0.0


class TestExpectedExposureLoss:
    def test_returns_float(self):
        attrs = _make_protected_attrs()
        ranking = np.array([0, 1, 2, 3, 4, 5])
        result = expected_exposure_loss(ranking, attrs, "size_group")
        assert isinstance(result, float)
        assert result >= 0.0


class TestComputeAllFairness:
    def test_returns_all_keys(self):
        attrs = _make_protected_attrs()
        ranking = np.array([0, 1, 2, 3, 4, 5])
        result = compute_all_fairness_metrics(ranking, attrs)
        assert "size_dpr" in result
        assert "geo_dpr" in result
        assert "inter_dpr" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
