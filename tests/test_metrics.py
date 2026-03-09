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
from src.fairness.domain_metrics import (
    supply_chain_hhi,
    certification_coverage,
    regional_resilience,
    compute_all_domain_metrics,
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


# --- Domain-specific manufacturing metrics ---

def _make_item_features():
    """Minimal item_features fixture matching the synthetic MSKG schema."""
    return pd.DataFrame({
        "item_id": [0, 1, 2, 3, 4, 5],
        "size_group": ["small", "small", "medium", "medium", "large", "large"],
        "geo_group": ["northeast", "south", "west", "midwest", "northeast", "south"],
        # Each manufacturer holds a whitespace-separated cert string (may be empty)
        "certifications": [
            "ISO9001 AS9100",
            "ISO9001 IATF16949",
            "ISO14001",
            "NADCAP",
            "ISO9001 ISO14001 AS9100",
            "",
        ],
    })


class TestSupplyChainHHI:
    def test_single_group_is_max(self):
        """A ranking with one size group should give HHI == 1.0."""
        features = _make_item_features()
        # Items 4 and 5 are both "large"
        ranking = np.array([4, 5])
        assert supply_chain_hhi(ranking, features) == pytest.approx(1.0)

    def test_two_equal_groups_is_half(self):
        """Two equally represented groups give HHI == 0.5."""
        features = _make_item_features()
        # Items 0,1 are small; items 4,5 are large — 2 of each
        ranking = np.array([0, 1, 4, 5])
        assert supply_chain_hhi(ranking, features) == pytest.approx(0.5)

    def test_three_equal_groups(self):
        """Three equally represented groups give HHI == 1/3."""
        features = _make_item_features()
        # One item per size: small=0, medium=2, large=4
        ranking = np.array([0, 2, 4])
        assert supply_chain_hhi(ranking, features) == pytest.approx(1 / 3)

    def test_more_diverse_is_lower(self):
        """A more diverse ranking should produce a lower HHI than a concentrated one."""
        features = _make_item_features()
        diverse = np.array([0, 2, 4])       # one from each size group
        concentrated = np.array([0, 1, 4])  # 2 small, 1 large
        assert supply_chain_hhi(diverse, features) < supply_chain_hhi(concentrated, features)

    def test_empty_ranking(self):
        features = _make_item_features()
        assert supply_chain_hhi(np.array([]), features) == 0.0

    def test_top_k_respected(self):
        """top_k should restrict which items are counted."""
        features = _make_item_features()
        # Full ranking is diverse; top-1 is always a single group -> HHI == 1.0
        ranking = np.array([0, 2, 4])
        assert supply_chain_hhi(ranking, features, top_k=1) == pytest.approx(1.0)

    def test_missing_group_col_returns_one(self):
        """When the group column is absent the function returns worst-case 1.0."""
        features = _make_item_features().drop(columns=["size_group"])
        ranking = np.array([0, 1, 2])
        assert supply_chain_hhi(ranking, features, group_col="size_group") == pytest.approx(1.0)

    def test_return_type(self):
        features = _make_item_features()
        result = supply_chain_hhi(np.array([0, 1, 2, 3, 4, 5]), features)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestCertificationCoverage:
    def test_full_coverage(self):
        """Selecting all items should cover every certification."""
        features = _make_item_features()
        ranking = np.array([0, 1, 2, 3, 4, 5])
        assert certification_coverage(ranking, features) == pytest.approx(1.0)

    def test_zero_coverage_empty_ranking(self):
        features = _make_item_features()
        assert certification_coverage(np.array([]), features) == 0.0

    def test_partial_coverage(self):
        """A subset of items should yield a coverage strictly between 0 and 1."""
        features = _make_item_features()
        # Item 2 has only ISO14001; total universe = {ISO9001, AS9100, IATF16949,
        # ISO14001, NADCAP} — 5 certs.  Item 2 alone covers 1/5 = 0.2.
        ranking = np.array([2])
        result = certification_coverage(ranking, features)
        assert 0.0 < result < 1.0

    def test_explicit_all_certs_universe(self):
        """Passing all_certs overrides the inferred universe."""
        features = _make_item_features()
        # Only ISO9001 in the universe; item 0 has it -> 1/1 = 1.0
        ranking = np.array([0])
        result = certification_coverage(ranking, features, all_certs=["ISO9001"])
        assert result == pytest.approx(1.0)

    def test_explicit_universe_not_covered(self):
        """A cert in all_certs that no recommended item holds should reduce coverage."""
        features = _make_item_features()
        ranking = np.array([2])  # item 2 has only ISO14001
        result = certification_coverage(
            ranking, features, all_certs=["ISO9001", "ISO14001"]
        )
        assert result == pytest.approx(0.5)

    def test_no_cert_column(self):
        features = _make_item_features().drop(columns=["certifications"])
        ranking = np.array([0, 1, 2])
        assert certification_coverage(ranking, features) == 0.0

    def test_item_with_empty_certs_not_counted(self):
        """Item 5 has an empty cert string and should contribute nothing."""
        features = _make_item_features()
        # Universe inferred from items 0-4 only; item 5 adds nothing
        universe = {"ISO9001", "AS9100", "IATF16949", "ISO14001", "NADCAP"}
        ranking = np.array([5])
        result = certification_coverage(ranking, features)
        assert result == pytest.approx(0.0)

    def test_top_k_respected(self):
        """top_k should restrict which items contribute certifications."""
        features = _make_item_features()
        # top_k=1 -> only item 0 (ISO9001, AS9100); full ranking covers everything
        ranking = np.array([0, 1, 2, 3, 4])
        partial = certification_coverage(ranking, features, top_k=1)
        full = certification_coverage(ranking, features)
        assert partial <= full

    def test_return_type_and_bounds(self):
        features = _make_item_features()
        result = certification_coverage(np.array([0, 1, 2, 3, 4, 5]), features)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestRegionalResilience:
    def test_single_region_is_zero(self):
        """All items from one region -> entropy = 0 -> resilience = 0."""
        attrs = _make_protected_attrs()
        # Items 0 and 3 are both "northeast"
        ranking = np.array([0, 3])
        assert regional_resilience(ranking, attrs) == pytest.approx(0.0)

    def test_perfectly_balanced_is_one(self):
        """Items evenly distributed across all four regions -> normalised entropy = 1."""
        attrs = pd.DataFrame({
            "item_id": [0, 1, 2, 3],
            "size_group": ["small"] * 4,
            "geo_group": ["northeast", "south", "west", "midwest"],
        })
        ranking = np.array([0, 1, 2, 3])
        assert regional_resilience(ranking, attrs) == pytest.approx(1.0)

    def test_more_diverse_is_higher(self):
        """A geographically varied ranking should score higher than a concentrated one."""
        attrs = _make_protected_attrs()
        diverse = np.array([0, 1, 2])      # northeast, south, west
        concentrated = np.array([0, 3, 1]) # northeast, northeast, south
        assert regional_resilience(diverse, attrs) > regional_resilience(concentrated, attrs)

    def test_empty_ranking(self):
        attrs = _make_protected_attrs()
        assert regional_resilience(np.array([]), attrs) == 0.0

    def test_top_k_respected(self):
        """top_k should restrict which items are evaluated."""
        attrs = _make_protected_attrs()
        # Full ranking is diverse; top-1 is a single region -> resilience = 0
        ranking = np.array([0, 1, 2])
        assert regional_resilience(ranking, attrs, top_k=1) == pytest.approx(0.0)

    def test_missing_geo_col(self):
        attrs = _make_protected_attrs().drop(columns=["geo_group"])
        ranking = np.array([0, 1, 2])
        assert regional_resilience(ranking, attrs, geo_col="geo_group") == 0.0

    def test_return_type_and_bounds(self):
        attrs = _make_protected_attrs()
        result = regional_resilience(np.array([0, 1, 2, 3, 4, 5]), attrs)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestComputeAllDomainMetrics:
    def test_returns_all_keys(self):
        features = _make_item_features()
        attrs = _make_protected_attrs()
        ranking = np.array([0, 1, 2, 3, 4, 5])
        result = compute_all_domain_metrics(ranking, features, attrs)
        assert set(result.keys()) == {"hhi", "cert_coverage", "regional_resilience"}

    def test_all_values_are_floats_in_range(self):
        features = _make_item_features()
        attrs = _make_protected_attrs()
        ranking = np.array([0, 1, 2, 3, 4, 5])
        result = compute_all_domain_metrics(ranking, features, attrs)
        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not a float"
            assert 0.0 <= value <= 1.0, f"{key}={value} out of [0, 1]"

    def test_top_k_propagated(self):
        """top_k passed to the aggregator should be honoured by every sub-metric."""
        features = _make_item_features()
        attrs = _make_protected_attrs()
        ranking = np.array([0, 1, 2, 3, 4, 5])
        result_full = compute_all_domain_metrics(ranking, features, attrs)
        result_top1 = compute_all_domain_metrics(ranking, features, attrs, top_k=1)
        # HHI for a single item is always 1.0 (one group, share=1)
        assert result_top1["hhi"] == pytest.approx(1.0)
        # cert_coverage with top_k=1 should be <= full coverage
        assert result_top1["cert_coverage"] <= result_full["cert_coverage"]
        # regional_resilience with one item is always 0.0
        assert result_top1["regional_resilience"] == pytest.approx(0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
