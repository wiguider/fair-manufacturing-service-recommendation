"""Domain-specific manufacturing metrics for supply chain diversity and resilience.

These metrics complement fairness metrics by capturing structural properties of
recommended supplier portfolios from a procurement and supply-chain-risk perspective.

References:
- HHI: Herfindahl & Hirschman (1950) — market concentration index
- Entropy-based resilience: Scholten & Scott (2005), Leat & Revoredo-Giha (2013)
"""

import numpy as np
import pandas as pd


def supply_chain_hhi(
    ranking: np.ndarray,
    item_features: pd.DataFrame,
    top_k: int = None,
    group_col: str = "size_group",
) -> float:
    """Herfindahl-Hirschman Index (HHI) of recommended manufacturers.

    Measures supplier concentration across company-size groups (or another
    categorical dimension). HHI is the sum of squared share fractions:

        HHI = sum_g (s_g)^2,   where s_g = count(g) / len(top_k)

    A value of 1.0 means all recommended items belong to a single group
    (maximum concentration / single point of supply-chain failure). A value of
    1/G means perfectly equal spread across G groups (minimum concentration).

    Args:
        ranking: Ordered array of item_ids, most relevant first.
        item_features: DataFrame with at least columns ``item_id`` and
            ``group_col``.
        top_k: If provided, only the first ``top_k`` items are considered.
        group_col: Column in ``item_features`` used for grouping; defaults to
            ``"size_group"``.

    Returns:
        HHI score in [0, 1]. Lower = more diversified supply chain.
    """
    if top_k is not None:
        ranking = ranking[:top_k]

    k = len(ranking)
    if k == 0:
        return 0.0

    if group_col not in item_features.columns:
        # Cannot compute without grouping column — return worst case
        return 1.0

    group_map: dict = dict(zip(item_features["item_id"], item_features[group_col]))
    groups = [group_map.get(iid, "unknown") for iid in ranking]

    counts = pd.Series(groups).value_counts()
    shares = counts / k
    hhi: float = float((shares ** 2).sum())
    return hhi


def certification_coverage(
    ranking: np.ndarray,
    item_features: pd.DataFrame,
    top_k: int = None,
    cert_col: str = "certifications",
    all_certs: list[str] | None = None,
) -> float:
    """Fraction of known certifications covered by the top-k recommendations.

    Certifications are expected to be stored as a whitespace-separated string
    in ``item_features[cert_col]`` (e.g. ``"ISO9001 AS9100 NADCAP"``).

    If ``all_certs`` is not supplied the universe is inferred from
    ``item_features`` itself (i.e. all certifications that appear anywhere in
    the catalogue).

    Args:
        ranking: Ordered array of item_ids, most relevant first.
        item_features: DataFrame with at least columns ``item_id`` and
            ``cert_col``.
        top_k: If provided, only the first ``top_k`` items are considered.
        cert_col: Column in ``item_features`` that holds certifications.
        all_certs: Explicit universe of certifications to use as denominator.
            When ``None`` the full catalogue is used.

    Returns:
        Coverage ratio in [0, 1]. Higher = more capability diversity.
    """
    if top_k is not None:
        ranking = ranking[:top_k]

    if len(ranking) == 0:
        return 0.0

    if cert_col not in item_features.columns:
        return 0.0

    def _parse_certs(val) -> set[str]:
        if pd.isna(val) or str(val).strip() == "":
            return set()
        return {c.strip() for c in str(val).split() if c.strip()}

    cert_map: dict = {
        row["item_id"]: _parse_certs(row[cert_col])
        for _, row in item_features.iterrows()
    }

    # Universe of certifications
    if all_certs is None:
        universe: set[str] = set()
        for certs in cert_map.values():
            universe |= certs
    else:
        universe = set(all_certs)

    if not universe:
        return 0.0

    covered: set[str] = set()
    for iid in ranking:
        covered |= cert_map.get(iid, set())

    return len(covered & universe) / len(universe)


def regional_resilience(
    ranking: np.ndarray,
    protected_attrs: pd.DataFrame,
    top_k: int = None,
    geo_col: str = "geo_group",
) -> float:
    """Normalised geographic entropy of the recommended supplier list.

    Shannon entropy over geographic regions, normalised by the maximum
    possible entropy log2(G) where G is the number of distinct regions
    observed in the ranking:

        H = -sum_g p_g * log2(p_g)
        score = H / log2(G)

    A score of 1.0 means suppliers are spread perfectly evenly across all
    observed regions (maximum resilience to regional disruptions). A score of
    0.0 means all recommended suppliers are from a single region (no resilience).

    Args:
        ranking: Ordered array of item_ids, most relevant first.
        protected_attrs: DataFrame with at least columns ``item_id`` and
            ``geo_col``.
        top_k: If provided, only the first ``top_k`` items are considered.
        geo_col: Column in ``protected_attrs`` used for region labels.

    Returns:
        Normalised geographic entropy in [0, 1]. Higher = better resilience.
    """
    if top_k is not None:
        ranking = ranking[:top_k]

    k = len(ranking)
    if k == 0:
        return 0.0

    if geo_col not in protected_attrs.columns:
        return 0.0

    geo_map: dict = dict(zip(protected_attrs["item_id"], protected_attrs[geo_col]))
    regions = [geo_map.get(iid, "unknown") for iid in ranking]

    counts = pd.Series(regions).value_counts()
    num_regions = len(counts)

    if num_regions == 1:
        return 0.0  # All from one region — zero entropy

    probs = counts / k
    entropy: float = float(-(probs * np.log2(probs)).sum())
    max_entropy: float = np.log2(num_regions)

    return entropy / max_entropy


def compute_all_domain_metrics(
    ranking: np.ndarray,
    item_features: pd.DataFrame,
    protected_attrs: pd.DataFrame,
    top_k: int = None,
) -> dict[str, float]:
    """Compute all domain-specific manufacturing metrics at once.

    Args:
        ranking: Ordered array of item_ids, most relevant first.
        item_features: Manufacturer feature DataFrame (``item_id``,
            ``size_group``, ``certifications``, …).
        protected_attrs: Protected-attribute DataFrame (``item_id``,
            ``size_group``, ``geo_group``).
        top_k: Evaluation cut-off applied uniformly to every metric.

    Returns:
        Dictionary with keys ``"hhi"``, ``"cert_coverage"``,
        ``"regional_resilience"``.
    """
    return {
        "hhi": supply_chain_hhi(ranking, item_features, top_k),
        "cert_coverage": certification_coverage(ranking, item_features, top_k),
        "regional_resilience": regional_resilience(ranking, protected_attrs, top_k),
    }
