"""Standard ranking evaluation metrics."""

import numpy as np


def precision_at_k(ranking: np.ndarray, relevant: set, k: int) -> float:
    """Precision@k: fraction of top-k items that are relevant."""
    top_k = ranking[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(ranking: np.ndarray, relevant: set, k: int) -> float:
    """Recall@k: fraction of relevant items found in top-k."""
    if len(relevant) == 0:
        return 0.0
    top_k = ranking[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(ranking: np.ndarray, relevant: set, k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    top_k = ranking[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(top_k)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranking: np.ndarray, relevant: set) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant item."""
    for i, item in enumerate(ranking):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(ranking: np.ndarray, relevant: set) -> float:
    """Average Precision for a single query."""
    if len(relevant) == 0:
        return 0.0
    hits = 0
    ap = 0.0
    for i, item in enumerate(ranking):
        if item in relevant:
            hits += 1
            ap += hits / (i + 1)
    return ap / len(relevant)


def mean_average_precision(rankings: dict[int, np.ndarray], relevant_sets: dict[int, set]) -> float:
    """MAP across all users."""
    aps = []
    for uid, ranking in rankings.items():
        if uid in relevant_sets:
            aps.append(average_precision(ranking, relevant_sets[uid]))
    return np.mean(aps) if aps else 0.0


def evaluate_ranking(
    rankings: dict[int, np.ndarray],
    relevant_sets: dict[int, set],
    ks: list[int] = None,
) -> dict[str, float]:
    """Evaluate all ranking metrics for a set of user rankings."""
    if ks is None:
        ks = [5, 10, 20]

    results = {}
    all_mrr = []
    per_k = {k: {"precision": [], "recall": [], "ndcg": []} for k in ks}

    for uid, ranking in rankings.items():
        rel = relevant_sets.get(uid, set())
        if not rel:
            continue

        all_mrr.append(mrr(ranking, rel))

        for k in ks:
            per_k[k]["precision"].append(precision_at_k(ranking, rel, k))
            per_k[k]["recall"].append(recall_at_k(ranking, rel, k))
            per_k[k]["ndcg"].append(ndcg_at_k(ranking, rel, k))

    results["mrr"] = np.mean(all_mrr) if all_mrr else 0.0
    results["map"] = mean_average_precision(rankings, relevant_sets)

    for k in ks:
        for metric in ["precision", "recall", "ndcg"]:
            vals = per_k[k][metric]
            results[f"{metric}@{k}"] = np.mean(vals) if vals else 0.0

    return results
