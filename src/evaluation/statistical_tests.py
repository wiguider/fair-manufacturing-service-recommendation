"""Statistical tests for comparing recommendation systems."""

import numpy as np
from scipy import stats


def paired_t_test(scores_a: np.ndarray, scores_b: np.ndarray, alpha: float = 0.05) -> dict:
    """Paired t-test between two systems' per-user scores."""
    stat, pvalue = stats.ttest_rel(scores_a, scores_b)
    return {
        "statistic": stat,
        "p_value": pvalue,
        "significant": pvalue < alpha,
        "effect_size": (scores_a.mean() - scores_b.mean()) / np.sqrt(
            (scores_a.std() ** 2 + scores_b.std() ** 2) / 2 + 1e-10
        ),
    }


def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray, alpha: float = 0.05) -> dict:
    """Wilcoxon signed-rank test (non-parametric)."""
    diff = scores_a - scores_b
    # Filter zero differences
    nonzero = diff != 0
    if nonzero.sum() < 10:
        return {"statistic": 0, "p_value": 1.0, "significant": False}
    stat, pvalue = stats.wilcoxon(scores_a[nonzero], scores_b[nonzero])
    return {
        "statistic": stat,
        "p_value": pvalue,
        "significant": pvalue < alpha,
    }


def bootstrap_ci(
    scores: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    n = len(scores)
    means = np.array([rng.choice(scores, size=n, replace=True).mean() for _ in range(n_bootstrap)])
    alpha = (1 - confidence) / 2
    lo = np.percentile(means, alpha * 100)
    hi = np.percentile(means, (1 - alpha) * 100)
    return scores.mean(), lo, hi


def compare_systems(
    per_user_scores: dict[str, dict[str, np.ndarray]],
    baseline_name: str,
    alpha: float = 0.05,
) -> dict:
    """Compare all systems against a baseline.

    Args:
        per_user_scores: {system_name: {metric_name: np.array of per-user scores}}
        baseline_name: Name of the baseline system.
        alpha: Significance level.

    Returns:
        Dict of comparison results.
    """
    baseline = per_user_scores[baseline_name]
    results = {}

    for sys_name, sys_scores in per_user_scores.items():
        if sys_name == baseline_name:
            continue
        results[sys_name] = {}
        for metric_name in sys_scores:
            if metric_name not in baseline:
                continue
            a = sys_scores[metric_name]
            b = baseline[metric_name]
            # Ensure same length
            min_len = min(len(a), len(b))
            a, b = a[:min_len], b[:min_len]

            results[sys_name][metric_name] = {
                "wilcoxon": wilcoxon_test(a, b, alpha),
                "t_test": paired_t_test(a, b, alpha),
                "ci": bootstrap_ci(a - b),
            }

    return results
