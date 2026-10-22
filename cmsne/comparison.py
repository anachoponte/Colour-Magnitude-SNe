"""Research-comparison utilities for the sequential CMD classifier design.

See docs/superpowers/specs/2026-10-22-sequential-cmd-classifier-design.md,
section D. Benchmarks the trajectory classifier against the existing static
ColourClassifier, and against a deliberately wrong naive-independent-
multiplication baseline, so the comparison -- not an a-priori choice --
decides which approach the project keeps.
"""

import numpy as np


def naive_sequential_posterior(step_probabilities):
    """Chain per-step posteriors by multiplying odds -- the WRONG baseline.

    This treats each grid step's probability as independent evidence and
    combines them via a naive-Bayes odds product. It is included only to
    quantify how badly that assumption overstates confidence relative to the
    jointly trained :class:`cmsne.trajectory.TrajectoryClassifier`
    (resolution #2 in the design spec) -- never use this to make a real
    classification decision.

    :param step_probabilities: array, shape ``(n_events, n_steps)``, each
        entry a per-step P(lensed) in (0, 1); NaN where that step has no
        observation yet (skipped, i.e. treated as uninformative odds=1).
    :return: array of shape ``(n_events,)``, the naively combined posterior.
    """
    p = np.asarray(step_probabilities, dtype=float)
    valid = np.isfinite(p)
    p_clipped = np.clip(p, 1e-12, 1 - 1e-12)

    log_odds_each = np.zeros_like(p_clipped)
    log_odds_each[valid] = np.log(p_clipped[valid] / (1 - p_clipped[valid]))

    combined_log_odds = np.sum(log_odds_each, axis=-1)
    combined_odds = np.exp(combined_log_odds)
    return combined_odds / (1 + combined_odds)
