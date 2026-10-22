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


def compare_classifiers(static_clf, trajectory_clf, signal_trajectories, background_trajectories,
                        step_colour_key='gr', target_fpr=0.10):
    """Benchmark the static, trajectory, and naive-multiplication approaches
    on the same held-out trajectory events (design spec section D).

    .. note::
       ``signal_trajectories``/``background_trajectories`` must already
       expose BOTH schemas -- the static classifier's two-point keys
       (``gr``, ``peakmag``) and the full trajectory keys (``gr_d0``, ...,
       ``peakmag_d60``) -- on the same events. Use
       :class:`cmsne.trajectory.DualSchemaGenerator` (or
       ``scripts/run_dual_schema.py`` for a BlueBEAR-scale production run)
       to generate such events from a real OpSim cadence; this module's own
       unit test builds a smaller synthetic version of the same schema.

    :param static_clf: a fitted :class:`cmsne.classifier.ColourClassifier`
        (the existing two-point method, unchanged).
    :param trajectory_clf: a fitted
        :class:`cmsne.trajectory.TrajectoryClassifier`.
    :param signal_trajectories, background_trajectories: held-out event
        dicts exposing both the static classifier's keys (``gr``, ...,
        ``peakmag``) and the trajectory schema (``gr_d0``, ...).
    :param step_colour_key: which adjacent colour's per-step probabilities
        feed the naive baseline (e.g. ``'gr'`` -> ``gr_d0``, ``gr_d3``, ...).
    :return: dict of headline metrics: ``static_recovery``,
        ``trajectory_recovery`` -- each at *that classifier's own* fitted
        threshold (set when ``static_clf``/``trajectory_clf`` were
        constructed and fit, not re-pinned to this call's ``target_fpr``;
        pass classifiers already fit at the contamination you want to
        compare at) -- and ``naive_vs_trajectory_mean_abs_diff`` (how much
        the naive baseline's posterior diverges from the jointly trained
        classifier's -- the quantified overconfidence from resolution #2).
        ``target_fpr`` itself only governs the per-step single-colour
        classifiers fit internally to build the naive baseline's inputs.
    """
    from .trajectory import TRAJECTORY_GRID, _step_suffix
    from .classifier import ColourClassifier

    static_recovery = static_clf.recovery_rate(signal_trajectories)
    trajectory_recovery = trajectory_clf.recovery_rate(signal_trajectories)

    # Per-step posteriors for the naive baseline: fit one single-colour
    # classifier per grid step (same construct-then-override pattern as
    # cmsne.contours), score each signal event, then combine naively.
    step_probs_sig = np.full((len(signal_trajectories), len(TRAJECTORY_GRID)), np.nan)
    for i, day in enumerate(TRAJECTORY_GRID):
        suffix = _step_suffix(day)
        colour_key, magnitude_key = f'{step_colour_key}{suffix}', f'peakmag{suffix}'
        step_clf = ColourClassifier(keys=[colour_key], include_peakmag=False, target_fpr=target_fpr)
        step_clf.magnitude_feature = magnitude_key
        step_clf.extra = (magnitude_key,)
        step_clf.fit(signal_trajectories, background_trajectories)
        step_probs_sig[:, i] = step_clf.probability(signal_trajectories)

    naive_posterior_sig = naive_sequential_posterior(step_probs_sig)
    trajectory_posterior_sig = trajectory_clf.probability(signal_trajectories)
    naive_vs_trajectory_mean_abs_diff = float(
        np.mean(np.abs(naive_posterior_sig - trajectory_posterior_sig)))

    return {
        'static_recovery': static_recovery,
        'trajectory_recovery': trajectory_recovery,
        'naive_vs_trajectory_mean_abs_diff': naive_vs_trajectory_mean_abs_diff,
    }
