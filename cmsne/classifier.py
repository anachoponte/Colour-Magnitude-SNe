"""Flexible colour-only classifier for the lensed-SN Ia identifier.

The historical method draws a single straight line in one colour vs magnitude
(:func:`cmsne.colour_magnitude.exponential_regression` + :func:`success_rate`).
The boundary + multi-colour studies showed that (a) a flexible, discontinuous
boundary beats the line and (b) using the whole ugrizy colour vector, rather than
one colour, is what carries the separation once a real cadence limits which bands
you have. :class:`ColourClassifier` packages both: a gradient-boosted tree on the
colour vector, NaN-native so missing bands are handled natively, with a decision
threshold pinned to a target contamination.

It is deliberately generic over the event dicts the generators emit: pass the
feature keys you have. Multi-colour events (:class:`cmsne.multicolour`) expose
``ug gr ri iz zy`` (+ ``peakmag``); the single-colour colour-magnitude events
expose ``lensed_colour`` and ``band_1_mag_l``.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve as _sklearn_roc_curve

from .colour_magnitude import weighted_quantile

# Adjacent ugrizy colours, matching cmsne.multicolour's feature names.
COLOUR_KEYS = ['ug', 'gr', 'ri', 'iz', 'zy']

# Brightness features recorded per event by cmsne.multicolour.event_multicolour:
#   'peakmag'          -- brightest observed sample across all bands;
#   'peakmag_fit'      -- parabola-fitted light-curve peak in the best-sampled band;
#   'peakmag_firstdet' -- magnitude of the earliest detection.
# The comparison in docs/methods_and_results.md (§3.8) found brightest-observed gives
# the highest recovery (it beats a single-band fitted peak and the first-detected
# magnitude), so it is the explicit default magnitude feature.
MAGNITUDE_FEATURES = ('peakmag', 'peakmag_fit', 'peakmag_firstdet')
DEFAULT_MAGNITUDE_FEATURE = 'peakmag'   # brightest-observed sample across all bands


def colour_feature_matrix(events, keys=COLOUR_KEYS, extra=(DEFAULT_MAGNITUDE_FEATURE,)):
    """Stack ``events`` (dicts) into a feature matrix over ``keys`` (+ ``extra``).

    Missing entries become NaN, which the classifier treats as "band not observed".

    :return: ``(X, columns)`` where ``columns`` is the ordered feature name list.
    """
    columns = list(keys) + [k for k in extra if k]
    X = np.array([[float(e.get(c, np.nan)) if e.get(c, None) is not None else np.nan
                   for c in columns] for e in events], dtype=float)
    return X, columns


def _weighted_std(values, weights):
    """Rate-weighted standard deviation of ``values``, over finite,
    non-negative-weight entries only.

    Returns ``1.0`` (a safe, finite fallback) if fewer than 2 such entries
    exist or the weighted variance is non-positive -- this keeps downstream
    bandwidths finite and positive rather than zero or NaN.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    if np.sum(finite) < 2 or np.sum(weights[finite]) <= 0:
        return 1.0
    v, w = values[finite], weights[finite]
    mean = np.sum(v * w) / np.sum(w)
    var = np.sum(w * (v - mean) ** 2) / np.sum(w)
    return float(np.sqrt(var)) if var > 0 else 1.0


def _scott_bandwidths(X, weights):
    """Per-column Gaussian-kernel bandwidth via Scott's rule, over pooled
    ``X``/``weights``.

    :param X: ``(n, d)`` feature matrix, NaN allowed (ignored per-column by
        :func:`_weighted_std`).
    :param weights: ``(n,)`` rate weights, pooled across every class the
        kernel is shared between.
    :return: ``(d,)`` array of positive, finite bandwidths.
    """
    # Local import: importance_sampling -> contours -> classifier would be a cycle at module load.
    from .importance_sampling import effective_sample_size

    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n_eff = effective_sample_size(weights)
    d = X.shape[1]
    scott_factor = n_eff ** (-1.0 / (d + 4)) if n_eff > 0 else 1.0
    stds = np.array([_weighted_std(X[:, j], weights) for j in range(d)])
    return scott_factor * stds


def _kernel_weighted_values(x, X, weights, bandwidths, dims):
    """Per-training-row kernel-weighted value ``w_i * k_i``, over columns
    ``dims`` only.

    :param x: ``(d,)`` query feature vector (full column set; only ``dims``
        entries are read).
    :param X: ``(n, d)`` training feature matrix, already filtered to rows
        that are finite over ``dims`` -- this function does not filter rows
        itself.
    :param weights: ``(n,)`` training rate weights.
    :param bandwidths: ``(d,)`` per-column bandwidths (full column set; only
        ``dims`` entries are read).
    :param dims: 1D int array, which columns to evaluate the kernel over.
    :return: ``(n,)`` array, ``weights[i] * exp(-0.5 * sum_d ((x_d - X[i,d])/h_d)**2)``.
    """
    n = X.shape[0]
    if len(dims) == 0 or n == 0:
        return np.zeros(n)
    xd = np.asarray(x, dtype=float)[dims]
    Xd = X[:, dims]
    hd = np.asarray(bandwidths, dtype=float)[dims]
    z = (Xd - xd) / hd
    k = np.exp(-0.5 * np.sum(z ** 2, axis=1))
    return np.asarray(weights, dtype=float) * k


def _local_density_ratio(x, Xs, ws, Xb, wb, bandwidths, dims):
    """Local weighted kernel density ratio and local ESS at point ``x``.

    :param Xs, ws: signal training feature matrix and rate weights.
    :param Xb, wb: background training feature matrix and rate weights.
    :return: ``(p_kde, local_ess_sig, local_ess_bg)``. ``p_kde`` is ``None``
        when both classes' kernel-weighted mass underflows to (numerically)
        zero at ``x`` -- the degenerate case callers fall back to the
        tree-only posterior for.
    """
    # Local import: importance_sampling -> contours -> classifier would be a cycle at module load.
    from .importance_sampling import effective_sample_size

    gs = _kernel_weighted_values(x, Xs, ws, bandwidths, dims)
    gb = _kernel_weighted_values(x, Xb, wb, bandwidths, dims)
    S, B = float(np.sum(gs)), float(np.sum(gb))
    local_ess_sig = effective_sample_size(gs)
    local_ess_bg = effective_sample_size(gb)
    if S + B < 1e-300:
        return None, local_ess_sig, local_ess_bg
    return S / (S + B), local_ess_sig, local_ess_bg


def _blend_weight(local_ess_sig, local_ess_bg, ess_threshold, log_width):
    """Sigmoid blend weight favouring the tree posterior when both classes
    have ample local support, and the KDE density ratio when either is thin.

    :return: ``alpha`` in ``(0, 1)``; 1.0 means "fully trust the tree".
    """
    m = min(local_ess_sig, local_ess_bg)
    z = (np.log(m + 1e-12) - np.log(ess_threshold)) / log_width
    return float(1.0 / (1.0 + np.exp(-z)))


class ColourClassifier:
    """Gradient-boosted colour classifier with a contamination-pinned threshold.

    :param keys: colour feature names to use.
    :param include_peakmag: also use an apparent-brightness feature (see
        ``magnitude_feature``); set ``False`` for a colour-only classifier.
    :param magnitude_feature: which brightness feature to use when
        ``include_peakmag`` is set. Defaults to ``'peakmag'`` (brightest-observed
        across all bands), the best of the three on test (see
        :data:`MAGNITUDE_FEATURES` and docs/methods_and_results.md §3.8).
    :param target_fpr: contamination (false-positive rate) the optional hard
        decision threshold is set to, on the rate-weighted background. The boundary
        between lensed and unlensed is a broad gradient, not a sharp line, so prefer
        :meth:`probability` / :meth:`rank` over the hard :meth:`predict`.
    :param calibrate: wrap the tree in isotonic probability calibration so
        :meth:`probability` returns a well-behaved posterior (at the balanced
        training prior) rather than a raw, uncalibrated tree score.
    :param hgb: extra keyword args forwarded to ``HistGradientBoostingClassifier``.
    """

    def __init__(self, keys=COLOUR_KEYS, include_peakmag=True,
                 magnitude_feature=DEFAULT_MAGNITUDE_FEATURE, target_fpr=0.10,
                 calibrate=True, **hgb):
        self.keys = list(keys)
        if include_peakmag and magnitude_feature:
            if magnitude_feature not in MAGNITUDE_FEATURES:
                raise ValueError("magnitude_feature must be one of %r, got %r"
                                 % (MAGNITUDE_FEATURES, magnitude_feature))
            self.magnitude_feature = magnitude_feature
            self.extra = (magnitude_feature,)
        else:
            self.magnitude_feature = None
            self.extra = ()
        self.target_fpr = target_fpr
        self.calibrate = calibrate
        self.params = dict(max_depth=4, learning_rate=0.1, max_iter=300,
                           l2_regularization=1.0, random_state=0)
        self.params.update(hgb)

    def fit(self, signal_events, background_events,
            signal_weights=None, background_weights=None):
        """Train on lensed-SN-Ia ``signal_events`` vs contaminant ``background_events``.

        Weights are rate weights per event (see the generators / ``rate_weight``).
        Signal and background totals are balanced so the classifier learns to
        separate rather than exploit the base rate; the background *composition*
        still follows its per-event weights, and the decision threshold is set to
        ``target_fpr`` on that rate-weighted background.
        """
        Xs, cols = colour_feature_matrix(signal_events, self.keys, self.extra)
        Xb, _ = colour_feature_matrix(background_events, self.keys, self.extra)
        self.columns = cols

        ws = np.ones(len(Xs)) if signal_weights is None else np.asarray(signal_weights, float)
        wb = np.ones(len(Xb)) if background_weights is None else np.asarray(background_weights, float)
        if ws.sum() > 0:
            ws = ws * (wb.sum() / ws.sum())            # balance totals

        X = np.vstack([Xs, Xb])
        y = np.r_[np.ones(len(Xs)), np.zeros(len(Xb))]
        w = np.r_[ws, wb]
        # Signal and background totals are balanced, so the trained posterior is at a
        # 0.5 prior; probability(prior=...) rescales it to any other prior.
        self.train_prior = float(ws.sum() / (ws.sum() + wb.sum()))

        base = HistGradientBoostingClassifier(**self.params)
        if self.calibrate:
            self.clf = CalibratedClassifierCV(base, method='isotonic', cv=3).fit(X, y, sample_weight=w)
        else:
            self.clf = base.fit(X, y, sample_weight=w)

        sb = self.clf.predict_proba(Xb)[:, 1]
        self.threshold = float(weighted_quantile(sb, wb, q=1.0 - self.target_fpr))
        return self

    def score(self, events):
        """Raw ranking score (calibrated posterior at the balanced training prior).

        Monotonic with :meth:`probability`, so it gives the same ordering; use it
        when you only need to rank.
        """
        X, _ = colour_feature_matrix(events, self.keys, self.extra)
        return self.clf.predict_proba(X)[:, 1]

    def probability(self, events, prior=None):
        """Posterior probability P(lensed SN Ia) for each event.

        The boundary is a gradient, not a line — this is the recommended output.
        With ``prior=None`` it returns the calibrated posterior at the balanced
        training prior (good for ranking and for reading off "how lensed-like").
        Pass the **true** lensed-SN-Ia prior (e.g. ``1e-3``) to get the honest
        posterior for a real stream, which correctly comes out small — see the
        base-rate discussion in the results write-up.

        :param prior: true fraction of lensed SN Ia among the transients scored, or
            ``None`` to keep the training prior.
        :return: array of probabilities in [0, 1].
        """
        p = np.clip(self.score(events), 1e-12, 1 - 1e-12)
        if prior is None:
            return p
        p0 = self.train_prior
        # likelihood ratio with the training prior divided out, re-applied at `prior`
        lr = (p / (1 - p)) * ((1 - p0) / p0)
        odds = lr * (prior / (1 - prior))
        return odds / (1 + odds)

    def rank(self, events):
        """Indices of ``events`` ordered most- to least-lensed-like (by score).

        The prioritised follow-up list: work down it until the budget runs out,
        rather than applying a hard cut.
        """
        return np.argsort(self.score(events))[::-1]

    def predict(self, events):
        """Boolean hard cut at the contamination-pinned threshold.

        Kept for convenience, but a hard boundary is the wrong picture for a broad,
        gradual transition — prefer :meth:`probability` / :meth:`rank`.
        """
        return self.score(events) >= self.threshold

    def roc_curve(self, signal_events, background_events,
                  signal_weights=None, background_weights=None):
        """ROC curve sweeping the classifier's score threshold.

        Unlike :meth:`predict`, which only reports recovery/FPR at the one
        ``target_fpr``-pinned threshold set during :meth:`fit`, this sweeps
        every achievable threshold over ``score()`` -- the full sensitivity
        vs contamination trade-off, not a single operating point.

        Weights are used as given (no signal/background rebalancing, unlike
        :meth:`fit`): the real rate-weighted trade-off is what an ROC curve
        should show, not the balanced-training view.

        :return: ``(fpr, tpr, thresholds)`` from ``sklearn.metrics.roc_curve``,
            rate-weighted if weights are given. ``thresholds`` are on the same
            [0, 1] scale as :meth:`score`/:meth:`probability`.
        """
        s_sig = self.score(signal_events)
        s_bg = self.score(background_events)
        y_true = np.r_[np.ones(len(s_sig)), np.zeros(len(s_bg))]
        y_score = np.r_[s_sig, s_bg]
        ws = np.ones(len(s_sig)) if signal_weights is None else np.asarray(signal_weights, float)
        wb = np.ones(len(s_bg)) if background_weights is None else np.asarray(background_weights, float)
        sample_weight = np.r_[ws, wb]
        return _sklearn_roc_curve(y_true, y_score, sample_weight=sample_weight)

    def recovery_rate(self, signal_events, weights=None):
        """Fraction of lensed SN Ia flagged (rate-weighted if ``weights`` given).

        The direct analogue of :func:`cmsne.colour_magnitude.success_rate`, but for
        the multi-colour gradient-boosted boundary rather than a straight line.
        """
        pred = self.predict(signal_events)
        if weights is None:
            return float(np.mean(pred)) if len(pred) else float('nan')
        w = np.asarray(weights, float)
        return float(w[pred].sum() / w.sum()) if w.sum() > 0 else float('nan')
