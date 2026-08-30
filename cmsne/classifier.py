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

from .colour_magnitude import weighted_quantile

# Adjacent ugrizy colours, matching cmsne.multicolour's feature names.
COLOUR_KEYS = ['ug', 'gr', 'ri', 'iz', 'zy']


def colour_feature_matrix(events, keys=COLOUR_KEYS, extra=('peakmag',)):
    """Stack ``events`` (dicts) into a feature matrix over ``keys`` (+ ``extra``).

    Missing entries become NaN, which the classifier treats as "band not observed".

    :return: ``(X, columns)`` where ``columns`` is the ordered feature name list.
    """
    columns = list(keys) + [k for k in extra if k]
    X = np.array([[float(e.get(c, np.nan)) if e.get(c, None) is not None else np.nan
                   for c in columns] for e in events], dtype=float)
    return X, columns


class ColourClassifier:
    """Gradient-boosted colour classifier with a contamination-pinned threshold.

    :param keys: colour feature names to use.
    :param include_peakmag: also use the (apparent) peak magnitude as a feature.
    :param target_fpr: contamination (false-positive rate) the decision threshold
        is set to, on the rate-weighted background.
    :param hgb: extra keyword args forwarded to ``HistGradientBoostingClassifier``.
    """

    def __init__(self, keys=COLOUR_KEYS, include_peakmag=True, target_fpr=0.10, **hgb):
        self.keys = list(keys)
        self.extra = ('peakmag',) if include_peakmag else ()
        self.target_fpr = target_fpr
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
        self.clf = HistGradientBoostingClassifier(**self.params).fit(X, y, sample_weight=w)

        sb = self.clf.predict_proba(Xb)[:, 1]
        self.threshold = float(weighted_quantile(sb, wb, q=1.0 - self.target_fpr))
        return self

    def score(self, events):
        """Probability each event is a lensed SN Ia."""
        X, _ = colour_feature_matrix(events, self.keys, self.extra)
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, events):
        """Boolean: score at or above the contamination-pinned threshold."""
        return self.score(events) >= self.threshold

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
