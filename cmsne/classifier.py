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

import hashlib
from typing import NamedTuple

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

# Local density-ratio blend defaults (see docs/superpowers/specs/
# 2026-10-25-local-density-posterior-blend-design.md). These are provisional
# starting points, not empirically final -- real tuning against production
# data is a manual-validation step, not an automated test.
DEFAULT_ESS_THRESHOLD = 10.0
DEFAULT_LOG_WIDTH = 0.3
DEFAULT_KDE_MAX_EVENTS = 4000
DEFAULT_KDE_MIN_ESS = 10.0
# Uncertainty-aware noise injection (see docs/superpowers/specs/
# 2026-10-26-uncertainty-aware-noise-injection-design.md). Provisional, like
# the other blend constants above -- tuned against real data in this plan's
# own Manual Validation step.
DEFAULT_N_UNCERTAINTY_REPLICATES = 8


def colour_feature_matrix(events, keys=COLOUR_KEYS, extra=(DEFAULT_MAGNITUDE_FEATURE,)):
    """Stack ``events`` (dicts) into a feature matrix over ``keys`` (+ ``extra``).

    Missing entries become NaN, which the classifier treats as "band not observed".

    :return: ``(X, columns)`` where ``columns`` is the ordered feature name list.
    """
    columns = list(keys) + [k for k in extra if k]
    X = np.array([[float(e.get(c, np.nan)) if e.get(c, None) is not None else np.nan
                   for c in columns] for e in events], dtype=float)
    return X, columns


def colour_feature_sigma_matrix(events, keys=COLOUR_KEYS, extra=(DEFAULT_MAGNITUDE_FEATURE,)):
    """Stack ``events``' per-feature measurement uncertainty (``f'{c}_err'``)
    into a matrix matching :func:`colour_feature_matrix`'s column order.

    Missing/absent ``_err`` entries become NaN -- :func:`_replicate_with_uncertainty`
    treats this as "nothing to perturb with", not an error. ``peakmag_fit`` in
    particular has no ``_err`` companion anywhere in this codebase (see
    docs/superpowers/specs/2026-10-26-uncertainty-aware-noise-injection-design.md)
    and will always read NaN here.

    :return: ``(n, d)`` sigma matrix, same column order as :func:`colour_feature_matrix`.
    """
    columns = list(keys) + [k for k in extra if k]
    sigma = np.array([[float(e.get(f'{c}_err', np.nan)) if e.get(f'{c}_err', None) is not None else np.nan
                        for c in columns] for e in events], dtype=float)
    return sigma


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


def _row_content_seed(base_seed, x_row, sigma_row):
    """Deterministic seed derived from a row's OWN feature + sigma values (not
    from its position in any particular array/batch), so replicating the same
    logical event always draws the same noise regardless of what else shares
    the call -- see :func:`_replicate_with_uncertainty`.
    """
    payload = np.concatenate([[float(base_seed)], np.nan_to_num(x_row, nan=-1e18),
                               np.nan_to_num(sigma_row, nan=-1e18)]).astype(np.float64)
    digest = hashlib.sha256(payload.tobytes()).digest()
    return int.from_bytes(digest[:8], 'little') % (2 ** 32)


def _replicate_with_uncertainty(X, sigma, weights, dims, n_replicates, seed=0):
    """Expand each row of ``X`` into ``n_replicates`` noisy copies, perturbed
    per dimension in ``dims`` by that row's own ``sigma`` (non-finite or
    non-positive sigma leaves that dimension exactly as-is for every
    replica of that row -- nothing to perturb with is not an error). Each
    replica carries ``weights / n_replicates``, so summed weight mass per
    original row is preserved exactly.

    Each row draws its noise from an independent RNG stream seeded from
    ``(seed, that row's own X, that row's own sigma)`` -- NOT from its
    position in ``X`` or from a single stream shared across every row in the
    call. This guarantees the same logical event (same features, same
    uncertainty) always replicates to the same values regardless of what
    else shares the call -- e.g. ``score([a, b])[1]`` and ``score([b])[0]``
    must agree for the same event ``b``, which a single shared RNG stream
    cannot guarantee (a row's draws would otherwise depend on how many other
    perturbable rows preceded it in that specific call).

    :return: ``(X_rep, w_rep, event_id)``. ``event_id[k]`` is the index into
        the ORIGINAL ``X``/``weights`` that ``X_rep[k]`` came from -- needed
        to group replicas back together (e.g. for
        :func:`_grouped_effective_sample_size`).

    If every entry of ``sigma[:, dims]`` is non-finite or non-positive (no
    real uncertainty data available), or ``n_replicates <= 1``, or ``dims``
    is empty, returns ``(X, weights, np.arange(len(X)))`` unchanged -- zero
    extra compute, exact backward compatibility with any population/fixture
    lacking uncertainty fields.
    """
    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    dims = np.asarray(dims, dtype=int)
    if n_replicates <= 1 or len(dims) == 0 or not np.any(np.isfinite(sigma[:, dims]) & (sigma[:, dims] > 0)):
        return X, weights, np.arange(len(X))

    n = len(X)
    X_rep = np.repeat(X, n_replicates, axis=0)
    event_id = np.repeat(np.arange(n), n_replicates)
    for i in range(n):
        row_sigma = sigma[i, dims]
        perturbable_dims = dims[np.isfinite(row_sigma) & (row_sigma > 0)]
        if len(perturbable_dims) == 0:
            continue
        rng = np.random.default_rng(_row_content_seed(seed, X[i], sigma[i]))
        sl = slice(i * n_replicates, (i + 1) * n_replicates)
        for d in perturbable_dims:
            X_rep[sl, d] += rng.normal(0.0, sigma[i, d], size=n_replicates)
    w_rep = np.repeat(weights, n_replicates) / n_replicates
    return X_rep, w_rep, event_id


def _grouped_effective_sample_size(g, event_id, n_original):
    """Kish's effective sample size computed AFTER summing each original
    event's noise-injection replicas' values back together, so
    ``n_replicates`` correlated copies of one real observation contribute
    ESS ~1, not ~n_replicates (see docs/superpowers/specs/
    2026-10-26-uncertainty-aware-noise-injection-design.md). Reduces to a
    plain :func:`~cmsne.importance_sampling.effective_sample_size` call
    when ``event_id`` is trivial (``np.arange(n_original)``, i.e. no
    replication occurred).
    """
    # Local import: importance_sampling -> contours -> classifier would be a cycle at module load.
    from .importance_sampling import effective_sample_size

    g_grouped = np.bincount(event_id, weights=g, minlength=n_original)
    return effective_sample_size(g_grouped)


def _widened_kernel_values(x, X, weights, bandwidths, dims, min_ess, growth=2.0, max_iter=20,
                            event_id=None, n_original=None):
    """Kernel-weighted values at ``x``, with the kernel's bandwidth
    isotropically widened (by ``growth`` per iteration, up to ``max_iter``
    times) until the resulting local ESS reaches ``min_ess``, or until
    widening stops helping (plateaus at this class's own overall ESS
    ceiling) -- whichever comes first.

    This exists ONLY to make the KDE side's own ``S``/``B`` sums
    statistically meaningful in genuinely data-starved regions, where a
    fixed bandwidth's local ESS can be dominated by a single surviving
    training point (see docs/superpowers/specs/
    2026-10-26-adaptive-kde-bandwidth-design.md). It is NOT used for the
    tree-vs-KDE blend-weight decision, which stays on the fixed-bandwidth
    ESS computed separately in :func:`_local_density_ratio`.

    :param event_id, n_original: if ``X``/``weights`` hold noise-injection
        replicas of the same original training events (see
        docs/superpowers/specs/2026-10-26-uncertainty-aware-noise-injection-design.md),
        pass the replica-to-original mapping here so ESS is computed on
        each original event's GROUPED contribution, not per replica --
        otherwise ``n_replicates`` copies of one real point would look
        like ``n_replicates`` independent points. ``None`` (the default,
        used by every pre-existing direct caller) means no replication is
        in play: a trivial, non-grouping mapping is used.
    :return: ``(g, ess)`` -- the kernel-weighted values and the local ESS
        at whichever scale the widening stopped at.
    """
    if event_id is None:
        event_id = np.arange(len(weights))
    if n_original is None:
        n_original = len(weights)

    scale = 1.0
    g = _kernel_weighted_values(x, X, weights, bandwidths, dims)
    ess = _grouped_effective_sample_size(g, event_id, n_original)
    if ess >= min_ess:
        return g, ess
    ceiling = _grouped_effective_sample_size(weights, event_id, n_original)
    for _ in range(max_iter):
        if ess >= min_ess or ess >= ceiling - 1e-9:
            break
        scale *= growth
        g_new = _kernel_weighted_values(x, X, weights, bandwidths * scale, dims)
        ess_new = _grouped_effective_sample_size(g_new, event_id, n_original)
        if ess_new <= ess:
            ess = ess_new
            break
        g, ess = g_new, ess_new
    return g, ess


def _local_density_ratio(x, Xs, ws, Xb, wb, bandwidths, dims, min_kde_ess=DEFAULT_KDE_MIN_ESS,
                          ess_ceiling_sig=None, ess_ceiling_bg=None,
                          event_id_s=None, event_id_b=None, n_orig_s=None, n_orig_b=None):
    """Local weighted kernel density ratio and local ESS at point ``x``.

    :param Xs, ws: signal training feature matrix and rate weights -- may
        hold noise-injection replicas of fewer original events (see
        docs/superpowers/specs/2026-10-26-uncertainty-aware-noise-injection-design.md).
    :param Xb, wb: background training feature matrix and rate weights, same caveat.
    :param min_kde_ess: minimum effective sample size each class's kernel is
        adaptively widened to reach before computing the density ratio (see
        :func:`_widened_kernel_values`). This does NOT affect
        ``local_ess_sig``/``local_ess_bg`` below, which stay at the fixed
        bandwidth and drive the tree-vs-KDE blend-weight decision. A class
        is skipped from widening entirely (its fixed-bandwidth value is used
        for ``p_kde`` unchanged) when that class's own overall ESS ceiling
        -- ``effective_sample_size`` of its raw training weights, ignoring
        distance -- is already below ``min_kde_ess``: widening such a class
        would only swallow its whole (small) population regardless of true
        distance from ``x``, destroying locality rather than recovering it.
    :param ess_ceiling_sig, ess_ceiling_bg: precomputed
        ``effective_sample_size(ws)``/``effective_sample_size(wb)``, for
        callers that query the same ``(Xs, ws, Xb, wb)`` repeatedly (e.g.
        :meth:`ColourClassifier._blend_with_local_density`, once per event)
        and already have these cached -- they depend only on ``ws``/``wb``,
        never on ``x``, so recomputing them on every call is wasted work at
        this project's training-set scale. ``None`` (the default, used by
        one-off/direct callers) computes them here instead.
    :param event_id_s, event_id_b, n_orig_s, n_orig_b: replica-to-original
        mapping for ``Xs``/``Xb`` respectively (see
        :func:`_widened_kernel_values`). ``None`` (the default, used by
        every pre-existing direct caller) means no replication is in play.
    :return: ``(p_kde, local_ess_sig, local_ess_bg)``. ``p_kde`` is ``None``
        when both classes' FIXED-bandwidth kernel-weighted mass underflows
        to (numerically) zero at ``x`` -- the degenerate case callers fall
        back to the tree-only posterior for. This guard is checked before
        any widening is attempted, so a point with no real local
        information from either class is never "rescued" into a
        near-arbitrary ratio by widening far enough to reach unrelated data.
    """
    if event_id_s is None:
        event_id_s = np.arange(len(ws))
    if n_orig_s is None:
        n_orig_s = len(ws)
    if event_id_b is None:
        event_id_b = np.arange(len(wb))
    if n_orig_b is None:
        n_orig_b = len(wb)

    gs_fixed = _kernel_weighted_values(x, Xs, ws, bandwidths, dims)
    gb_fixed = _kernel_weighted_values(x, Xb, wb, bandwidths, dims)
    local_ess_sig = _grouped_effective_sample_size(gs_fixed, event_id_s, n_orig_s)
    local_ess_bg = _grouped_effective_sample_size(gb_fixed, event_id_b, n_orig_b)
    if float(np.sum(gs_fixed)) + float(np.sum(gb_fixed)) < 1e-300:
        return None, local_ess_sig, local_ess_bg

    # Only widen a class if doing so can plausibly reach min_kde_ess -- if a
    # class's own overall ESS ceiling is already below that target, widening
    # toward it only swallows the class's entire (small) population
    # regardless of true distance, destroying locality rather than
    # recovering it. Use the fixed-bandwidth value unchanged instead.
    if ess_ceiling_sig is None:
        ess_ceiling_sig = _grouped_effective_sample_size(ws, event_id_s, n_orig_s)
    if ess_ceiling_bg is None:
        ess_ceiling_bg = _grouped_effective_sample_size(wb, event_id_b, n_orig_b)
    if ess_ceiling_sig >= min_kde_ess:
        gs, _ = _widened_kernel_values(x, Xs, ws, bandwidths, dims, min_kde_ess,
                                        event_id=event_id_s, n_original=n_orig_s)
    else:
        gs = gs_fixed
    if ess_ceiling_bg >= min_kde_ess:
        gb, _ = _widened_kernel_values(x, Xb, wb, bandwidths, dims, min_kde_ess,
                                        event_id=event_id_b, n_original=n_orig_b)
    else:
        gb = gb_fixed

    S, B = float(np.sum(gs)), float(np.sum(gb))
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


def _finite_pattern(x):
    """Boolean tuple: which entries of feature vector ``x`` are finite."""
    return tuple(bool(v) for v in np.isfinite(np.asarray(x, dtype=float)))


def _weighted_subsample(X, weights, max_events, seed=0):
    """Rate-weighted subsample of up to ``max_events`` positive-weight rows
    of ``X``/``weights``, without replacement, rescaled to preserve total
    weight mass.

    Zero-weight rows are dropped unconditionally -- they contribute nothing
    to the kernel-weighted sums this feeds and carry no rate mass to
    preserve. Matches the weighted-subsampling convention already used
    elsewhere in this project (``scripts/plot_evolving_boundaries.py``'s
    ``_draw_resampled_sources``), adapted to sample *without* replacement
    since this becomes the KDE's own reference set, where duplicate points
    would distort local density estimates.

    :return: ``(X_sub, weights_sub, chosen)``. ``chosen`` indexes into the
        INPUT ``X``/``weights`` -- needed by callers that must align another
        per-row array (e.g. per-event measurement-uncertainty sigma, see
        docs/superpowers/specs/2026-10-26-uncertainty-aware-noise-injection-design.md)
        with the same kept subsample. If there are ``max_events`` or fewer
        positive-weight rows, returns all of them (no subsampling), with
        weights unchanged.
    """
    weights = np.asarray(weights, dtype=float)
    positive = np.where(weights > 0)[0]
    if len(positive) <= max_events:
        return X[positive], weights[positive], positive
    wsum = weights[positive].sum()
    p = weights[positive] / wsum
    rng = np.random.default_rng(seed)
    chosen = rng.choice(positive, size=max_events, replace=False, p=p)
    kept_w = weights[chosen] * (wsum / weights[chosen].sum())
    return X[chosen], kept_w, chosen


class _KDEPatternSubset(NamedTuple):
    """Per-pattern KDE reference set, cached by
    :meth:`ColourClassifier._kde_subset_for_pattern`. ``Xs``/``ws``/``Xb``/``wb``
    are the noise-injection-REPLICATED subset (see
    docs/superpowers/specs/2026-10-26-uncertainty-aware-noise-injection-design.md);
    ``event_id_s``/``event_id_b`` map each replica back to its original
    (capped, pre-replication) row, and ``n_orig_s``/``n_orig_b`` are how many
    original rows that was -- both needed by
    :func:`_grouped_effective_sample_size`. ``ess_ceiling_sig``/``ess_ceiling_bg``
    are computed from the CAPPED, PRE-replication weights, not ``ws``/``wb``.
    """
    Xs: np.ndarray
    ws: np.ndarray
    Xb: np.ndarray
    wb: np.ndarray
    ess_ceiling_sig: float
    ess_ceiling_bg: float
    event_id_s: np.ndarray
    event_id_b: np.ndarray
    n_orig_s: int
    n_orig_b: int


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
                 calibrate=True, ess_threshold=DEFAULT_ESS_THRESHOLD,
                 log_width=DEFAULT_LOG_WIDTH, kde_max_events=DEFAULT_KDE_MAX_EVENTS,
                 kde_min_ess=DEFAULT_KDE_MIN_ESS,
                 n_uncertainty_replicates=DEFAULT_N_UNCERTAINTY_REPLICATES, **hgb):
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
        self.ess_threshold = ess_threshold
        self.log_width = log_width
        self.kde_max_events = kde_max_events
        self.kde_min_ess = kde_min_ess
        self.n_uncertainty_replicates = n_uncertainty_replicates

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
        sigma_s = colour_feature_sigma_matrix(signal_events, self.keys, self.extra)
        sigma_b = colour_feature_sigma_matrix(background_events, self.keys, self.extra)

        ws = np.ones(len(Xs)) if signal_weights is None else np.asarray(signal_weights, float)
        wb = np.ones(len(Xb)) if background_weights is None else np.asarray(background_weights, float)
        if ws.sum() > 0:
            ws = ws * (wb.sum() / ws.sum())            # balance totals

        # Noise-injection replication for the TREE only, on the full balanced
        # population -- see Global Constraints. _store_kde_training_data below
        # keeps the ORIGINAL, un-replicated arrays; the KDE replicates lazily,
        # per pattern, after its own kde_max_events cap (Task 4).
        dims_all = np.arange(Xs.shape[1])
        Xs_rep, ws_rep, _ = _replicate_with_uncertainty(
            Xs, sigma_s, ws, dims_all, self.n_uncertainty_replicates)
        Xb_rep, wb_rep, _ = _replicate_with_uncertainty(
            Xb, sigma_b, wb, dims_all, self.n_uncertainty_replicates)

        X = np.vstack([Xs_rep, Xb_rep])
        y = np.r_[np.ones(len(Xs_rep)), np.zeros(len(Xb_rep))]
        w = np.r_[ws_rep, wb_rep]
        # Signal and background totals are balanced, so the trained posterior is at a
        # 0.5 prior; probability(prior=...) rescales it to any other prior. Uses the
        # ORIGINAL ws/wb -- a class total is unaffected by replication either way
        # (weights sum back exactly), but there is no reason to route this through
        # the replicated arrays.
        self.train_prior = float(ws.sum() / (ws.sum() + wb.sum()))

        params = dict(self.params)
        replicated = len(Xs_rep) != len(Xs) or len(Xb_rep) != len(Xb)
        if replicated and 'early_stopping' not in self.params:
            # Replication creates K near-duplicate (noise-perturbed) copies of
            # each original event. HistGradientBoostingClassifier's own
            # early_stopping='auto' heuristic decides from n_samples > 10,000 --
            # if applied to the REPLICATED count, it could enable (or disable)
            # early stopping purely as a side effect of replication having
            # happened, rather than reflecting the ORIGINAL population's own
            # scale. That would confound an n_uncertainty_replicates sweep (the
            # K=1 no-op baseline keeps sklearn's native 'auto' decision on the
            # ORIGINAL count; K>1 must match that SAME decision, not silently
            # change it) -- and for a population whose REPLICATED count crosses
            # 10,000 while its ORIGINAL count does not, enabling early stopping
            # would let near-duplicate replicas of the same event land on both
            # sides of its internal train/validation split, so validation loss
            # stops reflecting genuinely held-out data (see docs/superpowers/
            # specs/2026-10-26-uncertainty-aware-noise-injection-design.md).
            # Deciding from the ORIGINAL count, applied whether or not
            # replication occurs, fixes both: the decision never differs from
            # what K=1 would already give on the same data, and a population
            # that would never have triggered 'auto' pre-replication doesn't
            # spuriously trigger it now. HistGradientBoostingClassifier's own
            # default (20) similarly counts REPLICA rows for min_samples_leaf,
            # so a "leaf" of 20 replica rows is only 20/K real events once
            # replication multiplies the row count -- a capacity increase that
            # stacks on top of noise injection's own smoothing effect and would
            # likewise confound the sweep; scale it by K to hold per-event leaf
            # granularity constant.
            params['early_stopping'] = (len(Xs) + len(Xb)) > 10_000
            params.setdefault('min_samples_leaf', 20 * self.n_uncertainty_replicates)
        base = HistGradientBoostingClassifier(**params)
        if self.calibrate:
            self.clf = CalibratedClassifierCV(base, method='isotonic', cv=3).fit(X, y, sample_weight=w)
        else:
            self.clf = base.fit(X, y, sample_weight=w)

        # Threshold-setting calibrates against self.score() -- the SAME quantity
        # predict()/recovery_rate()/roc_curve() compare it to downstream. Using
        # raw self.clf.predict_proba(Xb) here instead (an earlier revision's
        # choice) would calibrate against a different quantity than the one the
        # threshold is actually applied to whenever real uncertainty data makes
        # score() average over replicas, silently drifting the realized FPR away
        # from target_fpr (see docs/superpowers/specs/
        # 2026-10-26-uncertainty-aware-noise-injection-design.md). score() already
        # reduces to predict_proba exactly when no uncertainty data exists, so
        # this is a true no-op in that case.
        sb = self.score(background_events)
        self.threshold = float(weighted_quantile(sb, wb, q=1.0 - self.target_fpr))
        self._store_kde_training_data(Xs, ws, Xb, wb, sigma_s, sigma_b)
        return self

    def _store_kde_training_data(self, Xs, ws, Xb, wb, sigma_s, sigma_b):
        """Retain the full (positive-weight) training events/weights/sigma +
        bandwidths for the local density-ratio blend in :meth:`probability`,
        and reset the per-pattern cache. ``ws``/``wb`` must already be the
        *balanced* weights computed earlier in :meth:`fit`, matching
        ``p_tree``'s own balanced-training-prior convention. ``sigma_s``/
        ``sigma_b`` are per-event measurement-uncertainty matrices, same row
        order and column order as ``Xs``/``Xb`` (see
        :func:`colour_feature_sigma_matrix`).

        ``kde_max_events`` subsampling does NOT happen here -- it happens
        lazily, per missing-data pattern, in :meth:`_kde_subset_for_pattern`.
        Real survey populations can have most events missing one or more
        features (different bands get a detection for different events);
        subsampling the full, NaN-mixed population *before* any pattern-
        specific finite-row filter wastes most of the budget on rows a
        given pattern discards anyway -- e.g. on BlueBEAR production data,
        capping at 4000 events before filtering left only ~65 rows (1.6%)
        usable for the all-features-finite pattern, versus ~2600 actually
        available. Filtering first, then capping per pattern, spends the
        budget only on rows that pattern can use. See docs/superpowers/
        specs/2026-10-26-kde-pattern-aware-subsampling-design.md.

        Noise-injection replication for the KDE also does NOT happen here --
        it happens lazily, per pattern, in :meth:`_kde_subset_for_pattern`,
        AFTER that pattern's own kde_max_events cap, for the same reason: a
        cap applied before replication would be spent on redundant copies of
        the same few original events instead of genuinely diverse ones. See
        docs/superpowers/specs/2026-10-26-uncertainty-aware-noise-injection-design.md.
        """
        pos_s = ws > 0
        pos_b = wb > 0
        self._kde_Xs, self._kde_ws = Xs[pos_s], ws[pos_s]
        self._kde_sigma_s = sigma_s[pos_s]
        self._kde_Xb, self._kde_wb = Xb[pos_b], wb[pos_b]
        self._kde_sigma_b = sigma_b[pos_b]
        pooled_X = np.vstack([self._kde_Xs, self._kde_Xb])
        pooled_w = np.r_[self._kde_ws, self._kde_wb]
        self._kde_bandwidths = _scott_bandwidths(pooled_X, pooled_w)
        self._kde_pattern_cache = {}

    def score(self, events):
        """Raw ranking score (calibrated posterior at the balanced training prior).

        Monotonic with :meth:`probability`, so it gives the same ordering; use it
        when you only need to rank.

        Each query event is noise-injection-replicated from its OWN ``*_err``
        fields (if present) and the tree's predictions are averaged back to one
        value per event -- symmetric treatment with how training data is
        handled (see docs/superpowers/specs/
        2026-10-26-uncertainty-aware-noise-injection-design.md). With no
        ``*_err`` data, this is numerically identical to a direct
        ``predict_proba`` call.
        """
        X, _ = colour_feature_matrix(events, self.keys, self.extra)
        sigma = colour_feature_sigma_matrix(events, self.keys, self.extra)
        dims_all = np.arange(X.shape[1])
        X_rep, _, event_id = _replicate_with_uncertainty(
            X, sigma, np.ones(len(X)), dims_all, self.n_uncertainty_replicates)
        p_rep = self.clf.predict_proba(X_rep)[:, 1]
        counts = np.bincount(event_id, minlength=len(X))
        sums = np.bincount(event_id, weights=p_rep, minlength=len(X))
        return sums / counts

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
        p_tree = np.clip(self.score(events), 1e-12, 1 - 1e-12)
        p = self._blend_with_local_density(events, p_tree)
        if prior is None:
            return p
        p0 = self.train_prior
        # likelihood ratio with the training prior divided out, re-applied at `prior`
        lr = (p / (1 - p)) * ((1 - p0) / p0)
        odds = lr * (prior / (1 - prior))
        return odds / (1 + odds)

    def _blend_with_local_density(self, events, p_tree):
        """Blend ``p_tree`` with the local KDE density ratio, per event,
        weighted by local training support (see
        docs/superpowers/specs/2026-10-25-local-density-posterior-blend-design.md).
        Falls back to ``p_tree`` unchanged wherever the KDE side is
        degenerate (no nearby training data of either class, or no finite
        feature columns for that event).

        Each query event is ALSO noise-injection-replicated from its own
        ``*_err`` fields (if present), and the blended result is averaged
        back over those replicas -- symmetric with how training data is
        handled (see docs/superpowers/specs/
        2026-10-26-uncertainty-aware-noise-injection-design.md). ``p_tree``
        arrives already averaged over its own replicas (see
        :meth:`score`); only the KDE-side evaluation is repeated here.
        """
        X, _ = colour_feature_matrix(events, self.keys, self.extra)
        sigma = colour_feature_sigma_matrix(events, self.keys, self.extra)
        out = np.array(p_tree, dtype=float, copy=True)
        for i in range(len(events)):
            x = X[i]
            pattern = _finite_pattern(x)
            dims = np.where(np.asarray(pattern))[0]
            if len(dims) == 0:
                continue
            subset = self._kde_subset_for_pattern(pattern, dims)
            x_rep, _, _ = _replicate_with_uncertainty(
                x[np.newaxis, :], sigma[i][np.newaxis, :], np.array([1.0]),
                dims, self.n_uncertainty_replicates)
            blended_vals = []
            for xr in x_rep:
                p_kde, ess_sig, ess_bg = _local_density_ratio(
                    xr, subset.Xs, subset.ws, subset.Xb, subset.wb, self._kde_bandwidths, dims,
                    self.kde_min_ess, ess_ceiling_sig=subset.ess_ceiling_sig,
                    ess_ceiling_bg=subset.ess_ceiling_bg, event_id_s=subset.event_id_s,
                    event_id_b=subset.event_id_b, n_orig_s=subset.n_orig_s, n_orig_b=subset.n_orig_b)
                if p_kde is None:
                    continue
                alpha = _blend_weight(ess_sig, ess_bg, self.ess_threshold, self.log_width)
                blended_vals.append(alpha * p_tree[i] + (1 - alpha) * p_kde)
            if blended_vals:
                out[i] = float(np.mean(blended_vals))
        return out

    def _kde_subset_for_pattern(self, pattern, dims):
        """Training rows finite over ``dims``, weighted-subsampled to at
        most ``kde_max_events`` (independently per class), then
        noise-injection-replicated from each surviving event's own
        measurement uncertainty -- cached per ``pattern`` on this instance
        (reset by :meth:`fit`) as a :class:`_KDEPatternSubset`.

        Subsampling happens HERE, after the finite-row filter, not in
        :meth:`_store_kde_training_data` -- so the ``kde_max_events`` budget
        is spent only on rows this specific pattern can actually use (see
        docs/superpowers/specs/2026-10-26-kde-pattern-aware-subsampling-design.md).
        Replication ALSO happens here, AFTER that cap -- replicating before
        capping would spend the budget on redundant copies of a few original
        events instead of genuinely diverse ones (see docs/superpowers/specs/
        2026-10-26-uncertainty-aware-noise-injection-design.md). The ESS
        ceilings are computed from the CAPPED, PRE-replication weights --
        replicating a point into K pieces must not make its own class look
        like it has K times more statistical support.
        """
        if pattern not in self._kde_pattern_cache:
            # Local import: importance_sampling -> contours -> classifier would be a cycle at module load.
            from .importance_sampling import effective_sample_size

            finite_s = np.all(np.isfinite(self._kde_Xs[:, dims]), axis=1)
            finite_b = np.all(np.isfinite(self._kde_Xb[:, dims]), axis=1)
            finite_s_idx = np.where(finite_s)[0]
            finite_b_idx = np.where(finite_b)[0]
            Xs_cap, ws_cap, chosen_s = _weighted_subsample(
                self._kde_Xs[finite_s], self._kde_ws[finite_s], self.kde_max_events)
            Xb_cap, wb_cap, chosen_b = _weighted_subsample(
                self._kde_Xb[finite_b], self._kde_wb[finite_b], self.kde_max_events)
            sigma_s_cap = self._kde_sigma_s[finite_s_idx[chosen_s]]
            sigma_b_cap = self._kde_sigma_b[finite_b_idx[chosen_b]]

            Xs_rep, ws_rep, event_id_s = _replicate_with_uncertainty(
                Xs_cap, sigma_s_cap, ws_cap, dims, self.n_uncertainty_replicates)
            Xb_rep, wb_rep, event_id_b = _replicate_with_uncertainty(
                Xb_cap, sigma_b_cap, wb_cap, dims, self.n_uncertainty_replicates)

            self._kde_pattern_cache[pattern] = _KDEPatternSubset(
                Xs_rep, ws_rep, Xb_rep, wb_rep,
                effective_sample_size(ws_cap), effective_sample_size(wb_cap),
                event_id_s, event_id_b, len(Xs_cap), len(Xb_cap))
        return self._kde_pattern_cache[pattern]

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
